#!/bin/python
import io
import os
import re
import json
import numpy as np
import argparse
import torch
import torchaudio
from transformers import Wav2Vec2Processor
from datasets import load_from_disk

# local import
from utils import make_dataset, load_from_json
from models.wav2vec2_model import AutoMDDModel, AutoDualMDDModel, AutoProtoMDDModel
from wer import calc_wer

# decoder
from decoder import *

from g2p_en import G2p
import re

# mosec
from mosec import Server, Worker, get_logger
from mosec.mixin import TypedMsgPackMixin
import mosec.errors
from typing import List, Any, BinaryIO, Union
import msgspec

logger = get_logger()

model_type = "small"
language = "en"
sampling_rate = 16000

class Request(msgspec.Struct):
    id: str
    binary: bytes
    prompt: str

class Validation(TypedMsgPackMixin, Worker):
    def forward(self, web_data: Request):
        if web_data.id != "1":
            raise mosec.errors.ValidationError("id not 1")
        
        return web_data.binary, web_data.prompt

class Preprocess(TypedMsgPackMixin, Worker):
    def __init__(self):
        self.g2p = G2p()

    def forward(self, data):
        data, word_text = data  # unpack data and prompt
        with io.BytesIO(data) as byte_io:
            audio_array = self.decode_audio(byte_io, sampling_rate)

        if len(audio_array) == 2 and audio_array.shape[1] == 2:
            # conbime the channel
            audio_array = np.mean(audio_array, 1)

        logger.info({"audio_array": audio_array, "sampling_rate": sampling_rate})

        # g2p 
        # TODO: add sil
        # words to phones
        phone_list = self.g2p(word_text) # ['HH', 'AW1', 'Z', ' ', 'DH', 'AH0', ' ', 'W', 'EH1', 'DH', 'ER0', ' ', 'T', 'AH0', 'D', 'EY1']

        # remove spaces
        new_phone_list = []
        word_boundaries = [] # [(word1, (start_idx, end_idx)), (word2 , (start_idx, end_idx), ...]

        word_list = word_text.split()
        word_idx = 0
        word_start_idx = 0
        word_end_idx = 0
        last_word = word_list[0]

        for idx, phone in enumerate(phone_list):
            if phone != " ":
                new_phone_list.append(phone)
            if phone == " " or idx == len(phone_list) - 1:
                word = word_list[word_idx]
                word_end_idx = len(new_phone_list)
                word_boundaries.append((word, (word_start_idx, word_end_idx)))
                word_idx += 1
                word_start_idx = len(new_phone_list)
        
        phone_text = " ".join(new_phone_list)
        # remove digits
        phone_text = re.sub("[0-9]", "", phone_text)
        # to lowercase
        prompt = phone_text.lower()

        logger.info('Preprocess finished')
        
        return audio_array, prompt, word_boundaries

    def decode_audio(
        self,
        input_file: Union[str, BinaryIO],
        sampling_rate: int = 16000,
        split_stereo: bool = False,
    ):
        """Decodes the audio.

        Args:
        input_file: Path to the input file or a file-like object.
        sampling_rate: Resample the audio to this sample rate.
        split_stereo: Return separate left and right channels.

        Returns:
        A float32 Torch Tensor.

        If `split_stereo` is enabled, the function returns a 2-tuple with the
        separated left and right channels.
        """

        waveform, audio_sf = torchaudio.load(input_file)  # waveform: channels X T

        if audio_sf != sampling_rate:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=audio_sf, new_freq=sampling_rate
            )
        if split_stereo:
            return waveform[0], waveform[1]

        return waveform.mean(0)


class Inference(TypedMsgPackMixin, Worker):
    def __init__(self):
        # device = "cpu" 
        # compute_type = "int8" # change to "int8" if low on GPU mem (may reduce accuracy)
        device = "cuda:1" 
        compute_type = "int8" # change to "int8" if low on GPU mem (may reduce accuracy)
        model_path = "exp/l2arctic/train_l2arctic_baseline_wav2vec2_large_lv60_timitft_prompt"

        train_conf_path = os.path.join(model_path, "train_conf.json")
        config_path = os.path.join(model_path, "config.pth")
        best_model_path = os.path.join(model_path, "best")

        # load train_args, model_args
        train_args, model_args = load_from_json(train_conf_path)
        
        # load config and model
        config = torch.load(config_path)
        self.device = device
        self.processor = Wav2Vec2Processor.from_pretrained(model_path)
        self.model = AutoMDDModel(model_args, config=config).to(device)
        
        self.model.load_state_dict(torch.load(best_model_path+"/pytorch_model.bin", map_location=device))
        vocab_dict = self.processor.tokenizer.get_vocab()
        sort_vocab = sorted((value, key) for (key,value) in vocab_dict.items())
        
        # Lower case ALL letters
        vocab = []
        for _, token in sort_vocab:
            if token in ['<s>', '</s>']: continue
            vocab.append(token)
        
        self.decoder = GreedyDecoder(vocab, blank_index=vocab.index(self.processor.tokenizer.pad_token))
        logger.info(f'Model loaded {model_type} {device} {compute_type}')

        # Warm up
        logger.info('Warm up finished')

    def forward(self, data):
        audio_array_list, prompt_list, word_boundaries_list = [], [], []

        for i in range(len(data)):
            audio_array_list.append(data[i][0])
            prompt_list.append(data[i][1])
            word_boundaries_list += data[i][2]
                
        audio_array = np.stack(audio_array_list)[0]
        prompt = " ".join(prompt_list)
        
        input_values = self.processor(audio_array, sampling_rate=16000).input_values[0]
        input_values = torch.tensor(input_values, device=self.device).unsqueeze(0)

        with self.processor.as_target_processor():
            prompt_ids = self.processor(prompt).input_ids
            prompt_ids = torch.tensor(prompt_ids, device=self.device).unsqueeze(0)
        
        output = self.model(input_values, prompts=prompt_ids, labels=None, return_dict=True)
        logits = output.logits

        logits_detection = output.logits_detection_ppl # NOTE: detect
        decoded_output, decoded_offsets = self.decoder.decode(logits)
        
        pred_phones = decoded_output[0][0]

        prompt = " ".join(re.sub(r'sil', '', prompt).split())
        pred_phones = " ".join(re.sub(r'sil', '', pred_phones).split())
        per_result = calc_wer(prompt.split(), pred_phones.split())
        print(f"word_boundaries_list: {word_boundaries_list}")
        print(per_result)

        eval_result = per_result.split("\n")[:-2][-1].split()[1:]
        eval_result_noins = [ er for er in eval_result if er != "I"]

        print(f"eval_result: {eval_result}")
        print(f"eval_result_noins: {eval_result_noins}")
        '''
         [
            {"word1":
                    [
                        {"phone1": "C"}, 
                        {"phone2": "D"}
                    ]
            }, 
            {"word2":
                    [
                        {"phone1": "S"}
                    ]
            } 
        ]
        '''
        dictate_list = []
        prompt_list = prompt.split()

        for word_info in word_boundaries_list:
            word, idx_info = word_info
            start_idx, end_idx = idx_info
            phones = prompt[start_idx:end_idx]
            phones_eval_list = [ {prompt_list[i]: eval_result_noins[i]} for i in range(start_idx,end_idx)]

            dictate_list.append({word: phones_eval_list})
        
        print(dictate_list)
        
        capt_dict = {"Dictate": dictate_list}
        
        print(capt_dict)
        logger.info("Inference Finished")
        
        return [json.dumps(capt_dict)]
    
    def serialize(self, data):
        return data.encode("utf-8")


if __name__ == "__main__":
    server = Server()
    server.append_worker(Validation, num=2)
    server.append_worker(Preprocess, num=2)
    server.append_worker(Inference, num=1, max_batch_size=8)
    server.run()
