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
import soundfile as sf

# local import
from utils import make_dataset, load_from_json
from local.models.wav2vec2_model import AutoMDDModel, AutoDualMDDModel, AutoProtoMDDModel
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

# llm
from local.llm.gpt_feedback import GenerateText

logger = get_logger()

class Request(msgspec.Struct):
    id: str
    binary: bytes
    prompt_words: str
    prompt_phones: str
    do_g2p: bool

class Validation(TypedMsgPackMixin, Worker):
    def forward(self, web_data: Request):
        if web_data.id != "1":
            raise mosec.errors.ValidationError("id not 1")
        
        return web_data.binary, web_data.prompt_words, web_data.prompt_phones, web_data.do_g2p

class Preprocess(TypedMsgPackMixin, Worker):
    def __init__(self):
        self.g2p = G2p()
        self.sampling_rate = 16000

    def forward(self, data):
        data, word_text, phn_text, do_g2p = data  # unpack data and prompt
        
        with io.BytesIO(data) as byte_io:
            audio_array = self.decode_audio(byte_io, self.sampling_rate)

        if len(audio_array) == 2 and audio_array.shape[1] == 2:
            # conbime the channel
            audio_array = np.mean(audio_array, 1)

        logger.debug(f"Get audio_array: {audio_array}, do_g2p : {do_g2p}, prompt : {word_text}")

        # g2p 
        # TODO: add sil
        # words to phone
        # ['HH', 'AW1', 'Z', ' ', 'DH', 'AH0', ' ', 'W', 'EH1', 'DH', 'ER0', ' ', 'T', 'AH0', 'D', 'EY1']
        if do_g2p:
            phone_list = self.g2p(word_text)
        else:
            phone_list = [ ' ' if phn == "SIL" else phn for phn in phn_text.split() ]

        # remove spaces
        new_phone_list = []
        word_boundaries = [] # [(word1, (start_idx, end_idx)), (word2 , (start_idx, end_idx), ...]

        word_list = word_text.split()
        word_idx = 0
        word_start_idx = 0
        word_end_idx = 0
        last_word = word_list[0]
        
        # Create word boundaries (without the SIL token)
        for idx, phone in enumerate(phone_list):
            if phone != " ":
                new_phone_list.append(phone)
            
            if phone == " " or idx == len(phone_list) - 1:
                word = word_list[word_idx]
                word_end_idx = len(new_phone_list)
                word_boundaries.append((word, (word_start_idx, word_end_idx)))
                word_idx += 1
                word_start_idx = len(new_phone_list)
        
        # Added silence
        phone_text = " ".join([ "SIL" if phn == " " else phn for phn in phone_list])
        phone_text = "SIL " + phone_text + " SIL"
        # remove digits
        phone_text = re.sub("[0-9]", "", phone_text)
        # to lowercase
        prompt = phone_text.lower()
        
        logger.info(f"Prompt: {prompt}, word_boundaries: {word_boundaries}")
        logger.info("Preprocess finished")
        
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
    def __init__(self, 
                 model_path="models/mdd/exp/l2arctic/train_l2arctic_baseline_wav2vec2_large_lv60_timitft_prompt", 
                 device="cuda:1", sampling_rate=16000):
        # device = "cpu" 
        # compute_type = "int8" # change to "int8" if low on GPU mem (may reduce accuracy)
        self.compute_type = "int8" # change to "int8" if low on GPU mem (may reduce accuracy)
        self.device = device
        self.sampling_rate = sampling_rate
        
        train_conf_path = os.path.join(model_path, "train_conf.json")
        config_path = os.path.join(model_path, "config.pth")
        best_model_path = os.path.join(model_path, "best")

        # load train_args, model_args
        train_args, model_args = load_from_json(train_conf_path)
        
        # load config and model
        config = torch.load(config_path)
        self.processor = Wav2Vec2Processor.from_pretrained(model_path)
        self.model = AutoMDDModel(model_args, config=config).to(device)    
        self.model.load_state_dict(torch.load(best_model_path+"/pytorch_model.bin", map_location=device))
        self.model.eval()
        
        vocab_dict = self.processor.tokenizer.get_vocab()
        sort_vocab = sorted((value, key) for (key,value) in vocab_dict.items())
        
        # Lower case ALL letters
        vocab = []
        for _, token in sort_vocab:
            if token in ['<s>', '</s>']: continue
            vocab.append(token)
        
        self.decoder = GreedyDecoder(vocab, blank_index=vocab.index(self.processor.tokenizer.pad_token))
        logger.info(f'Model loaded {self.device}')
        
        self.llm_feedback = GenerateText()

        # Warm up
        sample = "./example/Something_good_just_happened.wav"
        prompt_phones = "sil s ah m th ih ng sil g uh d sil jh ah s t sil hh ae p ah n d sil"
        audio_array, sample_rate = sf.read(sample)
        input_values = self.processor(audio_array, sampling_rate=self.sampling_rate).input_values[0]
        input_values = torch.tensor(input_values, device=self.device).unsqueeze(0)

        with self.processor.as_target_processor():
            prompt_ids = self.processor(prompt_phones).input_ids
            prompt_ids = torch.tensor(prompt_ids, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            output = self.model(input_values, prompts=prompt_ids, labels=None, return_dict=True)
            logits = output.logits
        logger.info('Warm up finished')
    

    def forward(self, data):
        audio_array_list, prompt_list, word_boundaries_list = [], [], []

        for i in range(len(data)):
            audio_array_list.append(data[i][0])
            prompt_list.append(data[i][1])
            word_boundaries_list += data[i][2]
                
        audio_array = np.stack(audio_array_list)[0]
        prompt_phones = " ".join(prompt_list)
        
        input_values = self.processor(audio_array, sampling_rate=self.sampling_rate).input_values[0]
        input_values = torch.tensor(input_values, device=self.device).unsqueeze(0)

        with self.processor.as_target_processor():
            prompt_ids = self.processor(prompt_phones).input_ids
            prompt_ids = torch.tensor(prompt_ids, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            output = self.model(
                        input_values=input_values, 
                        prompts=prompt_ids, 
                        labels=prompt_ids, 
                        return_dict=True
                     )
            # 1 * T * V
            logits = output.logits
            loss = output.loss
        
        logits_detection = output.logits_detection_ppl # NOTE: detect
        decoded_output, decoded_offsets = self.decoder.decode(logits)
        
        predict_phones = decoded_output[0][0]
        # llm feedback
        prompt_words = " ".join([ word for word, idx_info in word_boundaries_list])
        fb_result = self.llm_feedback.get_text(
                        prompt_words=prompt_words, 
                        prompt_phones=prompt_phones, 
                        predict_phones=predict_phones
                    )
        
        # remove SIL
        prompt_phones = " ".join(re.sub(r'sil', '', prompt_phones).split())
        predict_phones = " ".join(re.sub(r'sil', '', predict_phones).split())
        per_result = calc_wer(prompt_phones.split(), predict_phones.split())
        
        # compute gop (without silence)
        with self.processor.as_target_processor():
            prompt_ids = self.processor(prompt_phones).input_ids
            prompt_ids = torch.tensor(prompt_ids, device=self.device).unsqueeze(0)
        
        gop_sd_result = self.compute_gop(labels=prompt_ids, logits=logits, loss=loss)
        
        # ditate
        eval_result = per_result.split("\n")[:-2][-1].split()[1:]
        eval_sd_result = [ er for er in eval_result if er != "I"]

        dictate_list = []
        gop_list = []
        prompt_list = prompt_phones.split()

        for word_info in word_boundaries_list:
            word, idx_info = word_info
            start_idx, end_idx = idx_info
            phones = prompt_phones[start_idx:end_idx]
            phones_eval_list = [[prompt_list[i], eval_sd_result[i]] for i in range(start_idx,end_idx)]
            phones_gop_list = [[prompt_list[i], gop_sd_result[i]] for i in range(start_idx,end_idx)]

            dictate_list.append([word, phones_eval_list])
            gop_list.append([word, phones_gop_list])
        
        capt_dict = {
            "Dictate": dictate_list, 
            "Transcript": predict_phones, 
            "Feedback": fb_result,
            "GOP": gop_list
        }
        
        logger.debug(capt_dict)
        logger.info("Inference Finished")
        
        return [json.dumps(capt_dict)]
    
    def serialize(self, data):
        return data.encode("utf-8")
     
    def compute_gop(self, labels, logits, loss):
        # https://github.com/frank613/CTC-based-GOP/blob/main/generate-GOP/gop-ctc-af-SD.py
        gop_sd_result = []
        ll_self = loss.detach().cpu()
        labels = labels.clone().detach().cpu().squeeze(0)
        logits = logits.clone().detach().cpu().squeeze(0)
        
        pids = labels.tolist()
        num_labels = len(labels)
        post_mat = logits.softmax(dim=-1).transpose(0,1)
        
        ll_self = self.ctc_loss(post_mat, labels)
        
        for i, pid in enumerate(pids):
            ll_denom = self.ctc_loss_denom(post_mat, labels, i, blank=0)
            gop = -ll_self + ll_denom
            print(pid, gop)
            gop_sd_result.append(gop.item())
            
        return gop_sd_result
    
    def ctc_loss(self, params, seq, blank=0):
        """
        CTC loss function.
        params - n x m matrix of n-D probability distributions(softmax output) over m frames.
        seq - sequence of phone id's for given example.
        Returns objective, alphas and betas.
        """
        seqLen = seq.shape[0] # Length of label sequence (# phones)
        numphones = params.shape[0] # Number of labels
        L = 2*seqLen + 1 # Length of label sequence with blanks
        T = params.shape[1] # Length of utterance (time)
        
        alphas = torch.zeros((L,T)).double()
        
        # Initialize alphas and forward pass 
        alphas[0,0] = params[blank,0]
        alphas[1,0] = params[seq[0],0]
        
        for t in range(1,T):
            start = max(0,L-2*(T-t)) 
            end = min(2*t+2,L)
            for s in range(start,L):
                l = int((s-1)/2)
                # blank
                if s%2 == 0:
                    if s==0:
                        alphas[s,t] = alphas[s,t-1] * params[blank,t]
                    else:
                        alphas[s,t] = (alphas[s,t-1] + alphas[s-1,t-1]) * params[blank,t]
                # same label twice
                elif s == 1 or seq[l] == seq[l-1]:
                    alphas[s,t] = (alphas[s,t-1] + alphas[s-1,t-1]) * params[seq[l],t]
                else:
                    alphas[s,t] = (alphas[s,t-1] + alphas[s-1,t-1] + alphas[s-2,t-1]) \
                        * params[seq[l],t]
        
        llForward = torch.log(alphas[L-1, T-1] + alphas[L-2, T-1])
    
        return -llForward
        
    def ctc_loss_denom(self, params, seq, pos, blank=0):
        """
        CTC loss function.
        params - n x m matrix of n-D probability distributions(softmax output) over m frames.
        seq - sequence of phone id's for given example.
        Returns objective, alphas and betas.
        """
        seqLen = seq.shape[0] # Length of label sequence (# phones)
        numphones = params.shape[0] # Number of labels
        L = 2*seqLen + 1 # Length of label sequence with blanks
        T = params.shape[1] # Length of utterance (time)
        P = params.shape[0] # number of tokens    
    
        ## constraint mask for disabling insertion, and in this version we don't allow phoneme->blank but remains in the arbitrary state 
        mask_ins = torch.eye(P)
        #mask_ins[blank,:] = torch.ones(P)
        
        ##extend the tensor to save "arbitrary state"
        alphas = torch.zeros((L,T,P)).double()
    
        # Initialize alphas 
        if pos == 0:
            alphas[0,0,0] = params[blank,0]
            # in this version we don't allow initial with the second blank,  it will simplifiy the later computation especially for the “normalized” version
            alphas[2,0,0] = 0
            alphas[3,0,0] = params[seq[1],0]
            
            alphas[1,0] = params[0:,0]  #an list all tokens
            alphas[1,0,0] = 0  #can't stay at blank, same as the alphas[0,0,0]
        else:
            alphas[0,0,0] = params[blank,0]
            alphas[1,0,0] = params[seq[0],0]
            
        for t in range(1,T):
            if pos == seqLen-1: ###different from non-composed one, +1 below for possible skip paths at the final states
                lowest_state = L-2*(T-t+1)
            else:
                lowest_state = L-2*(T-t)
            start = max(0,lowest_state) 
            for s in range(start,L):
                l = int((s-1)/2)
                # blank
                if s%2 == 0:
                    if s==0:
                        alphas[s,t,0] = alphas[s,t-1,0] * params[blank,t]
                    else:
                        sum = self.check_arbitrary(alphas, s-1, t-1, [blank]) 
                        if sum: ## the first blank(for current t) after the arbitrary state,need to collect the probability from the additional dimension
                            ##in this version no need to remove for t=1, because state 2 is not initialized anymore
                            alphas[s,t,0] = (alphas[s,t-1,0] + sum) * params[blank,t]
                        else:
                            alphas[s,t,0] = (alphas[s,t-1,0] + alphas[s-1,t-1,0]) * params[blank,t]
                elif pos != l and pos != l-1:
                    if s == 1 or seq[l] == seq[l-1]:   # the first label or same label twice
                        alphas[s,t,0] = (alphas[s,t-1,0] + alphas[s-1,t-1,0]) * params[seq[l],t]
                    else:
                        alphas[s,t,0] = (alphas[s,t-1,0] + alphas[s-1,t-1,0] + alphas[s-2,t-1,0]) \
                            * params[seq[l],t]
                elif pos == l-1: #last token is the arbitrary token, need to collect the probability from the additional dimension, and also consider the skip paths
                    sum = self.check_arbitrary(alphas, s-2, t-1, [blank,seq[l]])  ##remove the entry of the blank and the  "l"th token in the last dim, because it's already covered in other terms with the same path
                    if l-2 < 0 or seq[l-2] == seq[l]: ###dont allow token skip
                        skip_token = 0
                    else:
                        skip_token = alphas[s-4,t-1,0] * params[seq[l],t]
                    ##we allow skip empty in this version, following the graph in the paper. No need to remove because the state[s-1] is blocked for skip.
                    ##also we allow skip empty because we removed the state 2 probability in intialization
                    skip_empty = alphas[s-3,t-1,0] * params[seq[l],t] 
                    alphas[s,t,0] = (alphas[s,t-1,0] + alphas[s-1,t-1,0] + sum ) * params[seq[l],t] + skip_empty + skip_token
                else: #current pos can be arbitrary tokens, use the boardcast scale product to allow all the paths       
                    if s == 1: #the blank pathes from the first term is already removed for t=0 at initial step, so we don't do it again
                        empty_prob = alphas[s-1,t-1,0] * params[:,t]
                        empty_prob[0] = 0
                        alphas[s,t,:] = (alphas[s,t-1,:].view(1,-1) * params[:,t].view(-1,1) * mask_ins).sum(-1) + empty_prob
                    else: #enterting wildcard state, for the skip path and empty path, we need to remove the pos of the same label and blank token to avoid duplicated paths. 
                        skip_prob = alphas[s-2,t-1,0] * params[:,t]  
                        skip_prob[seq[l-1]] = 0    
                        skip_prob[0] = 0    
    
                        empty_prob = alphas[s-1,t-1,0] * params[:,t]
                        empty_prob[0] = 0
    
                        alphas[s,t,:] = (alphas[s,t-1,:].view(1,-1) * params[:,t].view(-1,1) * mask_ins).sum(-1) + skip_prob + empty_prob
             
        sum = self.check_arbitrary(alphas, L-2, T-1, [blank])    
        if sum: # last label is arbitrary,  we need the skip path alphas[L-3, T-1, 0] + alphas[L-4, T-1, 0]
            llForward = torch.log(alphas[L-1, T-1, 0] + sum + alphas[L-3, T-1, 0] + alphas[L-4, T-1, 0])
        else:
            llForward = torch.log(alphas[L-1, T-1, 0] + alphas[L-2, T-1, 0])
    
        return -llForward
    
    def check_arbitrary(self, in_alphas, s, t, zero_pos=[]):
        '''
        ##check if the last dim > 0, return the sum of last dimension (collect the posterior for each possible tokens),the zero_pos is excluded in the sum.
        ##zero pos here starst from 0
        '''
        if in_alphas[s,t].sum() > 0:
            if len(zero_pos) != 0:
                mask = torch.ones_like(in_alphas[s,t])
                for i in zero_pos:
                    mask[i] = 0
                return sum(in_alphas[s,t][mask.bool()])
            else:
                return sum(in_alphas[s,t][:])
        else:
            return False


if __name__ == "__main__":
    server = Server()
    server.append_worker(Validation, num=2)
    server.append_worker(Preprocess, num=2)
    server.append_worker(Inference, num=1, max_batch_size=8)
    server.run()
