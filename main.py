import torch, soundfile, io
import numpy as np
from transformers import WhisperProcessor
from mosec import Server, Worker, get_logger
import mosec.errors
from typing import List, Any
import msgspec
from mosec.mixin import TypedMsgPackMixin

import whisperx
logger = get_logger()

class Request(msgspec.Struct):
    id: str
    binary: bytes

class Validation(TypedMsgPackMixin, Worker):
    def forward(self, data: Request):
        if data.id != "1":
            raise mosec.errors.ValidationError("id not 1")
        
        return data.binary

class Preprocess(TypedMsgPackMixin, Worker):
    def __init__(self):
        self.processor = WhisperProcessor.from_pretrained(
            "openai/whisper-medium", language="Chinese", task="transcribe"
            )

    def forward(self, data):
        with io.BytesIO(data) as byte_io:
            array, sampling_rate = soundfile.read(byte_io)
        if len(array) == 2 and array.shape[1] == 2:
            # conbime the channel
            array = np.mean(array, 1)
        logger.info({"array": array, "sampling_rate": sampling_rate})

        audio = {"array": array, "sampling_rate": sampling_rate}

        res = self.processor(
            audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt"
        )
        logger.info('Preprocess finished')
        return res.input_features


class Inference(TypedMsgPackMixin, Worker):
    def __init__(self):
        # device = "cpu" 
        # compute_type = "int8" # change to "int8" if low on GPU mem (may reduce accuracy)
        model_type = "small.en"
        # model_type = "medium"
        device = "cuda" 
        compute_type = "int8" # change to "int8" if low on GPU mem (may reduce accuracy)
        
        self.model = whisperx.load_model(model_type, device, compute_type=compute_type, language="en")
        logger.info(f'Model loaded {model_type} {device} {compute_type}')

        # Warm up
        _ = self.model.transcribe("1272-128104-0000.wav")
        logger.info('Warm up finished')

    def forward(self, data: List[Any]) -> List[Any]:
        logger.debug(len(data))
        data = torch.cat(data, 0)
        logger.debug(data.shape)

        ids = self.model.model.generate_segment_batched(data, self.model.tokenizer, self.model.options)
        logger.info("Inference Finished")
        
        return ids
    
    def serialize(self, data: str) -> bytes:
        return data.encode("utf-8")


if __name__ == "__main__":
    server = Server()
    server.append_worker(Validation, num=2)
    server.append_worker(Preprocess, num=2)
    server.append_worker(Inference, num=1, max_batch_size=8)
    server.run()
