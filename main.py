import torch, soundfile, io
import numpy as np
from transformers import WhisperProcessor
from mosec import Server, Worker, get_logger
from typing import List, Any

import whisperx
logger = get_logger()
class Preprocess(Worker):
    def __init__(self):
        self.processor = WhisperProcessor.from_pretrained(
            "openai/whisper-medium", language="Chinese", task="transcribe"
            )

    def deserialize(self, data: bytes) -> any:
        with io.BytesIO(data) as byte_io:
            array, sampling_rate = soundfile.read(byte_io)
        if len(array) == 2 and array.shape[1] == 2:
            # conbime the channel
            array = np.mean(array, 1)
        return {"array": array, "sampling_rate": sampling_rate}

    def forward(self, data):
        res = self.processor(
            data["array"], sampling_rate=data["sampling_rate"], return_tensors="pt"
        )
        logger.info('Preprocess finished')
        return res.input_features


class Inference(Worker):
    def __init__(self):
        device = "cpu" 
        compute_type = "int8" # change to "int8" if low on GPU mem (may reduce accuracy)
        self.model = whisperx.load_model("medium", device, compute_type=compute_type, language="zh")

        # Warm up
        _ = self.model.transcribe("example/1272-128104-0000.wav")
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

class Postprocess(Worker):
    def __init__(self):
        self.processor = WhisperProcessor.from_pretrained(
            "openai/whisper-medium", language="Chinese", task="transcribe"
            )

    def forward(self, data):
        logger.info('Postprocess begin')
        return data
    

    def serialize(self, data: str) -> bytes:
        return data.encode("utf-8")


if __name__ == "__main__":
    server = Server()
    server.append_worker(Preprocess, num=2)
    server.append_worker(Inference, num=2, max_batch_size=16, max_wait_time=1000)
    server.run()
