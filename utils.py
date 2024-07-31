import copy
import json
import torch
import random
import numpy as np
import torchaudio
from datasets import Dataset, Audio, concatenate_datasets
from typing import Any, Dict, List, Optional, Union
from transformers import Wav2Vec2Processor
from dataclasses import dataclass, field
from pysndfx import AudioEffectsChain

def load_from_json(data_json):
    with open(data_json) as jsonfile:
        x = json.load(jsonfile)
        return x

def save_to_json(data_dict, path):
    with open(path, "w") as write_file:
        json.dump(data_dict, write_file, indent=4)

def cal_class_weight(labels, n_classes, alpha=1.0, epsilon=1e-5):
    # input: list
    # output: 1-d tensor

    # normal re-weighting
    labels = np.array(labels)
    n_samples = len(labels)
    n_samples_each = np.zeros(n_classes)
    for c in range(n_classes):
        indices = np.where(labels == (c+1))
        n_samples_each[c] = len(labels[indices])
    #class_weight = np.power(n_samples, alpha) / n_classes * np.power(n_samples_each, alpha)
    class_weight = np.power(n_samples, alpha) / np.power(n_samples_each, alpha)
    class_weight[np.isinf(class_weight)] = 0

    '''
    # cefr-sp
    labels = np.array(labels)
    class_ratio = np.array([np.sum(labels == (c+1)) for c in range(n_classes)])
    class_ratio = class_ratio / np.sum(class_ratio)
    class_weight = np.power(class_ratio, alpha) / np.sum(
        np.power(class_ratio, alpha)) / (class_ratio + epsilon)
    '''

    return torch.Tensor(class_weight)

def speech_file_to_array_fn(path):
    speech_array, sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
    speech = resampler(speech_array).squeeze().numpy()
    return speech

def make_dataset(data_json, model_args, do_augment=False):

    print("Loading data from {} ...".format(data_json))
    data_dict = load_from_json(data_json)

    dataset = Dataset.from_dict(data_dict)

    # batch[audio] include path, array
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    if "audio_ref" in data_dict:
        dataset = dataset.cast_column("audio_ref", Audio(sampling_rate=16000))

    if "speed_perturb" in model_args and do_augment :
        # NOTE: 3 times of data
        augmented_dataset = copy.deepcopy(dataset)
        for speed_factor in model_args["speed_perturb"]:
            if speed_factor != 1:

                # NOTE: data augmentation
                def speed_pertubation(batch):

                    # NOTE: speed perturbation
                    AE = AudioEffectsChain()
                    AE = AE.speed(speed_factor)
                    fx = (AE)
                    batch["audio"]["array"] = fx(batch["audio"]["array"])
                    # rename id
                    batch["id"] = batch["id"] + "_sp{}".format(speed_factor)

                    return batch

                print("[INFO] speed perturbation for speed {} ...".format(speed_factor))
                augmented_dataset = concatenate_datasets([
                    augmented_dataset,
                    dataset.map(speed_pertubation, num_proc=4)
                ])

        return augmented_dataset

    else:
        return dataset

@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    task_type: str
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        # NOTE: extra features
        if self.task_type in ["mdd", "mdd-tts"]:
            # NOTE: prompt
            # This should be "input_ids", otherwise may cause error !
            prompt_features = [{"input_ids": feature["prompts"]} for feature in features]
            with self.processor.as_target_processor():
                prompts_batch = self.processor.pad(
                    prompt_features,
                    padding=self.padding,
                    max_length=self.max_length_labels,  # max len = 79 in tr and cv
                    pad_to_multiple_of=self.pad_to_multiple_of_labels,
                    return_tensors="pt",
                )
            # padding is 0
            batch["prompts"] = prompts_batch["input_ids"]

            # NOTE: detection targets
            # detection_target is either 0 or 1, so we add 1 to detection targets. 0 for padding
            detection_targets = [{"input_ids": [x + 1 for x in feature["detection_targets"]]} for feature in features]
            with self.processor.as_target_processor():
                detections_batch = self.processor.pad(
                    detection_targets,
                    padding=self.padding,
                    max_length=self.max_length_labels,
                    pad_to_multiple_of=self.pad_to_multiple_of_labels,
                    return_tensors="pt",
                )
            detection_targets = detections_batch["input_ids"].masked_fill(detections_batch.attention_mask.ne(1), -100)
            batch["detection_targets"] = detection_targets
            
            # NOTE: detection targets (ppl)
            # detection_target is either 0 or 1, so we add 1 to detection targets. 0 for padding
            detection_targets_ppl = [{"input_ids": [x + 1 for x in feature["detection_targets_ppl"]]} for feature in features]
            with self.processor.as_target_processor():
                detections_ppl_batch = self.processor.pad(
                    detection_targets_ppl,
                    padding=self.padding,
                    max_length=self.max_length_labels,
                    pad_to_multiple_of=self.pad_to_multiple_of_labels,
                    return_tensors="pt",
                )
            detection_targets_ppl = detections_ppl_batch["input_ids"].masked_fill(detections_ppl_batch.attention_mask.ne(1), -100)
            batch["detection_targets_ppl"] = detection_targets_ppl
        
        if self.task_type == "mdd-tts":
            input_features = [{"input_values": feature["input_values_ref"]} for feature in features]
            
            batch_ref = self.processor.pad(
                input_features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )
            batch["input_values_ref"] = batch_ref["input_values"]
            batch["attention_mask_ref"] = batch_ref["attention_mask"]

            # NOTE: duration
            np_bins = [0, 5.0, 8.0, 12.0, 50, 100]
            dur_features = [{"input_ids": np.digitize(feature["prompt_dur"], np_bins).tolist()} for feature in features]
            with self.processor.as_target_processor():
                dur_batch = self.processor.pad(
                    dur_features,
                    padding=self.padding,
                    max_length=self.max_length_labels,  # max len = 79 in tr and cv
                    pad_to_multiple_of=self.pad_to_multiple_of_labels,
                    return_tensors="pt",
                )
            # padding is 0
            batch["prompt_dur"] = dur_batch["input_ids"]

            prompt_supphones_features = [{"input_ids": feature["prompt_supphones"]} for feature in features]
            with self.processor.as_target_processor():
                prompt_supphones_batch = self.processor.pad(
                    prompt_supphones_features,
                    padding=self.padding,
                    max_length=self.max_length_labels,  # max len = 79 in tr and cv
                    pad_to_multiple_of=self.pad_to_multiple_of_labels,
                    return_tensors="pt",
                )
            # padding is 0
            batch["prompt_supphones"] = prompt_supphones_batch["input_ids"]
        
        return batch

### --- for sequence classification --- ###
def make_vocab(phn_dict, exp_dir, unk_token="err"):
    vocab_dict = {}
    vocab_dict["[PAD]"] = 0
    with open(phn_dict, "r") as rf:
        for phn in rf.readlines():
            phn = phn.strip()
            if phn == unk_token:
                continue
            else:
                vocab_dict[phn] = len(vocab_dict)
    vocab_dict[unk_token] = len(vocab_dict)
    #vocab_dict["|"] = len(vocab_dict) # word_delimiter
    with open(exp_dir + '/vocab.json', 'w') as vocab_file:
        json.dump(vocab_dict, vocab_file, indent=4)
    # return vocab size
    return vocab_dict, len(vocab_dict)

def remove_special_characters(batch):
    chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'
    batch["text"] = re.sub(chars_to_ignore_regex, '', batch["text"]).lower()
    return batch
