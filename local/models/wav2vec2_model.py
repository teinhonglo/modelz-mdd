import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, Union
from transformers.file_utils import ModelOutput
from transformers import AutoModel, AutoConfig
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.models.wav2vec2 import Wav2Vec2PreTrainedModel

import os

_HIDDEN_STATES_START_POSITION = 2

class PredictionHead(nn.Module):
    def __init__(self, config, model_args, input_dim=None, output_dim=None):
        super(PredictionHead, self).__init__()
        
        if input_dim is None:
            input_dim = config.hidden_size
        if output_dim is None:
            output_dim = config.num_labels

        self.dense = nn.Linear(input_dim, input_dim)

        self.dropout = nn.Dropout(config.final_dropout)
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x

class Mlp(nn.Module):
    def __init__(self, dim, hidden_dim=None, out_dim=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_dim = out_dim or dim
        hidden_dim = hidden_dim or dim
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class ConvFF(nn.Module):
    def __init__(self, dim, hidden_dim=None, out_dim=None, act_layer=nn.ReLU, drop=0., kernel_size=[3, 3]):
        super().__init__()
        out_dim = out_dim or dim
        hidden_dim = hidden_dim or dim
        self.fc1 = nn.Conv1d(
            dim, hidden_dim,
            kernel_size=kernel_size[0], padding=(kernel_size[0]-1)//2
        )
        self.act = act_layer()
        self.fc2 = nn.Conv1d(
            hidden_dim, out_dim,
            kernel_size=kernel_size[1], padding=(kernel_size[1]-1)//2
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = x.transpose(1,2)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = x.transpose(1,2)
        return x

class ConvBank(nn.Module):
    '''
    Implementation from: https://github.com/s3prl/s3prl/blob/master/s3prl/downstream/libri_phone/model.py
    '''
    def __init__(self, input_dim, output_class_num, kernels=[3, 5, 7], cnn_size=32, hidden_size=64, dropout=0.5, **kwargs):
        super(ConvBank, self).__init__()
        self.drop_p = dropout
        
        self.in_linear = nn.Linear(input_dim, hidden_size)
        latest_size = hidden_size

        # conv bank
        self.cnns = nn.ModuleList()
        assert len(kernels) > 0
        for kernel in kernels:
            self.cnns.append(nn.Conv1d(latest_size, cnn_size, kernel, padding=kernel//2))
        latest_size = cnn_size * len(kernels)

        self.out_linear = nn.Linear(latest_size, output_class_num)

    def forward(self, features):
        hidden = F.dropout(F.relu(self.in_linear(features)), p=self.drop_p)

        conv_feats = []
        hidden = hidden.transpose(1, 2).contiguous()
        for cnn in self.cnns:   
            conv_feats.append(cnn(hidden))
        hidden = torch.cat(conv_feats, dim=1).transpose(1, 2).contiguous()
        hidden = F.dropout(F.relu(hidden), p=self.drop_p)

        predicted = self.out_linear(hidden)
        return predicted

class TransformerEncoder(nn.Module):
    def __init__(self, dim, num_heads, proj_type, model_args, scale=4):
        super().__init__()
        # NOTE: attention
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads=num_heads, batch_first=True
        )
        # NOTE: mlp
        self.norm2 = nn.LayerNorm(dim)
        
        print("Transformer Encoder (Transform)", proj_type)
        if proj_type == "conv":
            # default
            self.mlp = ConvFF(dim=dim, hidden_dim=3*dim, drop=0.1)
        elif proj_type == "mlp":
            self.mlp = Mlp(dim=dim, hidden_dim=scale*dim, drop=0.1) 

    def forward(self, x, key_padding_mask=None):
        # attention
        residual = x
        x = self.norm1(x)
        x, attn_weight = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = residual + x

        # mlp
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x
        
        return x

class AudioTextEncoder(nn.Module):
    def __init__(self, model_args, config, mlp_ratio=4., use_mask=False, hidden_size=None):
        super(AudioTextEncoder, self).__init__()
        self.use_mask = use_mask
        if hidden_size is None:
            hidden_size = config.hidden_size
        #input_labels = config.num_labels + 1 if self.use_mask else config.num_labels
        input_labels = config.num_labels
        
        # NOTE: phoneme encoder - attention
        self.phone_emb = nn.Embedding(input_labels, hidden_size, padding_idx=0)
        
        # supphoneme encoder
        if "num_supphones" in model_args:
            self.use_supphone = True
            self.supphone_emb = nn.Embedding(
                model_args["num_supphones"], hidden_size, padding_idx=0
            )
        else:
            self.use_supphone = False

        # duration encoder
        self.use_dur = model_args["use_dur"]
        if self.use_dur:
            self.dur_emb = nn.Embedding(
                7, hidden_size, padding_idx=0
            )
        
        # NOTE: transformer encoder
        self.phone_enc_layers = model_args["phone_enc_layers"]
        
        self.phone_encs = nn.ModuleList([TransformerEncoder(dim=hidden_size, num_heads=1, proj_type=model_args["prompt_tfrenc_tf"], model_args=model_args) for _ in range(self.phone_enc_layers)])
       
               
        self.audio_text_fusion = model_args["audio_text_fusion"]
        print("Audio-Text Fusion", self.audio_text_fusion)
        if self.audio_text_fusion in ["cat", "add"]:
            # NOTE: cross attention: speech
            self.speech_text_attn = nn.MultiheadAttention(
                hidden_size, num_heads=1, batch_first=True
            )
            self.speech_attn = nn.MultiheadAttention(
                hidden_size, num_heads=1, batch_first=True
            )
        elif self.audio_text_fusion == "parallel":
            self.speech_text_attn = nn.MultiheadAttention(
                hidden_size, num_heads=1, batch_first=True
            )
        else:
            raise ValueError(f"Audio-Text {self.audio_text_fusion} is not supported.")

        if self.audio_text_fusion == "cat":
            self.proj_layer = nn.Linear(2 * hidden_size, hidden_size)
        
    def forward(self, x, phn, dur=None, supphone=None, x_mask=None):
        """
            Input:
                phn (batch, phn_T)
                x (batch, x_T, D)
            output:
                y (batch, x_T, D)
        """
        # (64, 79(max in tr, cv), 1024)
        phone_embed = self.phone_emb(phn.long()).float()
        key_padding_mask = (phn == 0)

        if self.use_supphone:
            supphone_embed = self.supphone_emb(supphone.long()).float()
            phone_embed = phone_embed + supphone_embed
        
        if self.use_dur:
            dur_embed = self.dur_emb(dur.long()).float()
            phone_embed = phone_embed + dur_attn

        # NOTE: F1: 60.55
        for phone_enc in self.phone_encs:
            phone_embed = phone_enc(phone_embed, key_padding_mask=None)
        
        # NOTE: cross-attention: speech
        # bool==true for ignore, shape (batch, phn_T)
        if self.audio_text_fusion in ["cat", "add"]:
            y, attn_weight = self.speech_text_attn(
                x, phone_embed, phone_embed, key_padding_mask=key_padding_mask
            )

            # add fusion
            if self.audio_text_fusion == "cat":
                y = torch.cat([y, x], dim=-1)
                y = self.proj_layer(y)
            elif self.audio_text_fusion == "add":
                y = y + x

            y_sa, attn_weight = self.speech_attn(y, y, y, key_padding_mask=x_mask)
            y = y + y_sa

        elif self.audio_text_fusion == "parallel":
            x_mask = (
                x_mask if x_mask is not None else torch.zeros_like(x, dtype=torch.long)
            )

            key_padding_mask = torch.cat((x_mask, key_padding_mask), dim=1)   

            x_c = torch.cat((x, phone_embed), dim=1)
        
            y, attn_weight = self.speech_text_attn(
                x_c, x_c, x_c, key_padding_mask=key_padding_mask
            )

            phone_embed = phone_embed + y[:, x.shape[1]:, :]
            y = y[:, :x.shape[1], :]
            y = y + x
        
        return y, phone_embed

def kldiv_lsm_ctc(logits, ylens):
    # code from https://github.com/hirofumi0810/neural_sp/blob/master/neural_sp/models/criterion.py#L110
    """Compute KL divergence loss for label smoothing of CTC and Transducer models.
    Args:
        logits (FloatTensor): `[B, T, vocab]`
        ylens (IntTensor): `[B]`
    Returns:
        loss_mean (FloatTensor): `[1]`
    """
    bs, _, vocab = logits.size()

    log_uniform = logits.new_zeros(logits.size()).fill_(math.log(1 / (vocab - 1)))
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    loss = torch.mul(probs, log_probs - log_uniform)
    loss_mean = sum([loss[b, :ylens[b], :].sum() for b in range(bs)]) / ylens.sum()
    return loss_mean

@dataclass
class CausalLMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    logits_detection: torch.FloatTensor = None # NOTE: detect
    logits_detection_ppl: torch.FloatTensor = None # NOTE: detect_ppl
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

def check_model_args(model_args):
    if "phone_enc_layers" not in model_args:
        model_args["phone_enc_layers"] = 1

    if "audio_text_fusion" not in model_args:
        model_args["audio_text_fusion"] = "add"
    
    if "prompt_tfrenc_tf" not in model_args:
        model_args["prompt_tfrenc_tf"] = "conv"
    
    if "use_dur" not in model_args:
        model_args["use_dur"] = False

    if "pred_head" not in model_args:
        model_args["pred_head"] = "default"

    if 'use_mask' not in model_args:
        model_args["use_mask"] = None
    
    if 'use_mlm' not in model_args:
        model_args["use_mlm"] = None
    
    if 'use_prompt' not in model_args:
        model_args["use_prompt"] = None

    return model_args

class AutoMDDModel(nn.Module):

    def __init__(self, model_args, class_weight=None, config=None, pretrained=False):
        super(AutoMDDModel, self).__init__()
        # model_args
        model_args = check_model_args(model_args)
        self.model_args = model_args          

        # config
        if config is None:
            self.config = AutoConfig.from_pretrained(
                model_args["model_path"],
                num_labels=model_args["num_labels"],
                problem_type=model_args["problem_type"],
                ctc_zero_infinity=model_args["ctc_zero_infinity"],
                final_dropout=model_args["final_dropout"],
                layerdrop=model_args["layerdrop"],
                activation_dropout=model_args["activation_dropout"],
                attention_dropout=model_args["attention_dropout"],
                mask_time_prob=model_args["mask_time_prob"],
                mask_time_length=model_args["mask_time_length"],
                mask_feature_prob=model_args["mask_feature_prob"],
                mask_feature_length=model_args["mask_feature_length"]
            )
        else:
            self.config = config 

        # NOTE: Model-related
        # audio encoder
        if pretrained:
            self.model = AutoModel.from_pretrained(model_args["model_path"], config=self.config)
        else:
            self.model = AutoModel.from_config(self.config)

        # audio-text decoder
        # mlm
        self.use_mask = model_args["use_mask"]
        self.use_mlm = model_args["use_mlm"]

        if self.use_mask or self.use_mlm:
            self.mask_token_id = self.config.num_labels - 1
        
        # prompt
        self.use_prompt = model_args["use_prompt"]
        self.audio_text_fusion = model_args["audio_text_fusion"]
                
        if self.use_prompt:
            self.audio_text_enc = AudioTextEncoder(model_args, self.config, use_mask=self.use_mask)

        # layer sum
        num_layers = self.config.num_hidden_layers + 1  # transformer layers + input embeddings
        self.config.use_weighted_layer_sum = True if "use_weighted_layer_sum" in model_args else False
        self.config.use_nth_layer = model_args["use_nth_layer"] if "use_nth_layer" in model_args else -1
        
        if self.config.use_weighted_layer_sum:
            print("Use layer sum", model_args["use_weighted_layer_sum"])
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)

        # prediction head
        if model_args["pred_head"] == "default":
            print("Use Default PredHead")
            self.prediction_head = PredictionHead(config=self.config, model_args=self.model_args)
        elif model_args["pred_head"] == "conv":
            print("Use Conv PredHead")
            self.prediction_head = ConvBank(
                input_dim=self.config.hidden_size,
                output_class_num=self.config.num_labels,
                kernels=[3, 5, 7], cnn_size=32, hidden_size=64, dropout=self.config.final_dropout,
            )
        else:
            raise ValueError("Unrecognized pred_head")

        # NOTE: Loss-related

        # detection loss
        self.use_detection_loss = model_args["use_detection_loss"] if 'use_detection_loss' in model_args else "no"
        
        if self.use_detection_loss == "ppl":
            self.prediction_head_ppl_detection = PredictionHead(
                                                            config=self.config, 
                                                            model_args=self.model_args, 
                                                            input_dim=self.config.hidden_size, 
                                                            output_dim=3
                                                        )
        elif self.use_detection_loss == "prompt":
            self.prediction_head_prompt_detection = nn.Linear(self.config.hidden_size, 1)
            self.prompt_detection_loss = nn.BCEWithLogitsLoss(reduction='none')

        # NOTE: Loss-related
        # focal loss
        self.use_focal_loss = model_args["use_focal_loss"] if 'use_focal_loss' in model_args else None
        if self.use_focal_loss:
            self.config.ctc_loss_reduction = "none"
        
        # label smoothing
        self.lsm_prob = model_args["label_smoothing"] if 'label_smoothing' in model_args else 0

        # NOTE: other
        self.num_labels = self.config.num_labels
        self.class_weight = class_weight
        self.model.gradient_checkpointing_enable()

        # NOTE: freeze feature encoder
        self.freeze_feature_extractor()
        # freeze lowest k layers
        if "freeze_k_layers" in model_args:
            self.freeze_k_layers(model_args["freeze_k_layers"])
        # freeze all
        if "freeze_all" in model_args and model_args["freeze_all"]:
            self.freeze_all()
        if "mms" in model_args["model_path"]:
            self.freeze_mms()

    def mask_uniform(self, ys_pad, mask_token_id, num_labels, pad_token_id=0, ignore_label=-100):
        # modified from: https://github.com/espnet/espnet/blob/f841f1d964204a3d90a5db15d5651d89f26c3096/espnet/nets/pytorch_backend/maskctc/add_mask_token.py#L13-39
        # in this work, ys = prompts

        ys = [y[y != pad_token_id] for y in ys_pad]  # non-padded ys
        ys_out = torch.LongTensor(ys_pad.shape).fill_(ignore_label).to(ys_pad.device)

        for i in range(len(ys)): # len(ys) = batch size   
            
            # NOTE: only mask 0.2
            probability_matrix = torch.full(ys[i].shape, 0.05)
            idx_masked = torch.bernoulli(probability_matrix).bool()
            ys_out[i][idx_masked.nonzero()] = ys[i][idx_masked.nonzero()]
            ys_pad[i][idx_masked.nonzero()] = mask_token_id

        return ys_pad, ys_out

    def mask_accuracy(self, pad_logits, pad_targets, ignore_label=-100):
        pad_pred = pad_logits.argmax(2)
        mask = pad_targets != ignore_label
        numerator = torch.sum(
            pad_pred.masked_select(mask) == pad_targets.masked_select(mask)
        )
        denominator = torch.sum(mask)
        return float(numerator) / float(denominator)

    def freeze_k_layers(self, k):
        for name, parameter in self.model.encoder.layers[:k].named_parameters():
            if not "adapter" in name:
                parameter.requires_grad = False

    def freeze_feature_extractor(self):
        self.model.feature_extractor._freeze_parameters()

    def freeze_mms(self):
        self.model.init_adapter_layers()
        for param in self.model.parameters():
            param.requires_grad = False
        adapter_weights = self.model._get_adapters()
        for param in adapter_weights.values():
            param.requires_grad = True

    def freeze_all(self):
        for name, parameter in self.model.named_parameters():
            if not "adapter" in name:
                parameter.requires_grad = False

    def load_pretrained_wav2vec2(self, state_dict):
        self.model.load_state_dict(state_dict)
        self.model.feature_extractor._freeze_parameters()

    def _get_feat_extract_output_lengths(
        self, input_lengths: Union[torch.LongTensor, int]
    ):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return torch.div(input_length - kernel_size, stride, rounding_mode="floor") + 1

        for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        return input_lengths

    def forward(self,
        input_values,
        input_values_ref=None,
        attention_mask=None,
        attention_mask_ref=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
        prompts=None,
        prompt_supphones=None,
        detection_targets=None,
        detection_targets_ppl=None,
        prompt_dur=None,
    ):
        output_hidden_states = True if self.config.use_weighted_layer_sum or self.config.use_nth_layer != -1 else output_hidden_states
        
        # 1. NOTE: audio encoder
        outputs = self.model(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
       
        if self.config.use_weighted_layer_sum:
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
            hidden_states = torch.stack(hidden_states, dim=1)
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
            # NOTE: Dirty Codes for output_hidden_states
            output_hidden_states = False
            outputs = self.model(
                input_values,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif self.config.use_nth_layer != -1:
            hidden_states = outputs.hidden_states[self.config.use_nth_layer]
            # NOTE: Dirty Codes for output_hidden_states
            output_hidden_states = False
            outputs = self.model(
                input_values,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            hidden_states = outputs[0]
        
        batch_size = hidden_states.shape[0]
        # retrieve loss input_lengths from attention_mask
        attention_mask = (
            attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
        )
        # NOTE: main speech
        input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)
        max_length = input_lengths.max() 
        batch_size=input_lengths.shape[0]
        mask = torch.zeros((batch_size, max_length), dtype=torch.bool, device=input_lengths.device)
        for i in range(batch_size):
            mask[i, :input_lengths[i]] = True

        mask = ~mask
        
        if self.audio_text_fusion != "parallel":
            mask = None
        
        # 2. NOTE: audio-text encoder
        # labels, prompts: (batch, 74)
        if prompts is not None and self.use_prompt:
            if self.use_mask and labels is not None:
                prompts_mask, targets_mlm = self.mask_uniform(
                    prompts, mask_token_id=self.mask_token_id, num_labels=self.config.num_labels
                )
                hidden_states, phone_embeds = self.audio_text_enc(hidden_states, prompts_mask, prompt_dur, prompt_supphones, mask)
            else:
                hidden_states, phone_embeds = self.audio_text_enc(hidden_states, prompts, prompt_dur, prompt_supphones, mask)
        
        # logits: B x T x Class
        logits = self.prediction_head(hidden_states)
        
        if self.use_detection_loss == "ppl":
            logits_detection_ppl = self.prediction_head_ppl_detection(hidden_states)
            logits_detection = None
        elif self.use_detection_loss == "prompt":
            logits_detection = self.prediction_head_prompt_detection(phone_embeds)
            logits_detection_ppl = None
        else:
            logits_detection = None
            logits_detection_ppl = None

        loss = None
        
        if labels is not None:
            if labels.max() >= self.config.num_labels:
                raise ValueError(f"Label values must be <= vocab_size: {self.config.num_labels}")

            # retrieve loss input_lengths from attention_mask
            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            )
            # NOTE: main speech
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            # NOTE: ctc_loss doesn't support fp16
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)

            with torch.backends.cudnn.flags(enabled=False):
                loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,    # default: sum, use_focal_loss=none
                    zero_infinity=self.config.ctc_zero_infinity, # default: false
                )
             
            if self.use_detection_loss == "ppl":
                if detection_targets_ppl.max() >= 3:
                    raise ValueError(f"Detection Label values must be < 3")
                
                labels_mask_detection = detection_targets_ppl >= 0
                target_lengths_detection = labels_mask_detection.sum(-1)
                flattened_targets_detection = detection_targets_ppl.masked_select(labels_mask_detection)
                
                log_probs_detection = nn.functional.log_softmax(logits_detection_ppl, dim=-1, dtype=torch.float32).transpose(0, 1)
                
                with torch.backends.cudnn.flags(enabled=False):
                    loss_detection = nn.functional.ctc_loss(
                        log_probs_detection,
                        flattened_targets_detection,
                        input_lengths,
                        target_lengths_detection,
                        blank=self.config.pad_token_id,
                        reduction=self.config.ctc_loss_reduction,    # default: sum
                        zero_infinity=self.config.ctc_zero_infinity, # default: false
                    )
            elif self.use_detection_loss == "prompt":
                detection_mask = detection_targets != -100
                detection_targets[detection_targets != -100] -= 1

                loss_prompt_detection = self.prompt_detection_loss(logits_detection.squeeze(-1), detection_targets.float())
                masked_loss_prompt_detection = loss_prompt_detection * detection_mask
                loss_prompt_detection = masked_loss_prompt_detection.sum() / loss_prompt_detection.sum()
            
            # NOTE: focal loss
            if self.use_focal_loss:
                alpha, gamma = 0.25, 4.0 
                loss = alpha * ((1 - torch.exp(-loss)) ** gamma) * loss
                loss = loss.mean()

            # NOTE: batch ctc loss is better
            if self.config.ctc_loss_reduction == "sum":
                loss /= batch_size
            
            if self.use_detection_loss == "ppl": 
                loss_detection /= batch_size
                loss = 1. * loss + 1. * loss_detection
            elif self.use_detection_loss == "prompt": 
                loss = 1. * loss + 1. * loss_prompt_detection
                
            # NOTE: label smoothing
            if self.lsm_prob > 0:
                loss =  (1 - self.lsm_prob) * loss \
                    + self.lsm_prob * kldiv_lsm_ctc(logits, target_lengths)

        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss,
            logits=logits,
            logits_detection_ppl=logits_detection_ppl,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )

class AutoDualMDDModel(AutoMDDModel):

    def __init__(self, model_args, class_weight=None, config=None, pretrained=False):
        super(AutoDualMDDModel, self).__init__( model_args=model_args, 
                                                class_weight=class_weight, 
                                                config=config, 
                                                pretrained=pretrained)
        # The same as AutoMDDModel    
        self.use_gate = model_args["use_gate"] if 'use_gate' in model_args else None

        if self.use_gate:
            self.w1 = nn.Linear(2 * self.config.hidden_size, self.config.hidden_size)
        else:
            self.attn_ref = nn.MultiheadAttention(
                self.config.hidden_size, num_heads=1, batch_first=True
            )
            

    def forward(self,
        input_values,
        input_values_ref=None,
        attention_mask=None,
        attention_mask_ref=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
        prompts=None,
        detection_targets=None,
        detection_targets_ppl=None,
    ):
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states
        
        # 1. NOTE: audio encoder
        outputs = self.model(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        outputs_ref = self.model(
            input_values_ref,
            attention_mask=attention_mask_ref,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_states = outputs[0]
        hidden_states_ref = outputs_ref[0]
        batch_size = hidden_states.shape[0]

        # NOTE: ref speech
        attention_mask_ref = (
            attention_mask_ref if attention_mask_ref is not None else torch.ones_like(input_values_ref, dtype=torch.long)
        )
        
        input_lengths_ref = self._get_feat_extract_output_lengths(attention_mask_ref.sum(-1)).to(torch.long)
        max_length = input_lengths_ref.max() 
        mask_ref = torch.zeros((batch_size, max_length), dtype=torch.bool, device=input_lengths_ref.device)

        for i in range(batch_size):
            mask_ref[i, :input_lengths_ref[i]] = True
        mask_ref = ~mask_ref

        # NOTE: main speech
        #attention_mask = (
        #    attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
        #)
        #input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)
        #max_length = input_lengths.max() 
        #mask = torch.zeros((batch_size, max_length), dtype=torch.bool, device=input_lengths.device)
        #hidden_states_self_attn, _ = self.attn_self(hidden_states, hidden_states, hidden_states, key_padding_mask=mask)

        # https://github.com/vocaliodmiku/wav2vec2mdd-Text/blob/master/fairseq/fairseq/models/wav2vec/wav2vec2_sigmoid.py#L601
        if self.use_gate:
            gate = torch.sigmoid(self.W1(torch.cat((hidden_states, hidden_states_ref), dim=-1)))
            hidden_states = (1 - gate) * hidden_states + gate * hidden_states_ref
        else:
            hidden_states_ref_attn, _ = self.attn_ref(hidden_states, hidden_states_ref, hidden_states_ref, key_padding_mask=mask_ref)
            hidden_states = hidden_states + hidden_states_ref_attn
        
        # NOTE: main speech
        attention_mask = (
            attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
        )
        input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)
        max_length = input_lengths.max() 
        mask = torch.zeros((batch_size, max_length), dtype=torch.bool, device=input_lengths.device)

        for i in range(batch_size):
            mask[i, :input_lengths[i]] = True
        mask = ~mask
        
        hidden_states[mask] = 0.
        
        mask = None
        # 2. NOTE: audio-text encoder
        # labels, prompts: (batch, 74)
        if prompts is not None and self.use_prompt:
            if self.use_mask and labels is not None:
                prompts_mask, targets_mlm = self.mask_uniform(
                    prompts, mask_token_id=self.mask_token_id, num_labels=self.config.num_labels
                )
                hidden_states, phone_embeds = self.audio_text_enc(hidden_states, prompts_mask, mask)
            else:
                hidden_states, phone_embeds = self.audio_text_enc(hidden_states, prompts, mask)

        # logits: B x T x Class
        logits = self.prediction_head(hidden_states)
        
        if self.use_detection_loss:
            logits_detection_ppl = self.prediction_head_detect(hidden_states)
            logits_detection_ppl = logits_detection_ppl
        else:
            logits_detection_ppl = None

        loss = None
        
        if labels is not None:
            if labels.max() >= self.config.num_labels:
                raise ValueError(f"Label values must be <= vocab_size: {self.config.num_labels}")

            # retrieve loss input_lengths from attention_mask
            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            )
            # NOTE: main speech
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            # NOTE: ctc_loss doesn't support fp16
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)

            with torch.backends.cudnn.flags(enabled=False):
                loss = nn.functional.ctc_loss(
                        log_probs,
                        flattened_targets,
                        input_lengths,
                        target_lengths,
                        blank=self.config.pad_token_id,
                        reduction=self.config.ctc_loss_reduction,    # default: sum, use_focal_loss=none
                        zero_infinity=self.config.ctc_zero_infinity, # default: false
                    )
             
            if self.use_detection_loss:
                if detection_targets_ppl.max() >= 3:
                    raise ValueError(f"Detection Label values must be < 3")
                
                labels_mask_detect = detection_targets_ppl >= 0
                target_lengths_detect = labels_mask_detect.sum(-1)
                flattened_targets_detect = detection_targets_ppl.masked_select(labels_mask_detect)
                
                log_probs_detect = nn.functional.log_softmax(logits_detection_ppl, dim=-1, dtype=torch.float32).transpose(0, 1)
                
                with torch.backends.cudnn.flags(enabled=False):
                    loss_detect = nn.functional.ctc_loss(
                        log_probs_detect,
                        flattened_targets_detect,
                        input_lengths,
                        target_lengths_detect,
                        blank=self.config.pad_token_id,
                        reduction=self.config.ctc_loss_reduction,    # default: sum
                        zero_infinity=self.config.ctc_zero_infinity, # default: false
                    )
            
            # NOTE: focal loss
            if self.use_focal_loss:
                alpha, gamma = 0.25, 4.0 
                loss = alpha * ((1 - torch.exp(-loss)) ** gamma) * loss
                loss = loss.mean()

            # NOTE: batch ctc loss is better
            if self.config.ctc_loss_reduction == "sum":
                loss /= batch_size
            
                if self.use_detection_loss: 
                    loss_detect /= batch_size
                    loss = 1. * loss + 1. * loss_detect
                
            # NOTE: label smoothing
            if self.lsm_prob > 0:
                loss =  (1 - self.lsm_prob) * loss \
                    + self.lsm_prob * kldiv_lsm_ctc(logits, target_lengths)

        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss,
            logits=logits,
            logits_detection_ppl=logits_detection_ppl,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )

class AutoProtoMDDModel(AutoMDDModel):

    def __init__(self, model_args, class_weight=None, config=None, pretrained=False):
        super(AutoProtoMDDModel, self).__init__( model_args=model_args, 
                                                class_weight=class_weight, 
                                                config=config, 
                                                pretrained=pretrained)
        # model_args
        self.model_args = model_args
        if "phone_enc_layers" not in model_args:
            model_args["phone_enc_layers"] = 1

        if "audio_text_fusion" not in model_args:
            model_args["audio_text_fusion"] = "add"
        
        if "prompt_tfrenc_tf" not in model_args:
            model_args["prompt_tfrenc_tf"] = "conv"

        # config
        if config is None:
            self.config = AutoConfig.from_pretrained(
                model_args["model_path"],
                num_labels=model_args["num_labels"],
                problem_type=model_args["problem_type"],
                ctc_zero_infinity=model_args["ctc_zero_infinity"],
                final_dropout=model_args["final_dropout"],
                layerdrop=model_args["layerdrop"],
                activation_dropout=model_args["activation_dropout"],
                attention_dropout=model_args["attention_dropout"],
                mask_time_prob=model_args["mask_time_prob"],
                mask_time_length=model_args["mask_time_length"],
                mask_feature_prob=model_args["mask_feature_prob"],
                mask_feature_length=model_args["mask_feature_length"]
            )
        else:
            self.config = config

        # NOTE: model
        if pretrained:
            self.model = AutoModel.from_pretrained(model_args["model_path"], config=self.config)
        else:
            self.model = AutoModel.from_config(self.config)
        # final dropout
        self.dropout = nn.Dropout(self.config.final_dropout)

        # NOTE: protoypes
        self.num_prototypes = model_args["num_prototypes"]
        self.dist = model_args["dist"]
        if self.dist == "scos":
            self.w = nn.Parameter(torch.tensor(10.0))
            self.b = nn.Parameter(torch.tensor(-5.0))
        self.prototype = nn.Embedding(self.config.num_labels * self.num_prototypes, self.config.hidden_size)
        nn.init.xavier_normal_(self.prototype.weight)  # Xavier initialization
        nn.init.orthogonal_(self.prototype.weight)  # Make prototype vectors orthogonal

        # audio-text decoder
        # mlm
        self.use_mask = model_args["use_mask"] if 'use_mask' in model_args else None
        self.use_mlm = model_args["use_mlm"] if 'use_mlm' in model_args else None

        if self.use_mask or self.use_mlm:
            self.mask_token_id = self.config.num_labels - 1
        
        # prompt
        self.use_prompt = model_args["use_prompt"] if 'use_prompt' in model_args else None
                
        if self.use_prompt:
            self.audio_text_enc = AudioTextEncoder(model_args, self.config, use_mask=self.use_mask)

        # layer sum
        num_layers = self.config.num_hidden_layers + 1  # transformer layers + input embeddings
        self.config.use_weighted_layer_sum = True if "use_weighted_layer_sum" in model_args else False
        
        if self.config.use_weighted_layer_sum:
            print("Use layer sum", model_args["use_weighted_layer_sum"])
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)

        # prediction head
        if model_args["pred_head"] == "conv":
            print("[MDD] Use pred PredictionHeadConv")
            self.prediction_head = PredictionHeadConv(config=self.config, model_args=self.model_args)
        else:
            self.prediction_head = PredictionHead(config=self.config, model_args=self.model_args)

        self.use_detection_loss = model_args["use_detection_loss"] if 'use_detection_loss' in model_args else None
        
        if self.use_detection_loss:
            self.prediction_head_detect = PredictionHead(config=self.config, model_args=self.model_args, output_dim=3)
            
        # NOTE: Loss-related
        # focal loss
        self.use_focal_loss = model_args["use_focal_loss"] if 'use_focal_loss' in model_args else None
        if self.use_focal_loss:
            self.config.ctc_loss_reduction = "none"

        # NOTE: label smoothing
        self.lsm_prob = model_args["label_smoothing"] if 'label_smoothing' in model_args else 0

        # NOTE: other
        self.num_labels = self.config.num_labels
        self.class_weight = class_weight
        self.model.gradient_checkpointing_enable()

        # NOTE: freeze
        self.freeze_feature_extractor()
            
        # freeze lowest k layers
        if "freeze_k_layers" in model_args:
            self.freeze_k_layers(model_args["freeze_k_layers"])
        # freeze all
        if "freeze_all" in model_args and model_args["freeze_all"]:
            self.freeze_all()
        if "mms" in model_args["model_path"]:
            self.freeze_mms()
    
    def negative_sed(self, a, b):
        ''' negative square euclidean distance
        - input
            a (hidden states): batch x T x D
            b (prototypes): (num_label * num_prototypes) x D
        - output
            logits: batch x T x num_label
        '''

        # calculate centroid of prototypes
        b = b.reshape(self.num_labels, self.num_prototypes, -1)
        # num_labels, D
        b = b.mean(dim=1)

        batch = a.shape[0]       # batch
        num_labels = b.shape[0]  # num-labels
        seq_length = a.shape[1]  # sequence length
        if a.size(-1) != b.size(-1):
            raise Exception
     
        # batch, seq_length, num_labels, D
        a = a.unsqueeze(2).expand(batch, seq_length, num_labels, -1)
        # batch, num_labels, D
        b = b.unsqueeze(0).expand(batch, num_labels, -1)
        # batch, seq_length, num_labels, D
        b = b.unsqueeze(1).expand(batch, seq_length, num_labels, -1)

        # calculate distance
        logits = -((a - b)**2).sum(dim=-1)

        return logits

    def cosine_sim(self, a, b, scale=False):
        ''' cosine similarity
        - input
            a (hidden states): batch x T x D
            b (prototypes): (num_label * num_prototypes) x D
        - output
            logits: batch x T x num_label
        '''
        batch, seq_length, hidden_size = a.shape

        # batch x T x D -> (batch * T) x D
        a = a.unsqueeze(2)
        b = b.unsqueeze(0).unsqueeze(0)
        '''
        a = nn.functional.normalize(a)
        b = nn.functional.normalize(b)
        '''
        logits = F.cosine_similarity(a, b, dim=-1)
        
        if scale:
            torch.clamp(self.w, 1e-6)
            logits = self.w * logits

        logits = logits.reshape(-1, seq_length, self.num_labels, self.num_prototypes)
        logits = logits.mean(dim=-1)

        return logits

    def init_prototypes(self):
        pass

    def forward(self,
        input_values,
        input_values_ref=None,
        attention_mask=None,
        attention_mask_ref=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
        prompts=None,
        detection_targets=None,
        detection_targets_ppl=None,
    ):
        outputs = self.model(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        
        batch_size = hidden_states.shape[0]
        # retrieve loss input_lengths from attention_mask
        attention_mask = (
            attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
        )
        # NOTE: main speech
        input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)
        max_length = input_lengths.max() 
        batch_size=input_lengths.shape[0]
        mask = torch.zeros((batch_size, max_length), dtype=torch.bool, device=input_lengths.device)
        for i in range(batch_size):
            mask[i, :input_lengths[i]] = True

        mask = ~mask

        if self.audio_text_fusion != "parallel":
            mask = None
        # 2. NOTE: audio-text encoder
        # labels, prompts: (batch, 74)
        if prompts is not None and self.use_prompt:
            if self.use_mask and labels is not None:
                prompts_mask, targets_mlm = self.mask_uniform(
                    prompts, mask_token_id=self.mask_token_id, num_labels=self.config.num_labels
                )
                hidden_states, phone_embeds = self.audio_text_enc(hidden_states, prompts_mask, mask)
            else:
                hidden_states, phone_embeds = self.audio_text_enc(hidden_states, prompts, mask)

        # logits: B x T x Class
        # calculate distance
        if self.dist == "sed":
            logits_pt = self.negative_sed(hidden_states, self.prototype.weight)
        elif self.dist == "cos":
            logits_pt = self.cosine_sim(hidden_states, self.prototype.weight)
        elif self.dist == "scos":
            logits_pt = self.cosine_sim(hidden_states, self.prototype.weight, scale=True)
        else:
            raise ValueError("dist choices [sed, cos], {} is provided.".format(self.dist))

        logits = self.prediction_head(hidden_states)

        loss = None
        if labels is not None:
            if labels.max() >= self.config.num_labels:
                raise ValueError(f"Label values must be <= vocab_size: {self.config.num_labels}")

            # retrieve loss input_lengths from attention_mask
            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            )
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            # ctc_loss doesn't support fp16
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)
            log_probs_pt = nn.functional.log_softmax(logits_pt, dim=-1, dtype=torch.float32).transpose(0, 1)

            with torch.backends.cudnn.flags(enabled=False):
                loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,    # default: sum
                    zero_infinity=self.config.ctc_zero_infinity, # default: false
                )

                loss_pt = nn.functional.ctc_loss(
                    log_probs_pt,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,    # default: sum
                    zero_infinity=self.config.ctc_zero_infinity, # default: false
                )
            # NOTE: batch loss, better
            batch_size = hidden_states.shape[0]
            loss /= batch_size
            loss_pt /= batch_size

            # NOTE: label smoothing
            if self.lsm_prob > 0:
                loss =  (1 - self.lsm_prob) * loss \
                    + self.lsm_prob * kldiv_lsm_ctc(logits, target_lengths)
                loss_pt = (1 - self.lsm_prob) * loss_pt \
                    + self.lsm_prob * kldiv_lsm_ctc(logits_pt, target_lengths)

            loss = 1.0 * loss + 1.0 * loss_pt

        if not return_dict:
            _HIDDEN_STATES_START_POSITION = 2
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )
