
from torch import nn, Tensor
from typing import Iterable, Dict
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer,AutoModelForCausalLM,AutoConfig
import logging
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
import torch
logger = logging.getLogger(__name__)

class MaskedAutoEncoderLoss(nn.Module):
    def __init__(self, model: SentenceTransformer):
        super(MaskedAutoEncoderLoss, self).__init__()
        self.encoder = model[0].auto_model  # This will be the final model used during the inference time.
        self.embeddings=self.encoder.embeddings
        self.config = model[0].auto_model.config
        self.MLM_head=BertOnlyMLMHead(self.config)
        self.tokenizer = model.tokenizer
        self.cross_entropy = nn.CrossEntropyLoss()

        # self.decoder=

        # decoder_config.is_decoder = True
        # decoder_config.add_cross_attention = True
        # decoder_config.num_hidden_layers=1
        # kwargs_decoder = {"config": decoder_config}
        # try:
        #     self.decoder = AutoModelForCausalLM.from_pretrained(name_or_path, **kwargs_decoder)
        # except ValueError as e:
        #     logger.error(
        #         f'Model name or path "{name_or_path}" does not support being as a decoder. Please make sure the decoder model has an "XXXLMHead" class.'
        #     )
        #     raise e
        # if self.tokenizer_decoder.pad_token is None:
        #     # Needed by GPT-2, etc.
        #     self.tokenizer_decoder.pad_token = self.tokenizer_decoder.eos_token
        #     self.decoder.config.pad_token_id = self.decoder.config.eos_token_id

        # if len(AutoTokenizer.from_pretrained(name_or_path)) != len(self.tokenizer_encoder):
        #     logger.warning(
        #         "WARNING: The vocabulary of the encoder has been changed. One might need to change the decoder vocabulary, too."
        #     )
        
    def forward_exp(self, sentence_features: Iterable[Dict[str, Tensor]]):
        encoder_inputs,decoder_inputs, target_features= tuple(sentence_features)
        target_features[encoder_inputs.input_ids != self.tokenizer.mask_token_id] = -100
        last_hidden_states=self.encoder(**encoder_inputs)[0]
        lm_logits=self.MLM_head(last_hidden_states)
        loss = self.cross_entropy(lm_logits.view(-1, self.config.vocab_size), target_features.view(-1))
        
        cls_hiddens = last_hidden_states[:, :1]  # B 1 D

        embeddings = self.embeddings(input_ids=decoder_inputs)
        hiddens = torch.cat([cls_hiddens, embeddings[:, 1:]], dim=1)

        decoder_position_ids = self.embeddings.position_ids[:, :decoder_inputs.size(1)]
        decoder_position_embeddings = self.embeddings.position_embeddings(decoder_position_ids)  # B L D
        query = decoder_position_embeddings + cls_hiddens

        matrix_attention_mask = self.lm.get_extended_attention_mask(
            decoder_attention_mask,
            decoder_attention_mask.shape,
            decoder_attention_mask.device
        )

        hiddens = self.c_head(query=query,
                              key=hiddens,
                              value=hiddens,
                              attention_mask=matrix_attention_mask)[0]
        return loss
        
        # reps = self.encoder(source_features)["sentence_embedding"]  # (bsz, hdim)
        # # print(source_features)
        # # Prepare input and output
        # target_length = target_features["input_ids"].shape[1]
        # decoder_input_ids = target_features["input_ids"].clone()[:, : target_length - 1]
        # # print(decoder_input_ids.shape)
        # label_ids = target_features["input_ids"][:, 1:]
        # print( label_ids,decoder_input_ids)
        
        # # Decode
        # decoder_outputs = self.decoder(
        #     input_ids=decoder_input_ids,
        #     inputs_embeds=None,
        #     attention_mask=None,
        #     encoder_hidden_states=reps[:, None],  # (bsz, hdim) -> (bsz, 1, hdim)
        #     encoder_attention_mask=source_features["attention_mask"][:, 0:1],
        #     labels=None,
        #     return_dict=None,
        #     use_cache=False,
        # )

        # # Calculate loss
        # lm_logits = decoder_outputs[0]
        # ce_loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer_decoder.pad_token_id)
        # loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), label_ids.reshape(-1))
        # return loss
    # def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
    #     source_features, target_features= tuple(sentence_features)
    #     reps = self.encoder(source_features)["sentence_embedding"]  # (bsz, hdim)
    #     # print(source_features)
    #     # Prepare input and output
    #     target_length = target_features["input_ids"].shape[1]
    #     decoder_input_ids = target_features["input_ids"].clone()[:, : target_length - 1]
    #     # print(decoder_input_ids.shape)
    #     label_ids = target_features["input_ids"][:, 1:]
    #     print( label_ids,decoder_input_ids)
        
    #     # Decode
    #     decoder_outputs = self.decoder(
    #         input_ids=decoder_input_ids,
    #         inputs_embeds=None,
    #         attention_mask=None,
    #         encoder_hidden_states=reps[:, None],  # (bsz, hdim) -> (bsz, 1, hdim)
    #         encoder_attention_mask=source_features["attention_mask"][:, 0:1],
    #         labels=None,
    #         return_dict=None,
    #         use_cache=False,
    #     )

    #     # Calculate loss
    #     lm_logits = decoder_outputs[0]
    #     ce_loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer_decoder.pad_token_id)
    #     loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), label_ids.reshape(-1))
    #     return loss


    # def forward_(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
    #     encoder_inputs,decoder_inputs , target_features = tuple(sentence_features)
    #     reps = self.encoder(encoder_inputs)["sentence_embedding"]  # (bsz, hdim)

    #     # Prepare input and output
    #     target_length = decoder_inputs["input_ids"].shape[1]
    #     decoder_input_ids = decoder_inputs["input_ids"].clone()[:, : target_length - 1]
    #     label_ids = target_features["input_ids"][:, 1:]

    #     # Decode
    #     decoder_outputs = self.decoder(
    #         input_ids=decoder_input_ids,
    #         inputs_embeds=None,
    #         attention_mask=None,
    #         encoder_hidden_states=reps[:, None],  # (bsz, hdim) -> (bsz, 1, hdim)
    #         encoder_attention_mask=encoder_inputs["attention_mask"][:, 0:1],
    #         labels=None,
    #         return_dict=None,
    #         use_cache=False,
    #     )

    #     # Calculate loss
    #     lm_logits = decoder_outputs[0]
    #     ce_loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer_decoder.pad_token_id)
    #     loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), label_ids.reshape(-1))
    #     return loss
