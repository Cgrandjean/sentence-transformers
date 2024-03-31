
from torch import nn, Tensor
from typing import Iterable, Dict
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer,AutoModelForCausalLM,AutoConfig
import logging

logger = logging.getLogger(__name__)

class MaskedAutoEncoderLoss(nn.Module):
    def __init__(self, model: SentenceTransformer, decoder_name_or_path: str = None):
        """
        This loss expects as input a pairs of damaged sentences and the corresponding original ones.
        During training, the decoder reconstructs the original sentences from the encoded sentence embeddings.
        Here the argument 'decoder_name_or_path' indicates the pretrained model (supported by Hugging Face) to be used as the decoder.
        Since decoding process is included, here the decoder should have a class called XXXLMHead (in the context of Hugging Face's Transformers).
        The 'tie_encoder_decoder' flag indicates whether to tie the trainable parameters of encoder and decoder,
        which is shown beneficial to model performance while limiting the amount of required memory.
        Only when the encoder and decoder are from the same architecture, can the flag 'tie_encoder_decoder' work.

        The data generation process (i.e. the 'damaging' process) has already been implemented in ``DenoisingAutoEncoderDataset``,
        allowing you to only provide regular sentences.

        :param model: SentenceTransformer model
        :param decoder_name_or_path: Model name or path for initializing a decoder (compatible with Huggingface's Transformers)
        :param tie_encoder_decoder: whether to tie the trainable parameters of encoder and decoder

        References:
            * TSDAE paper: https://arxiv.org/pdf/2104.06979.pdf
            * `Unsupervised Learning > TSDAE <../../examples/unsupervised_learning/TSDAE/README.html>`_

        Requirements:
            1. The decoder should have a class called XXXLMHead (in the context of Hugging Face's Transformers)
            2. Should use a large corpus

        Inputs:
            +------------------------------------------------------+--------+
            | Texts                                                | Labels |
            +======================================================+========+
            | (damaged\_sentence, original\_sentence) pairs        | none   |
            +------------------------------------------------------+--------+
            | sentence fed through ``DenoisingAutoEncoderDataset`` | none   |
            +------------------------------------------------------+--------+

        Example:
            ::

                from sentence_transformers import SentenceTransformer, losses
                from sentence_transformers.datasets import DenoisingAutoEncoderDataset
                from torch.utils.data import DataLoader

                model_name = "bert-base-cased"
                model = SentenceTransformer(model_name)
                train_sentences = [
                    "First training sentence", "Second training sentence", "Third training sentence", "Fourth training sentence",
                ]
                batch_size = 2
                train_dataset = DenoisingAutoEncoderDataset(train_sentences)
                train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
                train_loss = losses.DenoisingAutoEncoderLoss(
                    model, decoder_name_or_path=model_name, tie_encoder_decoder=True
                )
                model.fit(
                    train_objectives=[(train_dataloader, train_loss)],
                    epochs=10,
                )
        """
        super(MaskedAutoEncoderLoss, self).__init__()
        self.encoder = model  # This will be the final model used during the inference time.
        self.tokenizer_encoder = model.tokenizer

        name_or_path = model[0].auto_model.config._name_or_path

        self.tokenizer_decoder = AutoTokenizer.from_pretrained(name_or_path)

        decoder_config = AutoConfig.from_pretrained(name_or_path)
        decoder_config.is_decoder = True
        decoder_config.add_cross_attention = True
        decoder_config.num_hidden_layers=1
        kwargs_decoder = {"config": decoder_config}
        try:
            self.decoder = AutoModelForCausalLM.from_pretrained(name_or_path, **kwargs_decoder)
        except ValueError as e:
            logger.error(
                f'Model name or path "{name_or_path}" does not support being as a decoder. Please make sure the decoder model has an "XXXLMHead" class.'
            )
            raise e
        if self.tokenizer_decoder.pad_token is None:
            # Needed by GPT-2, etc.
            self.tokenizer_decoder.pad_token = self.tokenizer_decoder.eos_token
            self.decoder.config.pad_token_id = self.decoder.config.eos_token_id

        if len(AutoTokenizer.from_pretrained(name_or_path)) != len(self.tokenizer_encoder):
            logger.warning(
                "WARNING: The vocabulary of the encoder has been changed. One might need to change the decoder vocabulary, too."
            )

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        source_features, target_features= tuple(sentence_features)
        reps = self.encoder(source_features)["sentence_embedding"]  # (bsz, hdim)
        # print(source_features)
        # Prepare input and output
        target_length = target_features["input_ids"].shape[1]
        decoder_input_ids = target_features["input_ids"].clone()[:, : target_length - 1]
        # print(decoder_input_ids.shape)
        label_ids = target_features["input_ids"][:, 1:]
        print( label_ids,decoder_input_ids)
        
        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            inputs_embeds=None,
            attention_mask=None,
            encoder_hidden_states=reps[:, None],  # (bsz, hdim) -> (bsz, 1, hdim)
            encoder_attention_mask=source_features["attention_mask"][:, 0:1],
            labels=None,
            return_dict=None,
            use_cache=False,
        )

        # Calculate loss
        lm_logits = decoder_outputs[0]
        ce_loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer_decoder.pad_token_id)
        loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), label_ids.reshape(-1))
        return loss


    def forward_(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        encoder_inputs,decoder_inputs , target_features = tuple(sentence_features)
        reps = self.encoder(encoder_inputs)["sentence_embedding"]  # (bsz, hdim)

        # Prepare input and output
        target_length = decoder_inputs["input_ids"].shape[1]
        decoder_input_ids = decoder_inputs["input_ids"].clone()[:, : target_length - 1]
        label_ids = target_features["input_ids"][:, 1:]

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            inputs_embeds=None,
            attention_mask=None,
            encoder_hidden_states=reps[:, None],  # (bsz, hdim) -> (bsz, 1, hdim)
            encoder_attention_mask=encoder_inputs["attention_mask"][:, 0:1],
            labels=None,
            return_dict=None,
            use_cache=False,
        )

        # Calculate loss
        lm_logits = decoder_outputs[0]
        ce_loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer_decoder.pad_token_id)
        loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), label_ids.reshape(-1))
        return loss
