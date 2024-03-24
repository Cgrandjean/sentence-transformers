
from torch.utils.data import Dataset,DataLoader
import nltk
nltk.download('punkt')
from nltk import word_tokenize, TreebankWordDetokenizer
from transformers.utils.import_utils import is_nltk_available, NLTK_IMPORT_ERROR
from transformers import AutoTokenizer
from sentence_transformers.readers import InputExample
import numpy as np
from typing import List

class MaskedAutoEncoderDataset(Dataset):
    """
    The DenoisingAutoEncoderDataset returns InputExamples in the format: texts=[noise_fn(sentence), sentence]
    It is used in combination with the DenoisingAutoEncoderLoss: Here, a decoder tries to re-construct the
    sentence without noise.

    :param sentences: A list of sentences
    :param noise_fn: A noise function: Given a string, it returns a string with noise, e.g. deleted words
    """

    def __init__(self, sentences: List[str],tokenizer ):
        if not is_nltk_available():
            raise ImportError(NLTK_IMPORT_ERROR.format(self.__class__.__name__))

        self.sentences = sentences
        self.tokenizer=tokenizer

    def __getitem__(self, item):
        sent = self.sentences[item]
        return InputExample(texts=[self.noisen(sent,MASK_ratio=0.15), self.noisen(sent,MASK_ratio=0.4),sent])

    def __len__(self):
        return len(self.sentences)

    # Masking noise.
    def noisen(self,text, MASK_ratio=0.15):
        mask_id=self.tokenizer.mask_token_id
        words= text.split()#word_tokenize(text)
        # Apply the masking logic to each word and rejoin the sentence
        splitted_tokens = self.tokenizer.batch_encode_plus(words,return_attention_mask=False,return_token_type_ids=False,add_special_tokens=False)['input_ids']#encode each tokens in each
        masked_tokens =[[ mask_id if np.random.rand() < MASK_ratio else tok_id for tok_id in token]  for token in splitted_tokens]
        masked_sentence=' '.join([self.tokenizer.decode(masked_token).replace(" ",'') for masked_token in masked_tokens])
        return masked_sentence