import os
import urllib.request
import tarfile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
import random
from tqdm import tqdm
import math
import re
from ai4india.transformers import AutoTokenizer
import tempfile         
import shutil  

def set_seed(seed):
    """
    It will help to set the seed to create reproducible setup

    Args:
        seed (int): random integer which can help to generate seed
    """
    random.seed(seed)
    torch.manual_seed(seed)

# Dataset Class
class IterableTextDataset(IterableDataset):
    """
    An iterable dataset for processing text data in a memory-efficient way.
    Instead of loading all data into memory, it streams data from disk.
    Inherits from PyTorch's IterableDataset for streaming support.

    Args:
        file_path (str): Path to the text file containing sentences
        tokenizer: Tokenizer object for converting text to tokens
        max_length (int): Maximum sequence length to process (default: 30)
    """

    def __init__(self, file_path, tokenizer, max_length=30):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_lenght = max_length
        self._count_sentences()
    
    def __iter__(self):

        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # removing all leading/trailing whitespaces
                sentence = line.strip()
                # Replace all numbers with ### placeholder
                # This reduces vocabulary size and helps model generalize
                snetence = re.sub(r"\d+", "###", sentence)
                # convert sentence to token IDs
                encoded_sentence = self.tokenizer.encode(sentence, max_length=self.max_lenght, truncation=True)

                # Only use sequences with at least 2 tokens
                # (need at least one input and one target token)
                if len(encoded_sentence) >= 2:
                    input_seq = encoded_sentence[:-1]
                    target_seq = encoded_sentence[1:]
                    yield torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_seq, dtype=torch.long)

    def __len__(self):
        return self._num_sentences
    
    def _count_sentences(self):
        print(f"\nCounting sentences in {self.file_path}...")
        with open(self.file_path, 'r', encoding="utf-8") as f:
            self._num_sentences = sum(1 for _ in f)
        print(f"\nFound {self._num_sentences} sentences in {self.file_path}")

def create_collate_fn(tokenizer):
    pass



