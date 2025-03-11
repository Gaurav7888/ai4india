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
from ai4india.transformer import AutoTokenizer
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
    """Creates a collate function for batching sequences of different lengths."""
    
    def collate_fn(batch):
        "function: Collate function that handles padding in batches"
        input_seqs , target_seqs = zip(*batch)
        pad_index = tokenizer.pad_token_id
        input_padded = nn.utils.rnn.pad_sequence( input_seqs, batch_first=True, padding_value=pad_index)
        target_padded = nn.utils.rnn.pad_sequence(target_seqs, batch_first=True, padding_value=pad_index)
        return input_padded, target_padded
    return collate_fn

def check_file_exists(filename):
    return os.path.exists(filename)

def download_file(url):
   
    filename = "news.tar.gz"

    if not check_file_exists(filename):
        print(f"\nDownloading dataset from {url}...")
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req) as response:
            with open(filename, "wb") as out_file:
                out_file.write(response.read())
        print("\nDownload completed.")
    else:
        print(f"\n{filename} already downloaded.")
    return filename

def is_within_directory(directory, target):

    abs_directory = os.path.abspath(directory)
    abs_target = os.path.abspath(target)
    prefix = os.path.commonprefix([abs_directory, abs_target])
    return prefix == abs_directory

def extract_dataset(filename):
    data_dir = os.path.join(os.path.dirname(filename), "news")
    train_path = os.path.join(data_dir, "train.txt")
    test_path = os.path.join(data_dir, "test.txt")

    if check_file_exists(train_path) and check_file_exists(test_path):
        print("\nData files already extracted.")
        return train_path, test_path

    print("\nListing archive contents:")
    with tarfile.open(filename, "r:gz") as tar:
        for member in tar.getmembers():
            print(f"\nArchive member: {member.name}")

        print("\nExtracting files...")
        # Extract to current directory first
        tar.extractall('.')

    if not (check_file_exists(train_path) and check_file_exists(test_path)):
        raise FileNotFoundError(f"\nRequired files not found in the archive. Please check the paths above.")

    print("\nExtraction completed.")
    return train_path, test_path

def create_datasets(train_file, test_file, tokenizer, max_length=30):
    # Create training dataset
    train_dataset = IterableTextDataset(train_file, tokenizer, max_length)
    # Create test dataset
    test_dataset = IterableTextDataset(test_file, tokenizer, max_length)

    # Print dataset sizes
    print(f"\nTraining sentences: {len(train_dataset)}")
    print(f"\nTest sentences: {len(test_dataset)}")

    return train_dataset, test_dataset

def create_dataloaders(train_dataset, test_dataset, batch_size, collate_fn):
    # Create training data loader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,    # Function to handle padding
        num_workers=0             # Number of worker processes (0 = single process)
    )
    # Create test data loader
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=0
    )
    return train_dataloader, test_dataloader

def download_and_prepare_data(url, batch_size, tokenizer, max_length=30):

    # Step 1: Download dataset archive from URL
    filename = download_file(url)

    # Step 2: Extract training and test files from archive
    train_file, test_file = extract_dataset(filename)

    # Step 3: Create dataset objects for streaming data
    train_dataset, test_dataset = create_datasets(train_file, test_file, tokenizer, max_length)

    # Step 4: Create function to handle batch creation
    collate_fn = create_collate_fn(tokenizer)

    # Step 5: Create and return data loaders
    return create_dataloaders(train_dataset, test_dataset, batch_size, collate_fn)


