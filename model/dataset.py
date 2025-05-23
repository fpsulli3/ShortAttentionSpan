import tiktoken
import torch
from torch.utils.data import IterableDataset, Dataset, DataLoader
from pathlib import Path

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        print("txt", txt)

        token_ids = tokenizer.encode(txt)

        print("token_ids", token_ids)
        
        for i in range(0, len(token_ids) - max_length, stride):
            print("iterating")
            input_chunk = token_ids[i:i+max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

        print("input_ids", self.input_ids)
        print("target_ids", self.target_ids)

    def __len__(self):
        print("len input_ids", self.input_ids)
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

class LLMIterableDataset(IterableDataset):
    def __init__(self, files, tokenizer, max_length):
        self.files = files 
        self.tokenizer = tokenizer 
        self.max_length = max_length
        self.eot = tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
        
        self.chunk_size_bytes = 5 * max_length # assume 5 bytes per token

    def __iter__(self):
        tokens = []
        for file in self.files:
            print(f"Loading {file}")
            with open(file, 'r', encoding='utf-8') as f:
                while chunk := f.read(self.chunk_size_bytes):
                    chunk_tokens = self.tokenizer.encode(chunk)
                    tokens.extend(chunk_tokens)
                    while len(tokens) > self.max_length:
                        input_tokens = torch.tensor(tokens[:self.max_length])
                        target_tokens = torch.tensor(tokens[1:self.max_length+1])
                        tokens = tokens[self.max_length:]
                        yield input_tokens, target_tokens
            tokens.append(self.eot)

                    
def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    return dataloader

def create_llm_dataloader(filedir, batch_size=4, max_length=1024, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    path = Path(filedir)
    files = sorted(path.rglob(f"*.txt"))
    dataset = LLMIterableDataset(files, tokenizer, max_length)
    return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=drop_last,
            num_workers=num_workers,
        )
