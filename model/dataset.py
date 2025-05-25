import tiktoken
import torch
import itertools
import random
from datasets import load_dataset
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


class WikiShuffleDataset(IterableDataset):
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.article_pool = []
        self.read_positions = []
        self.active_articles = set()

    def setup_epoch(self, num_articles):
        pool_size = len(self.article_pool)
        if num_articles > pool_size:
            raise ValueError(f"num_articles must not be greater than the number of articles in the pool, pool_size={pool_size}, got num_articles={num_articles}")

        self.active_articles.clear()
        while len(self.active_articles) < num_articles:
            i = random.randint(0, pool_size - 1)
            if i not in self.active_articles:
                self.active_articles.add(i)

        self.read_positions = [0] * pool_size

    def prefetch(self, starting_index, ending_index):
        if ending_index <= starting_index:
            raise ValueError(f"ending_index must be greater than starting_index, got starting_index={starting_index}, ending_index={ending_index}")

        stream = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)
        articles = list(itertools.islice(stream, starting_index, ending_index))

        self.article_pool = [self.tokenizer.encode(x['text']) for x in articles]
        print(f"articles prefetched: {len(self.article_pool)}")

    def __iter__(self):
        while len(self.active_articles) > 0:
            article_index = random.choice(list(self.active_articles))
            article = self.article_pool[article_index]
            sequence_start = self.read_positions[article_index]
            sequence_end = sequence_start + self.max_length 
            
            sequence = article[sequence_start:sequence_end]
            targets = article[sequence_start+1:sequence_end+1]

            if len(sequence) != self.max_length or len(targets) != self.max_length:
                print(f"removing article {article_index}")
                self.active_articles.remove(article_index)
            else:
                self.read_positions[article_index] += self.max_length
                yield torch.tensor(sequence), torch.tensor(targets)
               
class SequenceFileDataset(IterableDataset):
    def __init__(self):
        self.sequence_pairs = []

    def load(self, sequence_file):
        sequences = []
        with open(sequence_file, "r") as f:
            for line in f:
                sequences.append(list(map(int, line.strip().split())))
        self.sequence_pairs = list(zip(sequences[::2],sequences[1::2]))
        
    def __iter__(self):
        for sequence_pair in self.sequence_pairs:
            input_tokens, target_tokens = sequence_pair
            if input_tokens[1:] != target_tokens[:-1]:
                print(f"Invalid sequence pair detected:\ninput: {input_tokens}\nouput: {target_tokens}")
            yield torch.tensor(input_tokens), torch.tensor(target_tokens)

def select_random_articles(num_articles, article_pool):
    pool_size = len(article_pool)
    if num_articles > pool_size:
        raise ValueError(f"num_articles must not be greater than the number of articles in the pool, pool_size={pool_size}, got num_articles={num_articles}")

    selected_articles = []
    active_articles = set()
    while len(active_articles) < num_articles:
        i = random.randint(0, pool_size - 1)
        if i not in active_articles:
            active_articles.add(i)
            selected_articles.append(article_pool[i])

    return selected_articles

def fetch_random_wiki_articles(num_articles, pool_starting_index, pool_ending_index):
    if pool_ending_index <= pool_starting_index:
        raise ValueError(f"ending_index must be greater than starting_index, got pool_starting_index={pool_starting_index}, ending_index={pool_ending_index}")

    stream = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)
    article_pool = list(itertools.islice(stream, pool_starting_index, pool_ending_index))

    article_pool = [x['text'] for x in article_pool]
    print(f"articles fetched: {len(article_pool)}")
    return select_random_articles(num_articles, article_pool)

def concat_and_tokenize_articles(articles, tokenizer):
    concatenated = "<|endoftext|>".join(articles)
    return tokenizer.encode(concatenated, allowed_special={"<|endoftext|>"})

def chunk_sequences(tokenized_text, chunk_length):
    chunked = [tokenized_text[i:i + chunk_length] for i in range(0, len(tokenized_text), chunk_length)]
    return chunked[:len(chunked)-1]

def shuffle_sequences(sequences):
    return random.shuffle(sequences)

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
