from os import wait
import torch
import tiktoken
from torch.utils.data import DataLoader
from model.llm import GPTModel 
from model.config import GPT_CONFIG_WIKI
import argparse
import math
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd
import os
import model.dataset as ds
import random

# args

parser = argparse.ArgumentParser(description="Train GPT model")

parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Required path to save the trained model. Omit extension.",
    )

parser.add_argument(
        "--num_epochs",
        type=int,
        required=True,
        help="The number of epochs to run",
    )

args = parser.parse_args()


# config

config = GPT_CONFIG_WIKI

# file paths

output_dir = f"outputs/{args.output}"
os.makedirs(output_dir, exist_ok=True)

log_path = f"{output_dir}/loss.csv"
autosave_path = f"{output_dir}/autosave.pt"
training_sequence_file = f"{output_dir}/training_sequences.txt"
validation_sequence_file = f"{output_dir}/validation_sequences.txt"

# logging

starting_epoch = 0

def load_loss_log():
    global epoch_train_losses, epoch_val_losses, starting_epoch 

    df = pd.read_csv(log_path)
    epoch_train_losses = df["train_loss"].tolist()
    epoch_val_losses = df["val_loss"].tolist()
    starting_epoch = int(df["epoch"].max()) + 1
    print(f"Resuming from epoch {starting_epoch}")

def log_loss(epoch, train_loss, val_loss):
    new_row = pd.DataFrame([{
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
    }])

    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        df = pd.concat([df, new_row], ignore_index=True)
    else:
        df = new_row 

    df.to_csv(log_path, index=False)


# data loaders

tokenizer = tiktoken.get_encoding("gpt2")
vocab_size = tokenizer.n_vocab
config["vocab_size"] = vocab_size

if not os.path.exists(training_sequence_file) or not os.path.exists(validation_sequence_file):
    print("One or more sequence files is missing. Creating from scratch...")

    print("Fetching articles...")
    fetched_articles = ds.fetch_random_wiki_articles(16500, 0, 40000)
    print(f"{len(fetched_articles)} fetched and selected")

    training_articles = fetched_articles[:15000]
    validation_articles = fetched_articles[15000:]
    print(f"We have {len(training_articles)} training articles and {len(validation_articles)} validation articles")

    print("Tokenizing...")
    training_tokens = ds.concat_and_tokenize_articles(training_articles, tokenizer)
    validation_tokens = ds.concat_and_tokenize_articles(validation_articles, tokenizer)

    print("Chunking...")
    training_input_sequences = ds.chunk_sequences(training_tokens, config["context_length"])
    training_target_sequences = ds.chunk_sequences(training_tokens[1:], config["context_length"])
    training_sequence_pairs = list(zip(training_input_sequences, training_target_sequences))

    validation_input_sequences = ds.chunk_sequences(validation_tokens, config["context_length"])
    validation_target_sequences = ds.chunk_sequences(validation_tokens[1:], config["context_length"])
    validation_sequence_pairs = list(zip(validation_input_sequences, validation_target_sequences))

    print("Shuffling sequences...")
    random.shuffle(training_sequence_pairs)
    random.shuffle(validation_sequence_pairs)

    print("Saving...")
    with open(training_sequence_file, "w") as f:
        for sequence_pair in training_sequence_pairs:
            f.write(" ".join(map(str, sequence_pair[0])) + "\n")
            f.write(" ".join(map(str, sequence_pair[1])) + "\n")

    with open(validation_sequence_file, "w") as f:
        for sequence_pair in validation_sequence_pairs:
            f.write(" ".join(map(str, sequence_pair[0])) + "\n")
            f.write(" ".join(map(str, sequence_pair[1])) + "\n")

    print("Wiki articles fetched, tokenized, chunked, shuffled, and saved :)")
    user_input = input("Press enter to continue or 'x' to exit...").strip().lower()

    if user_input == "x":
        print("Exiting.")
        exit()

training_dataset = ds.SequenceFileDataset()
validation_dataset = ds.SequenceFileDataset()

training_dataset.load(training_sequence_file)
validation_dataset.load(validation_sequence_file)

training_data_loader = DataLoader(
        training_dataset,
        batch_size = config["batch_size"],
        shuffle=False,
        drop_last=True,
        num_workers=0,
    )

validation_data_loader = DataLoader(
        validation_dataset,
        batch_size = config["batch_size"],
        shuffle=False,
        drop_last=True,
        num_workers=0,
    )

# device setup

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"The device is ${device}")


# model setup

model = GPTModel(cfg=config)
model.train()
model.to(device)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

def lr_schedule(step):
    warmup_steps = 500
    total_steps = args.num_epochs

    if step < warmup_steps:
            return step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return 0.5 * (1+math.cos(math.pi * progress))

# scheduler = LambdaLR(optimizer, lr_lambda=lr_schedule)


# restore previous session if possible

if os.path.exists(log_path) and os.path.exists(autosave_path):
    print("Found existing log and autosave. Restoring from previous.")
    load_loss_log()
    model.load_state_dict(torch.load(autosave_path))
else:
    if not os.path.exists(log_path):
        print("Log path not found")

    if not os.path.exists(autosave_path):
        print("Autosave not found")

    print("Starting from scratch")

    if os.path.exists(log_path):
        os.remove(log_path)

    if os.path.exists(autosave_path):
        os.remove(autosave_path)

    starting_epoch = 0
    epoch_train_losses = []
    epoch_val_losses = []
    

# training loop

milestone_save_every = 100 # epochs
print_every = 10 # batches
final_epoch = args.num_epochs

def train_epoch(epoch):
    print(f"\nTraining epoch {epoch+1}/{final_epoch}\n")
    running_loss = 0.0
    training_losses = []

    model.train()
    for batch_idx, (input_tokens, target_tokens) in enumerate(training_data_loader):
        input_tokens = input_tokens.to(device)
        target_tokens = target_tokens.to(device)

        logits = model(input_tokens)
        
        loss = loss_fn(
                logits.view(-1, logits.size(-1)),
                target_tokens.view(-1),
            )

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        training_losses.append(loss.item())
        running_loss += loss.item()
    
        if (batch_idx + 1) % print_every == 0:
            avg_loss = running_loss / print_every
            print(f"Batch {batch_idx+1}: avg loss = {avg_loss:.4f}")
            running_loss = 0.0

    # scheduler.step()
    avg_training_loss = sum(training_losses) / len(training_losses)
    print(f"Training loss: {avg_training_loss:.4f}")
    return avg_training_loss

def validate_epoch(epoch):
    print(f"Validating epoch {epoch+1}/{final_epoch}")
    model.eval()
    with torch.no_grad():
        validation_losses = []
        for batch in validation_data_loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)
            loss = loss_fn(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                )
            validation_losses.append(loss.item())

        avg_validation_loss = sum(validation_losses) / len(validation_losses)
        print(f"Validation loss: {avg_validation_loss:.4f}")
        return avg_validation_loss

for epoch in range(starting_epoch, final_epoch):
    # train and validate
    train_loss = train_epoch(epoch)
    validation_loss = validate_epoch(epoch)

    # update trackers
    log_loss(epoch, train_loss, validation_loss)

    # autosave
    torch.save(model.state_dict(), autosave_path)

    # save milestone
    if (epoch + 1) % milestone_save_every == 0:
        torch.save(model.state_dict(), f"outputs/{args.output}/milestone_{epoch + 1}.pt")


torch.save(model.state_dict(), f"outputs/{args.output}/final.pt")

