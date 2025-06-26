import torch
import tiktoken
from torch.utils.data import DataLoader
from model.llm import GPTModel 
from model.config import GPT_CONFIG_GPT2_SMALL
import argparse
import pandas as pd
import os
import model.dataset as ds
import boto3
from torch.cuda.amp import autocast, GradScaler
import math

# args

parser = argparse.ArgumentParser(description="Train GPT model")

parser.add_argument(
        "--num_epochs",
        type=int,
        required=True,
        help="The number of epochs to run",
    )

args = parser.parse_args()


# config

config = GPT_CONFIG_GPT2_SMALL

# file paths

data_dir = os.environ.get("DATA_DIR", "/mnt/data")
output_dir = os.environ.get("OUT_DIR", "/mnt/output/")

os.makedirs(output_dir, exist_ok=True)

log_path = os.path.join(output_dir, "loss.csv")
autosave_path = os.path.join(output_dir, "autosave.pt")
training_sequence_file = os.path.join(output_dir, "training_sequences.txt")
validation_sequence_file = os.path.join(output_dir, "validation_sequences.txt")

# logging

starting_epoch = 0
s3 = boto3.client("s3")
BUCKET_NAME = "fpsulli3-llm-bucket"

def upload_log(local_path):
    s3_key = f"logs/{os.path.basename(local_path)}"
    s3.upload_file(local_path, BUCKET_NAME, s3_key)

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
    upload_log(log_path)

def upload_checkpoint(local_path, s3_key=None):
    if s3_key is None:
        s3_key = f"checkpoints/{os.path.basename(local_path)}"
    s3.upload_file(local_path, BUCKET_NAME, s3_key)

# data loaders

tokenizer = tiktoken.get_encoding("gpt2")
vocab_size = tokenizer.n_vocab
config["vocab_size"] = vocab_size


token_files = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".tokens")
        ]


shuffle_files = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".shuffle")
        ]

token_files = sorted(token_files)
shuffle_files = sorted(shuffle_files)

print(f"Token files found: {token_files}")
print(f"Shuffle files found: {shuffle_files}")

if len(token_files) < 2:
    raise ValueError(f"We need at least 2 token files for training and validation, len(token_files)={len(token_files)}")

if len(token_files) != len(shuffle_files):
    raise ValueError(f"The number of shuffle and token files must match, len(token_files)={len(token_files)}, len(shuffle_files)={len(shuffle_files)}")

for i in range(0, len(token_files)):
    token_file = token_files[i]
    shuffle_file = shuffle_files[i]
    token_file_basename = os.path.basename(token_file)
    shuffle_file_basename = os.path.basename(shuffle_file)
    token_file_name = os.path.splitext(token_file_basename)[0]
    shuffle_file_name = os.path.splitext(shuffle_file_basename)[0]
    if token_file_name != shuffle_file_name:
        raise ValueError(f"Found unmatching token and shuffle file names. token file: {token_file_name}, shuffle file: {shuffle_file_name}")

training_token_files = token_files[:-1]
validation_token_files = token_files[-1:]

training_shuffle_files = shuffle_files[:-1]
validation_shuffle_files = shuffle_files[-1:]

# Dataset creation will be moved inside the epoch loop to reset for each epoch

# device setup

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"The device is ${device}")


# model setup

model = GPTModel(cfg=config)
model.train()
model.to(device)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
scaler = GradScaler()

# Learning rate scheduler with warmup and cosine annealing
warmup_steps = 2000
# Note: total_steps will be calculated after first dataset creation
# Scheduler will be created after first dataset to get accurate total_steps
scheduler = None
total_steps = None

# Gradient accumulation settings
accumulation_steps = 8

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
autosave_every = 1000 # batches
final_epoch = args.num_epochs

def train_epoch(epoch):
    print(f"\nTraining epoch {epoch+1}/{final_epoch}\n")
    running_loss = 0.0
    training_losses = []
    accumulated_loss = 0.0

    model.train()
    optimizer.zero_grad()
    
    for batch_idx, (input_tokens, target_tokens) in enumerate(training_data_loader):
        input_tokens = input_tokens.to(device)
        target_tokens = target_tokens.to(device)

        with autocast():
            logits = model(input_tokens)
            loss = loss_fn(
                    logits.view(-1, logits.size(-1)),
                    target_tokens.view(-1),
                )
            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps

        scaler.scale(loss).backward()
        accumulated_loss += loss.item()
        
        # Perform optimizer step every accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0:
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            
            # Record loss (unscaled)
            training_losses.append(accumulated_loss * accumulation_steps)
            running_loss += accumulated_loss * accumulation_steps
            accumulated_loss = 0.0
    
        if (batch_idx + 1) % print_every == 0:
            avg_loss = running_loss / (print_every // accumulation_steps)
            current_lr = scheduler.get_last_lr()[0]
            print(f"Batch {batch_idx+1}: avg loss = {avg_loss:.4f}, lr = {current_lr:.6f}")
            running_loss = 0.0

        if (batch_idx + 1) % autosave_every == 0:
            torch.save(model.state_dict(), autosave_path)
            upload_checkpoint(autosave_path)

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
    # Recreate datasets to reset file handles for each epoch
    training_dataset = ds.PreshuffledTokenFileDataset(training_token_files, training_shuffle_files, config["context_length"])
    validation_dataset = ds.PreshuffledTokenFileDataset(validation_token_files, validation_shuffle_files, config["context_length"])
    
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
    
    # Create scheduler on first epoch with calculated total steps
    if scheduler is None:
        # Calculate total steps: 6,442,450,944 bytes ÷ 2 bytes/token ÷ 1024 tokens/seq ÷ 4 batch_size = 786,432 batches/epoch
        batches_per_epoch = 786432
        total_steps = args.num_epochs * batches_per_epoch
        def get_lr(step):
            if step < warmup_steps:
                return 3e-4 * step / warmup_steps
            else:
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return 3e-5 + (3e-4 - 3e-5) * 0.5 * (1 + math.cos(math.pi * progress))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)
    
    # train and validate
    train_loss = train_epoch(epoch)
    validation_loss = validate_epoch(epoch)

    # update trackers
    log_loss(epoch, train_loss, validation_loss)

    # autosave
    torch.save(model.state_dict(), autosave_path)

    # save milestone
    milestone_path = os.path.join(output_dir, f"milestone_{epoch+1}.pt")
    torch.save(model.state_dict(), milestone_path)
    upload_checkpoint(milestone_path)


final_path = os.path.join(output_dir, "final.pt")
torch.save(model.state_dict(), final_path)

