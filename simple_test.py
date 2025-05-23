import torch
import tiktoken
import argparse
from model.llm import GPTModel 

parser = argparse.ArgumentParser(description="Generate text from a prompt")
parser.add_argument("--prompt", type=str, required=True, help="Prompt to feed the model")
args = parser.parse_args()

prompt = args.prompt

GPT_CONFIG = {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 384,
        "n_layers": 6,
        "drop_rate": 0.1,
        "qkv_bias": False,
    }

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"The device is ${device}")

model = GPTModel(cfg=GPT_CONFIG)
model.load_state_dict(torch.load("gpt_weights.pt"))
model.to(device)
model.eval()

tokenizer = tiktoken.get_encoding("gpt2")
input_ids = tokenizer.encode(prompt)
input_ids = torch.tensor([input_ids]).to(device)

with torch.no_grad():
    for n in range(100):
        logits = model(input_ids)
        next_token_logits = logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=1)  # shape (B,)
        input_ids = torch.cat([input_ids, next_token_id.unsqueeze(-1)], dim=1)

    result = tokenizer.decode(input_ids[0].tolist())
    print(f"Result: {result}")









