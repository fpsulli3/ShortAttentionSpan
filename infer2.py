import torch
import tiktoken
import argparse
from model.llm import GPTModel 
from model.config import GPT_CONFIG_GPT2_SMALL
import torch.nn.functional as F

parser = argparse.ArgumentParser(description="Generate text from a prompt")
parser.add_argument("--prompt", type=str, required=True, help="Prompt to feed the model")
parser.add_argument("--model", type=str, required=True, help="The model weights file to load")
args = parser.parse_args()

prompt = args.prompt

GPT_CONFIG = GPT_CONFIG_GPT2_SMALL

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"The device is ${device}")

tokenizer = tiktoken.get_encoding("gpt2")
GPT_CONFIG["vocab_size"] = tokenizer.n_vocab

model = GPTModel(cfg=GPT_CONFIG)
model.load_state_dict(torch.load(args.model))
model.to(device)
model.eval()

input_ids = tokenizer.encode(prompt)
input_ids = torch.tensor([input_ids]).to(device)

temperature = 0.9
top_k = 120

with torch.no_grad():
    for n in range(900):
        logits = model(input_ids)
        next_token_logits = logits[:, -1, :]
        probs = F.softmax(next_token_logits / temperature, dim=-1)
        top_k_probs, top_k_indices = torch.topk(probs, k=top_k)
        
        sample = torch.multinomial(top_k_probs[0], num_samples=1)

        next_token = top_k_indices[0, sample]
        next_token = next_token.unsqueeze(-1)
        input_ids = torch.cat([input_ids, next_token], dim=1)
        next_token_decoded = tokenizer.decode(next_token[0].tolist())[0]
        if next_token_decoded == '<|endoftext|>':
            break

print(f"Result: {tokenizer.decode(input_ids[0].tolist())}", end="\r", flush=True)









