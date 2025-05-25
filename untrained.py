import torch
import tiktoken
import argparse
from model.llm import GPTModel 
from model.config import GPT_CONFIG_WIKI
import torch.nn.functional as F

parser = argparse.ArgumentParser(description="Generate text from a prompt")
parser.add_argument("--prompt", type=str, required=True, help="Prompt to feed the model")
parser.add_argument("--model", type=str, required=True, help="The model weights file to load")
args = parser.parse_args()

prompt = args.prompt

GPT_CONFIG = GPT_CONFIG_WIKI

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"The device is ${device}")

tokenizer = tiktoken.get_encoding("gpt2")
GPT_CONFIG["vocab_size"] = tokenizer.n_vocab

model = GPTModel(cfg=GPT_CONFIG)
#model.load_state_dict(torch.load(args.model))
model.to(device)
model.eval()

input_ids = tokenizer.encode(prompt)
input_ids = torch.tensor([input_ids]).to(device)

temperature = 0.8
top_k = 50

with torch.no_grad():
    for n in range(100):
        logits = model(input_ids)
        next_token_logits = logits[:, -1, :]
        probs = F.softmax(next_token_logits / temperature, dim=-1)
        top_k_probs, top_k_indices = torch.topk(probs, k=top_k)
        
        sample = torch.multinomial(top_k_probs[0], num_samples=1)

        next_token = top_k_indices[0, sample]
        input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=1)

    result = tokenizer.decode(input_ids[0].tolist())
    print(f"Result:\n\n {result}")









