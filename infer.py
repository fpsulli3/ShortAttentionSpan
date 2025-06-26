import torch
import tiktoken
import argparse
from model.llm import GPTModel 
from model.config import GPT_CONFIG_GPT2_SMALL
import torch.nn.functional as F

parser = argparse.ArgumentParser(description="Generate text from a prompt")
parser.add_argument("--prompt", type=str, required=True, help="Prompt to feed the model")
parser.add_argument("--model", type=str, required=True, help="The model weights file to load")
parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (default: 0.7)")
parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling (default: 50)")
parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling (default: 0.9)")
parser.add_argument("--max_tokens", type=int, default=225, help="Maximum tokens to generate (default: 225)")
parser.add_argument("--repetition_penalty", type=float, default=1.1, help="Repetition penalty (default: 1.1)")
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

def apply_repetition_penalty(logits, input_ids, penalty=1.1):
    """Apply repetition penalty to logits"""
    if penalty == 1.0:
        return logits
    
    for token_id in set(input_ids[0].tolist()):
        if logits[0, token_id] < 0:
            logits[0, token_id] *= penalty  # Make negative logits more negative
        else:
            logits[0, token_id] /= penalty  # Make positive logits less positive
    return logits

def top_p_sampling(logits, top_p=0.9):
    """Apply top-p (nucleus) sampling"""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    # Scatter back to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    logits[indices_to_remove] = float('-inf')
    return logits

# Stop tokens - only stop on end of text token
eot_token = tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]

with torch.no_grad():
    for n in range(args.max_tokens):
        logits = model(input_ids)
        next_token_logits = logits[:, -1, :].clone()
        
        # Apply repetition penalty
        next_token_logits = apply_repetition_penalty(next_token_logits, input_ids, args.repetition_penalty)
        
        # Apply temperature
        next_token_logits = next_token_logits / args.temperature
        
        # Apply top-p sampling
        next_token_logits = top_p_sampling(next_token_logits, args.top_p)
        
        # Apply top-k sampling
        if args.top_k > 0:
            top_k_logits, top_k_indices = torch.topk(next_token_logits, k=min(args.top_k, next_token_logits.size(-1)))
            probs = F.softmax(top_k_logits, dim=-1)
            sample = torch.multinomial(probs, num_samples=1)
            next_token = top_k_indices.gather(-1, sample)
        else:
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        
        # Stop if we hit the end of text token (but don't include it in output)
        if next_token.item() == eot_token:
            break
            
        input_ids = torch.cat([input_ids, next_token], dim=1)
        
        # Truncate context if too long
        if input_ids.size(1) > GPT_CONFIG["context_length"]:
            input_ids = input_ids[:, -GPT_CONFIG["context_length"]:]

    result = tokenizer.decode(input_ids[0].tolist())
    print(f"Result:\n\n{result}")









