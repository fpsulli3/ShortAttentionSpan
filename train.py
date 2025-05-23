import torch
import tiktoken
from model.llm import GPTModel 
from model.dataset import create_llm_dataloader

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
model.train()
model.to(device)

file_dir = "/home/fpsulli3/llm/training"
data_loader = create_llm_dataloader(file_dir)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

save_every = 100
print_every = 10
running_loss = 0.0

num_epochs=300

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    running_loss = 0.0

    for batch_idx, (input_tokens, target_tokens) in enumerate(data_loader):
        input_tokens = input_tokens.to(device)
        target_tokens = target_tokens.to(device)

        logits = model(input_tokens)
        
        loss = loss_fn(
                logits.view(-1, GPT_CONFIG["vocab_size"]),
                target_tokens.view(-1),
            )

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()

        if (batch_idx + 1) % print_every == 0:
            avg_loss = running_loss / print_every
            print(f"Batch {batch_idx+1}: loss = {avg_loss:.4f}")
            running_loss = 0.0

        if (batch_idx + 1) % save_every == 0:
            torch.save(model.state_dict(), "gpt_weights.pt")


torch.save(model.state_dict(), "gpt_weights.pt")













