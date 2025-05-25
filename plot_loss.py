import pandas as pd
import matplotlib.pyplot as plt

# Load your logged CSV
df = pd.read_csv("outputs/wiki_test/loss.csv")

# Plot training and validation loss
plt.figure(figsize=(8, 5))
plt.plot(df["epoch"], df["train_loss"], label="Train Loss", marker="o")
plt.plot(df["epoch"], df["val_loss"], label="Validation Loss", marker="o")

plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("loss_curve.png")
plt.show()

