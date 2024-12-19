# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from transformers import GPT2Tokenizer
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Config

# 1. Load and preprocess the dataset
# Load dataset (use WikiText-103 for example)
dataset = load_dataset("wikitext", "wikitext-103-raw-v1")

# Load GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Set the pad_token to be the eos_token (End of Sequence token)
tokenizer.pad_token = tokenizer.eos_token

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length")

# Tokenize the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 2. Define the GPT-like Model
class GPT3LikeModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads):
        super(GPT3LikeModel, self).__init__()
        
        # Create the embedding layer
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        # Define a series of Transformer blocks (using a simple decoder architecture)
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads)
            for _ in range(num_layers)
        ])
        
        # Final linear layer for language modeling
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        
        # Pass through each transformer block
        for block in self.transformer_blocks:
            x = block(x)
        
        # Get logits for prediction
        logits = self.lm_head(x)
        return logits

# Parameters
vocab_size = tokenizer.vocab_size  # This is from the GPT2 tokenizer
hidden_size = 768  # Typical size for GPT-2 (adjust as needed)
num_layers = 12  # Typical number of layers for smaller GPT models
num_heads = 12  # Number of attention heads

# Create model
model = GPT3LikeModel(vocab_size, hidden_size, num_layers, num_heads)

# 3. Set up optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=5e-5)
criterion = nn.CrossEntropyLoss()

# 4. Set up a DataLoader for the tokenized dataset
dataloader = DataLoader(tokenized_datasets["train"], batch_size=4, shuffle=True)

# 5. Define the training loop
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids)
        logits = outputs.view(-1, logits.size(-1))  # Flatten logits for classification
        loss = criterion(logits, labels.view(-1))  # Flatten labels to match logits
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

# 6. Set up training device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 7. Training loop (for one epoch)
import torch

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch in dataloader:
        # Ensure input_ids and labels are tensors, then move to the device
        input_ids = torch.tensor(batch['input_ids']).to(device)
        labels = torch.tensor(batch['labels']).to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss

        # Backward pass
        loss.backward()

        # Optimize
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

# 8. Text Generation Function
def generate_text(model, prompt, max_length=50):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    outputs = model.generate(input_ids, max_length=max_length, num_beams=5, no_repeat_ngram_size=2)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 9. Example of text generation
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load a pre-trained GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def generate_text(model, prompt, max_length=50):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(inputs['input_ids'], max_length=max_length, num_return_sequences=1)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

prompt = "Once upon a time"
generated_text = generate_text(model, prompt)
print(generated_text)


# 10. Optionally, you can save the model after training
torch.save(model.state_dict(), "gpt3_like_model.pth")
