import torch
from transformers import GPT2Tokenizer
import torch.nn as nn

# Define the GPT3LikeModel (same as used during training)
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

# Load the tokenizer (same as used during training)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Ensure padding token is set correctly

# Parameters (make sure these match the parameters used during training)
vocab_size = tokenizer.vocab_size
hidden_size = 768
num_layers = 12
num_heads = 12

# Create the model (initialize it, but the weights will be loaded from the saved file)
model = GPT3LikeModel(vocab_size, hidden_size, num_layers, num_heads)

# Load the model weights from the saved .pth file
model.load_state_dict(torch.load("gpt3_like_model.pth"))

# Set the model to evaluation mode
model.eval()

# Define a function for text generation
def generate_text(model, prompt, max_length=50):
    model.eval()
    # Tokenize the prompt text
    input_ids = tokenizer.encode(prompt, return_tensors='pt')  # Convert prompt to input IDs

    # Generate text using the model
    with torch.no_grad():
        outputs = model(input_ids)  # Get model output
        logits = outputs[:, -1, :]  # Only consider the last token's logits (next token prediction)
        predicted_token_id = torch.argmax(logits, dim=-1)  # Get the most likely next token
        generated_ids = input_ids.squeeze().tolist() + [predicted_token_id.item()]  # Append generated token to input

        # Generate the rest of the sequence iteratively
        for _ in range(max_length - 1):
            input_ids = torch.tensor([generated_ids]).to(input_ids.device)  # Update input with new token
            outputs = model(input_ids)
            logits = outputs[:, -1, :]
            predicted_token_id = torch.argmax(logits, dim=-1)
            generated_ids.append(predicted_token_id.item())

        # Decode the generated token IDs to text
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        return generated_text

# Example: Generate text using a prompt
prompt = "Once upon a time"
generated_text = generate_text(model, prompt)
print(generated_text)

