import os
import urllib.request
import tarfile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
import random
from tqdm import tqdm
import math
import re
from transformers import AutoTokenizer
import tempfile         
import shutil 

def compute_loss_and_perplexity(model, dataloader, tokenizer, criterion, device, max_sentences=1000):
    
    model.eval()

    total_loss = 0.0
    total_tokens = 0.0
    sentences_processed = 0

    with torch.no_grad():

        for input_seq, target_seq in tqdm(dataloader, desc="Evaluating", leave=False):

            input_seq = input_seq.to(device)      # Shape: (batch_size, seq_len)
            target_seq = target_seq.to(device)    # Shape: (batch_size, seq_len)

            batch_size_current = input_seq.size(0) # Get current batch size (might be smaller for last batch)

            logits = model(input_seq)  # Shape: (batch_size, seq_len, vocab_size)

            # Reshape logits and target for loss calculation
            logits = logits.reshape(-1, logits.size(-1))  # Shape: (batch_size * seq_len, vocab_size)
            target = target_seq.reshape(-1)              # Shape: (batch_size * seq_len)

            mask = target != tokenizer.pad_token_id 
            loss = criterion(logits[mask], target[mask])

            loss_value = loss.item() * mask.sum().item()  
            total_loss += loss_value                     
            total_tokens += mask.sum().item()    

            sentences_processed += batch_size_current
            if sentences_processed >= max_sentences:
                break

            average_loss = total_loss / total_tokens           # Normalize loss by number of tokens
            perplexity = math.exp(average_loss)               # Convert loss to perplexity

            return average_loss, perplexity
        
def generate_text(model, start_string, tokenizer, device, max_length=50):

    """ Generates text continuation from a given start string using greedy decoding."""

    model.eval()

    # Convert input string to token indices
    input_indices = tokenizer.encode(start_string, add_special_tokens=False)

    # Convert indices to tensor and move to specified device (GPU/CPU)
    input_tensor = torch.tensor([input_indices], dtype=torch.long).to(device)

    # Keep track of all generated tokens, starting with input sequence
    generated_indices = input_indices.copy()

    # Generate tokens until we hit max length or end-of-sequence token
    for _ in range(max_length - len(input_indices)):
        # Get model predictions for the entire sequence
        logits = model(input_tensor)
        # Only take predictions for the last token position
        logits = logits[:, -1, :]

        # Prevent the model from generating unknown tokens by setting their probability to negative infinity
        if tokenizer.unk_token_id is not None:
            logits[:, tokenizer.unk_token_id] = float("-inf")

        # Greedy decoding: select the token with highest probability
        next_token = torch.argmax(logits, dim=-1)

        # Add the chosen token to our generated sequence
        generated_indices.append(next_token.item())

        # If we generate an end-of-sequence token, stop generation
        if next_token.item() == tokenizer.eos_token_id:
            break

        # Add the new token to input tensor for next iteration
        input_tensor = torch.cat([input_tensor, next_token.unsqueeze(0)], dim=1)

    # Convert token indices back to text, removing any special tokens
    return tokenizer.decode(generated_indices, skip_special_tokens=True)


def save_model(model, tokenizer, model_name):
    
    # Create the models directory if it doesn't exist
    save_dir = os.path.join("models", model_name)
    os.makedirs(save_dir, exist_ok=True)

    # Save the model state dictionary and configuration
    model_path = os.path.join(save_dir, f"{model_name}.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": {
            "vocab_size": len(tokenizer),
            "emb_dim": model.embedding.embedding_dim,
            "num_heads": len(model.layers[0].attn.heads),
            "num_blocks": len(model.layers),
            "pad_idx": model.embedding.padding_idx
        }
    }, model_path)

    # Save the tokenizer
    tokenizer_path = os.path.join(save_dir, "tokenizer")
    tokenizer.save_pretrained(tokenizer_path)

    print(f"Model and tokenizer saved as '{model_name}'")

def load_model(model_name, device=None):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = os.path.join("models", model_name)

    # Check if model exists
    if not os.path.exists(save_dir):
        raise FileNotFoundError(f"No saved model found with name '{model_name}'")

    # Load the tokenizer
    tokenizer_path = os.path.join(save_dir, "tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Load the model state and config
    model_path = os.path.join(save_dir, f"{model_name}.pth")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Create a new model instance with the saved configuration
    model = DecoderLanguageModel(
        vocab_size=checkpoint["model_config"]["vocab_size"],
        emb_dim=checkpoint["model_config"]["emb_dim"],
        num_heads=checkpoint["model_config"]["num_heads"],
        num_blocks=checkpoint["model_config"]["num_blocks"],
        pad_idx=checkpoint["model_config"]["pad_idx"]
    )

    # Load the saved state dictionary
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"\nModel '{model_name}' loaded successfully")
    return model, tokenizer

def get_hyperparameters():
    emb_dim = 128
    num_heads = 8
    num_blocks = 2
    batch_size = 128
    learning_rate = 0.001
    num_epochs = 1
    context_size = 30
    return emb_dim, num_heads, num_blocks, batch_size, learning_rate, num_epochs, context_size


def initialize_weights(model):
    """
    Initialize the weights of different model components using appropriate schemes.
    Each layer type receives specialized initialization for optimal training.
    """
    for module in model.modules():
        if isinstance(module, nn.Linear):
            # Xavier uniform initialization for linear layers
            # Helps maintain variance across network layers
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)  # Initialize biases to zero
        elif isinstance(module, nn.Embedding):
            # Initialize embedding layers with normal distribution
            nn.init.normal_(module.weight, mean=0, std=0.02)
            if module.padding_idx is not None:
                # Ensure padding tokens have zero embeddings
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)
        elif isinstance(module, AttentionHead):
            # Initialize query, key, and value projection matrices
            # Xavier uniform helps maintain good gradient flow
            nn.init.xavier_uniform_(module.W_Q)
            nn.init.xavier_uniform_(module.W_K)
            nn.init.xavier_uniform_(module.W_V)
        elif isinstance(module, MultiHeadAttention):
            # Initialize output projection matrix for attention mechanism
            nn.init.xavier_uniform_(module.W_O)
        elif isinstance(module, DecoderLanguageModel):
            # Initialize final output projection layer
            nn.init.xavier_uniform_(module.output)
        elif isinstance(module, RMSNorm):
            # Initialize RMSNorm scale parameters to ones
            # This starts with identity transformation
            nn.init.ones_(module.scale)
        elif isinstance(module, MLP):
            # Initialize feed-forward network parameters
            nn.init.xavier_uniform_(module.W_1)
            nn.init.xavier_uniform_(module.W_2)
            nn.init.zeros_(module.B_1)
            nn.init.zeros_(module.B_2)

def rope(x, theta_base=10000.0):
    """
    Implements Rotary Position Embedding (RoPE) for transformer attention.
    RoPE encodes position information through rotation matrices applied to pairs of dimensions.
    """
    batch_size, seq_len, emb_dim = x.size()
    assert emb_dim % 2 == 0, "Embedding dimensionality must be even for RoPE"

    # Generate sequence position indices
    pos = torch.arange(0, seq_len, dtype=torch.float32, device=x.device)
    pos = pos.unsqueeze(0).expand(batch_size, seq_len)

    # Compute frequency bands for each dimension pair
    # Modified: frequencies start from p=1 and use (p-1) in exponent
    p = torch.arange(1, emb_dim // 2 + 1, dtype=torch.float32, device=x.device)
    theta_p = 1.0 / (theta_base ** (2 * (p - 1) / emb_dim))

    # Compute rotation angles for each position and frequency
    pos = pos.unsqueeze(-1)
    theta = pos * theta_p

    # Compute rotation components
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)

    # Split input into alternating dimensions
    x1 = x[..., 0::2]  # Dimensions at indices 0,2,4,...
    x2 = x[..., 1::2]  # Dimensions at indices 1,3,5,...

    # Apply 2D rotations to each pair
    x_rotated_1 = x1 * cos_theta - x2 * sin_theta
    x_rotated_2 = x1 * sin_theta + x2 * cos_theta

    # Recombine rotated pairs into final output
    x_rotated = torch.stack((x_rotated_1, x_rotated_2), dim=-1).reshape(batch_size, seq_len, emb_dim)

    return x_rotated

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    A simplified alternative to Layer Normalization that only uses RMS statistics
    """
    def __init__(self, emb_dim, epsilon=1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(emb_dim))  # Learnable scale parameter
        self.epsilon = epsilon  # Small constant for numerical stability

    def forward(self, x):
        # Compute root mean square normalization
        squared_x = x ** 2
        mean_squared = torch.mean(squared_x, dim=-1, keepdim=True)
        rms = torch.sqrt(mean_squared + self.epsilon)

        # Normalize and scale
        x_normalized = x / rms
        output = x_normalized * self.scale
        return output

class AttentionHead(nn.Module):
    """
    Single head of self-attention
    Transforms input using learned projections and computes scaled dot-product attention
    """
    def __init__(self, emb_dim, d_h):
        super().__init__()
        # Initialize projection matrices for queries, keys, and values
        self.W_Q = nn.Parameter(torch.rand(emb_dim, d_h))
        self.W_K = nn.Parameter(torch.rand(emb_dim, d_h))
        self.W_V = nn.Parameter(torch.rand(emb_dim, d_h))
        self.d_h = d_h  # Dimensionality of attention head

    def forward(self, x, mask):
        # Project input into query, key, and value spaces
        Q = x @ self.W_Q
        K = x @ self.W_K
        V = x @ self.W_V

        # Apply rotary position embeddings to queries and keys
        Q, K = rope(Q), rope(K)

        # Compute attention scores with scaling factor
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_h)

        # Apply causal mask and attention weights
        masked_scores = scores.masked_fill(mask == 0, float("-inf"))
        attention_weights = torch.softmax(masked_scores, dim=-1)

        return attention_weights @ V

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism
    Allows the model to jointly attend to information from different positions
    """
    def __init__(self, emb_dim, num_heads):
        super().__init__()
        d_h = emb_dim // num_heads  # Dimensionality of each attention head

        # Create multiple attention heads
        self.heads = nn.ModuleList([
            AttentionHead(emb_dim, d_h)
            for _ in range(num_heads)
        ])

        # Output projection matrix
        self.W_O = nn.Parameter(torch.rand(emb_dim, emb_dim))

    def forward(self, x, mask):
        # Process input through each attention head
        head_outputs = [head(x, mask) for head in self.heads]

        # Concatenate outputs and project to final dimensionality
        x = torch.cat(head_outputs, dim=-1)
        return x @ self.W_O

class MLP(nn.Module):
    """
    Multi-Layer Perceptron for transformer feed-forward network
    Uses a larger intermediate dimensionality (4x) with ReLU activation
    """
    def __init__(self, emb_dim):
        super().__init__()
        # Initialize weights and biases for two-layer feed-forward network
        self.W_1 = nn.Parameter(torch.rand(emb_dim, emb_dim * 4))
        self.B_1 = nn.Parameter(torch.rand(emb_dim * 4))
        self.W_2 = nn.Parameter(torch.rand(emb_dim * 4, emb_dim))
        self.B_2 = nn.Parameter(torch.rand(emb_dim))

    def forward(self, x):
        # First linear transformation and activation
        x = x @ self.W_1 + self.B_1
        x = torch.relu(x)

        # Second linear transformation
        x = x @ self.W_2 + self.B_2
        return x

class DecoderBlock(nn.Module):
    """
    Single transformer decoder block
    Combines self-attention and feed-forward layers with residual connections
    """
    def __init__(self, emb_dim, num_heads):
        super().__init__()
        # Layer components
        self.norm1 = RMSNorm(emb_dim)
        self.attn = MultiHeadAttention(emb_dim, num_heads)
        self.norm2 = RMSNorm(emb_dim)
        self.mlp = MLP(emb_dim)

    def forward(self, x, mask):
        # Self-attention sub-block with residual connection
        attn_out = self.attn(self.norm1(x), mask)
        x = x + attn_out

        # Feed-forward sub-block with residual connection
        mlp_out = self.mlp(self.norm2(x))
        x = x + mlp_out
        return x

class DecoderLanguageModel(nn.Module):
    """
    Complete decoder-only transformer language model
    Processes input sequences using multiple decoder blocks and projects to vocabulary
    """
    def __init__(self, vocab_size, emb_dim, num_heads, num_blocks, pad_idx):
        super().__init__()
        # Token embedding layer
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)

        # Stack of decoder blocks
        self.layers = nn.ModuleList([
            DecoderBlock(emb_dim, num_heads) for _ in range(num_blocks)
        ])

        # Output projection to vocabulary size
        self.output = nn.Parameter(torch.rand(emb_dim, vocab_size))

    def forward(self, x):
        # Embed input tokens
        x = self.embedding(x)

        # Create causal attention mask
        _, seq_len, _ = x.size()
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))

        # Process through decoder blocks
        for layer in self.layers:
            x = layer(x, mask)

        # Project to vocabulary distribution
        return x @ self.output