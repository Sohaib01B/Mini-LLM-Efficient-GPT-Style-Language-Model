"""
Mini-GPT Model Architecture
Same as Kaggle version
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# kaggle version 
"""
🚀 Mini-LLM: Efficient GPT-Style Language Model
Optimized for Kaggle GPU (T4 - 15GB VRAM)
KAGGLE VERSION - Handles large datasets (1.8GB+)
"""

# ========================================
# 1. INSTALLATION & IMPORTS
# ========================================

!pip install -q transformers datasets accelerate einops wandb tiktoken

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import pandas as pd
import numpy as np
import math
import time
from pathlib import Path
from tqdm.auto import tqdm
import gc
import warnings
warnings.filterwarnings('ignore')

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🎮 Device: {device}")
if torch.cuda.is_available():
    print(f"📊 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ========================================
# 2. LOAD DATASET FROM KAGGLE INPUT
# ========================================

# Verify dataset
print("\n📁 Verification du dataset:")
!ls -la "/kaggle/input/tinystories-narrative-classification"
print("\n📂 Train file:")
!ls -la "/kaggle/input/tinystories-narrative-classification/train.csv"
print("\n📂 Validation file:")
!ls -la "/kaggle/input/tinystories-narrative-classification/validation.csv"
print("good")

# ========================================
# OPTIMIZED LOADING FOR LARGE FILES (1.8GB+)
# ========================================

print("\n📊 Loading LARGE dataset with optimizations...")

# First, peek at the file structure (read only 1000 rows)
print("🔍 Analyzing file structure...")
sample_df = pd.read_csv("/kaggle/input/tinystories-narrative-classification/train.csv", nrows=1000)
print(f"📋 Columns: {sample_df.columns.tolist()}")
print(f"🔍 Sample data:\n{sample_df.head(2)}")

# Determine text column (adjust if needed)
text_column = sample_df.columns[0]  # Change index if needed
print(f"\n✅ Using column: '{text_column}'")

# IMPROVED: Load more samples for better quality
print("\n⚡ Loading data in chunks (memory efficient)...")
SAMPLE_SIZE = 200000  # DOUBLED: 200K samples for better quality
chunk_size = 10000

train_texts = []
chunk_count = 0

for chunk in pd.read_csv("/kaggle/input/tinystories-narrative-classification/train.csv", chunksize=chunk_size):
    train_texts.extend(chunk[text_column].astype(str).tolist())
    chunk_count += 1
    if len(train_texts) >= SAMPLE_SIZE:
        train_texts = train_texts[:SAMPLE_SIZE]
        break
    print(f"📦 Loaded {len(train_texts):,} samples...", end='\r')

print(f"\n✅ Train samples loaded: {len(train_texts):,}")

# Load validation (usually smaller)
print("📊 Loading validation data...")
try:
    val_texts = []
    for chunk in pd.read_csv("/kaggle/input/tinystories-narrative-classification/validation.csv", chunksize=chunk_size):
        val_texts.extend(chunk[text_column].astype(str).tolist())
        if len(val_texts) >= 10000:  # Limit validation to 10K
            val_texts = val_texts[:10000]
            break
    print(f"✅ Validation samples: {len(val_texts):,}")
except Exception as e:
    print(f"⚠️  Validation file issue: {e}")
    # Use 10% of train as validation
    val_size = len(train_texts) // 10
    val_texts = train_texts[-val_size:]
    train_texts = train_texts[:-val_size]
    print(f"✅ Created validation split: {len(val_texts):,} samples")

# Clear memory
del sample_df
gc.collect()
torch.cuda.empty_cache()

print("\n💾 Memory optimized!")

# ========================================
# 3. OPTIMIZED TOKENIZER FOR LARGE DATASETS
# ========================================

class SimpleTokenizer:
    """Fast character-level tokenizer"""
    def __init__(self, texts, max_samples_for_vocab=10000):
        # Build vocab from subset (faster for large datasets)
        sample_texts = texts[:max_samples_for_vocab] if len(texts) > max_samples_for_vocab else texts

        print(f"📖 Building vocabulary from {len(sample_texts):,} samples...")
        all_chars = set()
        for text in sample_texts:
            all_chars.update(text)

        self.chars = sorted(list(all_chars))

        # Special tokens
        self.pad_token = '<PAD>'
        self.sos_token = '<SOS>'
        self.eos_token = '<EOS>'
        self.unk_token = '<UNK>'

        special_tokens = [self.pad_token, self.sos_token, self.eos_token, self.unk_token]
        self.vocab = special_tokens + self.chars

        # Create mappings
        self.char_to_idx = {ch: idx for idx, ch in enumerate(self.vocab)}
        self.idx_to_char = {idx: ch for idx, ch in enumerate(self.vocab)}

        self.vocab_size = len(self.vocab)
        self.pad_token_id = 0
        self.sos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3

        print(f"✅ Vocabulary size: {self.vocab_size}")

    def encode(self, text, add_special_tokens=True):
        tokens = [self.char_to_idx.get(ch, self.unk_token_id) for ch in text]
        if add_special_tokens:
            tokens = [self.sos_token_id] + tokens + [self.eos_token_id]
        return tokens

    def decode(self, tokens):
        chars = [self.idx_to_char.get(t, self.unk_token) for t in tokens
                if t not in [self.pad_token_id, self.sos_token_id, self.eos_token_id]]
        return ''.join(chars)

# Build tokenizer from training data
tokenizer = SimpleTokenizer(train_texts)

# ========================================
# 4. DATASET CLASS
# ========================================

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=256):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        tokens = self.tokenizer.encode(text)

        # Truncate if too long
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]

        # Pad if too short
        padding_length = self.max_length - len(tokens)
        tokens = tokens + [self.tokenizer.pad_token_id] * padding_length

        # Input: all tokens except last, Target: all tokens except first
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)

        return input_ids, labels

# Create datasets (now using text lists, not DataFrames)
MAX_LENGTH = 256  # Sequence length (adjust based on your data)
train_dataset = TextDataset(train_texts, tokenizer, MAX_LENGTH)
val_dataset = TextDataset(val_texts, tokenizer, MAX_LENGTH)

# Clear memory
del train_texts, val_texts
gc.collect()
torch.cuda.empty_cache()

print(f"\n✅ Datasets created!")
print(f"   • Training: {len(train_dataset):,} samples")
print(f"   • Validation: {len(val_dataset):,} samples")

# ========================================
# 5. MINI-GPT MODEL ARCHITECTURE
# ========================================

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, C = x.shape

        # Calculate Q, K, V
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply causal mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = attn @ v  # (B, num_heads, T, head_dim)
        out = out.transpose(1, 2).reshape(B, T, C)  # (B, T, C)
        out = self.proj(out)

        return out

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(self, x, mask=None):
        # Pre-norm architecture (more stable)
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ffn(self.ln2(x))
        return x

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_layers=6, num_heads=8,
                 d_ff=2048, max_seq_len=512, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Output
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying (share embeddings with output)
        self.head.weight = self.token_emb.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Count parameters
        n_params = sum(p.numel() for p in self.parameters())
        print(f"🧠 Model Parameters: {n_params:,} ({n_params/1e6:.2f}M)")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None):
        B, T = x.shape

        # Create causal mask (lower triangular)
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)

        # Embeddings
        tok_emb = self.token_emb(x)
        pos_emb = self.pos_emb(torch.arange(T, device=x.device))
        x = self.dropout(tok_emb + pos_emb)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask)

        # Output
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Generate text autoregressively"""
        self.eval()
        for _ in range(max_new_tokens):
            # Crop context if needed
            idx_cond = idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len:]

            # Forward pass
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

# ========================================
# 6. TRAINING CONFIGURATION (OPTIMIZED FOR BETTER QUALITY)
# ========================================

# Model hyperparameters (IMPROVED for better quality)
MODEL_CONFIG = {
    'vocab_size': tokenizer.vocab_size,
    'd_model': 512,           # INCREASED: 384→512 for better representations
    'num_layers': 8,          # INCREASED: 6→8 for deeper understanding
    'num_heads': 8,           # INCREASED: 6→8 for better attention
    'd_ff': 2048,             # INCREASED: 1536→2048 (4x d_model)
    'max_seq_len': MAX_LENGTH,
    'dropout': 0.1
}

# Training hyperparameters (OPTIMIZED for quality)
BATCH_SIZE = 32              # REDUCED: 48→32 (more stable with larger model)
LEARNING_RATE = 3e-4         # REDUCED: 5e-4→3e-4 (more stable learning)
WEIGHT_DECAY = 0.1
WARMUP_STEPS = 2000          # INCREASED: 1000→2000 (longer warmup)
MAX_EPOCHS = 8               # INCREASED: 5→8 (more training)
GRAD_CLIP = 1.0
EVAL_INTERVAL = 500          # Evaluate every N steps
SAVE_INTERVAL = 2000         # Save checkpoint every N steps

print("\n" + "="*60)
print("⚙️  TRAINING CONFIGURATION (IMPROVED)")
print("="*60)
print(f"📊 Data:")
print(f"   • Training samples: {len(train_dataset):,}")
print(f"   • Validation samples: {len(val_dataset):,}")
print(f"   • Sequence length: {MAX_LENGTH}")
print(f"\n🧠 Model (LARGER):")
print(f"   • Embedding dim: {MODEL_CONFIG['d_model']}")
print(f"   • Layers: {MODEL_CONFIG['num_layers']}")
print(f"   • Attention heads: {MODEL_CONFIG['num_heads']}")
print(f"\n🔧 Training (MORE EPOCHS):")
print(f"   • Batch size: {BATCH_SIZE}")
print(f"   • Epochs: {MAX_EPOCHS}")
print(f"   • Learning rate: {LEARNING_RATE}")
print(f"   • Warmup steps: {WARMUP_STEPS}")
print("="*60)

# Create model
model = MiniGPT(**MODEL_CONFIG).to(device)

# Optimizer (AdamW with weight decay)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    betas=(0.9, 0.95),
    weight_decay=WEIGHT_DECAY
)

# Learning rate scheduler (cosine with warmup)
def get_lr(step, warmup_steps, max_steps):
    if step < warmup_steps:
        return LEARNING_RATE * step / warmup_steps
    if step > max_steps:
        return LEARNING_RATE * 0.1
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return LEARNING_RATE * 0.1 + coeff * (LEARNING_RATE - LEARNING_RATE * 0.1)

# Data loaders (num_workers=0 to avoid multiprocessing warnings)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                       num_workers=0, pin_memory=True)

# Mixed precision training
scaler = GradScaler()

# ========================================
# 7. TRAINING TIME ESTIMATION
# ========================================

def estimate_training_time():
    """Estimate total training time"""
    samples_per_epoch = len(train_dataset)
    batches_per_epoch = len(train_loader)
    total_batches = batches_per_epoch * MAX_EPOCHS

    # Rough estimates for T4 GPU (based on model size)
    # ~23M params model processes ~50-80 batches/sec on T4
    ms_per_batch = 150  # milliseconds (conservative estimate)

    total_time_sec = (total_batches * ms_per_batch) / 1000
    total_time_min = total_time_sec / 60
    total_time_hr = total_time_min / 60

    print("\n" + "="*60)
    print("⏱️  TRAINING TIME ESTIMATION")
    print("="*60)
    print(f"📊 Dataset info:")
    print(f"   • Training samples: {len(train_dataset):,}")
    print(f"   • Validation samples: {len(val_dataset):,}")
    print(f"   • Batch size: {BATCH_SIZE}")
    print(f"   • Batches per epoch: {batches_per_epoch:,}")
    print(f"\n🔧 Training config:")
    print(f"   • Epochs: {MAX_EPOCHS}")
    print(f"   • Total batches: {total_batches:,}")
    print(f"   • Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"\n⏰ Estimated time:")
    print(f"   • Per batch: ~{ms_per_batch}ms")
    print(f"   • Per epoch: ~{total_time_min/MAX_EPOCHS:.1f} minutes")
    print(f"   • TOTAL: ~{total_time_hr:.1f} hours ({total_time_min:.0f} minutes)")

    if total_time_hr < 1:
        print(f"\n✅ Quick training: {total_time_min:.0f} minutes!")
    elif total_time_hr < 3:
        print(f"\n✅ Reasonable: ~{total_time_hr:.1f} hours")
    else:
        print(f"\n⚠️  Long training: {total_time_hr:.1f} hours")
        print(f"   💡 Consider: reduce epochs or data size")

    print("="*60 + "\n")

    return total_time_hr

# Run estimation
estimated_hours = estimate_training_time()

# ========================================
# 8. TRAINING LOOP
# ========================================

def evaluate(model, dataloader, max_batches=50):
    """Evaluate model on validation set"""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            if i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            with autocast():
                _, loss = model(x, y)
            total_loss += loss.item()
    model.train()
    return total_loss / min(len(dataloader), max_batches)

def train():
    """Main training function"""
    model.train()
    step = 0
    best_val_loss = float('inf')

    print("\n" + "="*60)
    print("🚀 TRAINING STARTED")
    print("="*60)

    max_steps = len(train_loader) * MAX_EPOCHS

    for epoch in range(MAX_EPOCHS):
        print(f"\n📅 Epoch {epoch+1}/{MAX_EPOCHS}")
        pbar = tqdm(train_loader, desc=f"Training")

        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)

            # Update learning rate
            lr = get_lr(step, WARMUP_STEPS, max_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # Forward pass with mixed precision
            optimizer.zero_grad()
            with autocast():
                logits, loss = model(x, y)

            # Backward pass
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{lr:.2e}',
                'step': step
            })

            # Evaluation
            if step % EVAL_INTERVAL == 0 and step > 0:
                val_loss = evaluate(model, val_loader)
                print(f"\n📊 Step {step} | Val Loss: {val_loss:.4f}")

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({
                        'step': step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss,
                        'config': MODEL_CONFIG
                    }, '/kaggle/working/best_model.pt')
                    print(f"💾 Best model saved! (Val Loss: {val_loss:.4f})")

            # Save checkpoint
            if step % SAVE_INTERVAL == 0 and step > 0:
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': MODEL_CONFIG
                }, f'/kaggle/working/checkpoint_step_{step}.pt')

            step += 1

    print("\n" + "="*60)
    print("✅ TRAINING COMPLETED!")
    print("="*60)

# Start training
train()

# ========================================
# 9. TEXT GENERATION
# ========================================

def generate_text(prompt, max_length=200, temperature=0.75, top_k=45):
    """Generate text from a prompt with OPTIMIZED defaults"""
    model.eval()

    # Encode prompt
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    x = torch.tensor([tokens], dtype=torch.long).to(device)

    # Generate
    with torch.no_grad():
        y = model.generate(x, max_new_tokens=max_length, temperature=temperature, top_k=top_k)

    # Decode
    generated = tokenizer.decode(y[0].tolist())
    return generated

# Load best model
checkpoint = torch.load('/kaggle/working/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
print(f"\n✅ Loaded best model (Val Loss: {checkpoint['val_loss']:.4f})")

# Test generation
print("\n" + "="*60)
print("🎨 TEXT GENERATION EXAMPLES")
print("="*60)

test_prompts = [
    "Once upon a time",
    "One day, a little",
    "There was a"
]

for prompt in test_prompts:
    print(f"\n🔹 Prompt: '{prompt}'")
    # Use better generation parameters
    generated = generate_text(prompt, max_length=150, temperature=0.75, top_k=45)
    print(f"📝 Generated: {generated}")
    print("-" * 60)

# ========================================
# 10. INTERACTIVE GENERATION WITH BETTER CONTROLS
# ========================================

print("\n💬 Interactive Mode (type 'quit' to exit)")
print("💡 Tips: Try 'params' to adjust generation settings\n")

# Default generation parameters
gen_temp = 0.75
gen_topk = 45
gen_length = 200

while True:
    prompt = input("\n🎯 Enter prompt (or 'params' to adjust settings): ")

    if prompt.lower() == 'quit':
        break

    if prompt.lower() == 'params':
        print("\n⚙️  Current Settings:")
        print(f"   • Temperature: {gen_temp}")
        print(f"   • Top-K: {gen_topk}")
        print(f"   • Max Length: {gen_length}")

        try:
            new_temp = input(f"\nNew temperature (0.5-1.5, current {gen_temp}): ").strip()
            if new_temp:
                gen_temp = float(new_temp)

            new_topk = input(f"New top-k (20-100, current {gen_topk}): ").strip()
            if new_topk:
                gen_topk = int(new_topk)

            new_length = input(f"New max length (50-500, current {gen_length}): ").strip()
            if new_length:
                gen_length = int(new_length)

            print(f"✅ Settings updated!")
        except:
            print("⚠️  Invalid input, keeping current settings")
        continue

    generated = generate_text(prompt, max_length=gen_length, temperature=gen_temp, top_k=gen_topk)
    print(f"\n✨ Generated:\n{generated}\n")

# ========================================
# 11. SAVE FINAL MODEL
# ========================================

print("\n💾 Model saved to /kaggle/working/")
print("✅ Files available:")
print("   • best_model.pt (best validation loss)")
print("   • checkpoint_step_*.pt (periodic checkpoints)")

# ========================================
# 12. GENERATION PARAMETERS TUNING GUIDE
# ========================================

print("\n" + "="*60)
print("🎛️  GENERATION PARAMETER GUIDE")
print("="*60)
print("""
Adjust these parameters for different text styles:

🌡️ TEMPERATURE (creativity):
   • 0.5-0.7  → More focused, repetitive, predictable
   • 0.8-0.9  → Balanced (RECOMMENDED)
   • 1.0-1.2  → Creative, diverse
   • 1.5+     → Very random, chaotic
   

🎯 TOP_K (diversity):
   • 10-20    → Conservative choices
   • 30-50    → Good variety (RECOMMENDED)
   • 100+     → Maximum diversity

📏 MAX_LENGTH:
   • 100-150  → Short stories
   • 200-300  → Medium (RECOMMENDED)
   • 500+     → Long generation

Example usage:
   generate_text("Once upon", max_length=300, temperature=0.9, top_k=50)
""")
print("="*60)

print("\n✅ Done! Models saved in /kaggle/working/")

# ========================================
# 13. SAUVEGARDE DU TOKENIZER
# ========================================

import pickle
import json

# Sauvegarder le tokenizer
tokenizer_data = {
    'vocab': tokenizer.vocab,
    'char_to_idx': tokenizer.char_to_idx,
    'idx_to_char': tokenizer.idx_to_char,
    'vocab_size': tokenizer.vocab_size,
    'pad_token_id': tokenizer.pad_token_id,
    'sos_token_id': tokenizer.sos_token_id,
    'eos_token_id': tokenizer.eos_token_id,
    'unk_token_id': tokenizer.unk_token_id
}

with open('/kaggle/working/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer_data, f)

print("✅ Tokenizer sauvegardé!")

# Sauvegarder aussi en JSON pour inspection
tokenizer_info = {
    'vocab_size': tokenizer.vocab_size,
    'special_tokens': {
        'pad_token': tokenizer.pad_token,
        'sos_token': tokenizer.sos_token,
        'eos_token': tokenizer.eos_token,
        'unk_token': tokenizer.unk_token
    },
    'sample_chars': tokenizer.chars[:20]  # Premiers 20 caractères
}

with open('/kaggle/working/tokenizer_info.json', 'w') as f:
    json.dump(tokenizer_info, f, indent=2)