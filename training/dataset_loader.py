"""
Dataset loading utilities for PC training
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
import os

def load_dataset(data_path):
    """
    Load dataset from CSV file
    Expected CSV format: column 'text' with stories
    """
    if not os.path.exists(data_path):
        # Create sample dataset if none exists
        print("📝 Creating sample dataset...")
        sample_data = {
            'text': [
                "Once upon a time there was a little rabbit.",
                "The sun was shining brightly in the sky.",
                "A brave knight went on an adventure.",
                "In a small village lived a happy family.",
                "The magical forest was full of wonders."
            ] * 1000  # 5000 samples
        }
        df = pd.DataFrame(sample_data)
        df.to_csv(data_path, index=False)
        print(f"✅ Created sample dataset: {data_path}")
    
    # Load actual dataset
    df = pd.read_csv(data_path)
    texts = df['text'].astype(str).tolist()
    
    # Split train/validation
    split_idx = int(0.9 * len(texts))
    train_texts = texts[:split_idx]
    val_texts = texts[split_idx:]
    
    print(f"📊 Loaded {len(train_texts)} training, {len(val_texts)} validation samples")
    return train_texts, val_texts

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