"""
🚀 Mini-LLM Training - PC Version
Optimized for local GPU/CPU training
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import math
from tqdm import tqdm
import os
from dataset_loader import load_dataset, TextDataset
from model import MiniGPT
from tokenizer import SimpleTokenizer

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🎮 Training on: {self.device}")
        
        # Load data
        print("📊 Loading dataset...")
        train_texts, val_texts = load_dataset(config['data_path'])
        
        # Create tokenizer
        self.tokenizer = SimpleTokenizer(train_texts)
        
        # Create datasets
        train_dataset = TextDataset(train_texts, self.tokenizer, config['max_length'])
        val_dataset = TextDataset(val_texts, self.tokenizer, config['max_length'])
        
        # Data loaders
        self.train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
        
        # Model
        model_config = config['model']
        model_config['vocab_size'] = self.tokenizer.vocab_size
        self.model = MiniGPT(**model_config).to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        print(f"🧠 Model Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"📚 Training samples: {len(train_dataset):,}")
        print(f"📚 Validation samples: {len(val_dataset):,}")
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(self.device), y.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            _, loss = self.model(x, y)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                _, loss = self.model(x, y)
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'tokenizer': {
                'char_to_idx': self.tokenizer.char_to_idx,
                'idx_to_char': self.tokenizer.idx_to_char,
                'vocab_size': self.tokenizer.vocab_size,
                'pad_token_id': self.tokenizer.pad_token_id,
                'sos_token_id': self.tokenizer.sos_token_id,
                'eos_token_id': self.tokenizer.eos_token_id,
                'unk_token_id': self.tokenizer.unk_token_id,
                'chars': self.tokenizer.chars
            },
            'config': self.config
        }
        
        filename = f"best_model.pt" if is_best else f"checkpoint_epoch_{epoch+1}.pt"
        torch.save(checkpoint, f"saved_models/{filename}")
        print(f"💾 Saved: {filename} (val_loss: {val_loss:.4f})")
    
    def train(self):
        best_val_loss = float('inf')
        
        print("\n🚀 Starting Training...")
        for epoch in range(self.config['epochs']):
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss = self.validate()
            
            print(f"📊 Epoch {epoch+1}/{self.config['epochs']} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss, is_best=True)
            
            # Save checkpoint every few epochs
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch, val_loss)

def main():
    # Configuration for PC training
    config = {
        'data_path': 'datasets/stories.csv',  # Users will put their data here
        'max_length': 256,
        'batch_size': 16,  # Smaller for PC GPUs
        'learning_rate': 3e-4,
        'weight_decay': 0.1,
        'epochs': 10,
        'model': {
            'd_model': 512,
            'num_layers': 6,  # Smaller for PC training
            'num_heads': 8,
            'd_ff': 2048,
            'max_seq_len': 256,
            'dropout': 0.1
        }
    }
    
    # Create saved_models directory
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('datasets', exist_ok=True)
    
    # Start training
    trainer = Trainer(config)
    trainer.train()

if __name__ == '__main__':
    main()