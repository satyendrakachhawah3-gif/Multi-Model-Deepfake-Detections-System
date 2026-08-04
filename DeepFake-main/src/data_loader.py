import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from datasets import load_dataset
import cv2
import torch

class MultiModalDataset(Dataset):
    """Unified multi-modal dataset"""
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        
        # Load CSV with paths
        self.samples = pd.read_csv(self.data_dir / f"{split}.csv")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        row = self.samples.iloc[idx]
        return {
            'text': row['text'],
            'image_path': row['image_path'],
            'video_path': row['video_path'],
            'label': row['label']
        }

def get_dataloaders(data_dir, batch_size=8):
    """Get train/val/test loaders"""
    transform = transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_ds = MultiModalDataset(data_dir, 'train', transform)
    val_ds = MultiModalDataset(data_dir, 'val', transform)
    
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(val_ds, batch_size=batch_size))
