import os
import yaml
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from utils import AstroBaselineDataset
from models import SimpleMLP, SimpleCNN

def train():
    # 1. Load Configuration
    with open("./Baseline/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Starting training on device: {device}")
    os.makedirs(config['paths']['checkpoint_dir'], exist_ok=True)

    # 2. Data Loading & Splitting
    print(f"📂 Loading manifest from {config['data']['manifest_path']}...")
    df = pd.read_csv(config['data']['manifest_path'])
    
    if config['data'].get('filter_duration'):
        print(f"🔍 Filtering for duration: {config['data']['filter_duration']}")
        df = df[df['duration'] == config['data']['filter_duration']].reset_index(drop=True)

    
    unique_kics = df['kic'].unique()
    np.random.seed(42)
    np.random.shuffle(unique_kics)

    sample_size = config['data'].get('sample_size')
    if sample_size:
        unique_kics = unique_kics[:sample_size]
        df = df[df['kic'].isin(unique_kics)].reset_index(drop=True)

    train_size = int(0.8 * len(unique_kics))
    val_size = int(0.1 * len(unique_kics))
    
    train_kics = unique_kics[:train_size]
    val_kics = unique_kics[train_size:train_size+val_size]
    test_kics = unique_kics[train_size+val_size:]

    full_ds = AstroBaselineDataset(df, mode=config['data']['mode'], 
                                   target_length=config['data']['target_length'])
    
    train_idx = df[df['kic'].isin(train_kics)].index.tolist()
    val_idx = df[df['kic'].isin(val_kics)].index.tolist()
    test_idx = df[df['kic'].isin(test_kics)].index.tolist()

    print(f"📊 Dataset Split: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    train_loader = DataLoader(Subset(full_ds, train_idx), batch_size=config['data']['batch_size'], shuffle=True)
    val_loader = DataLoader(Subset(full_ds, val_idx), batch_size=config['data']['batch_size'])
    test_loader = DataLoader(Subset(full_ds, test_idx), batch_size=config['data']['batch_size'])

    # 3. Model Initialization
    print(f"🏗️  Initializing {config['model']['type'].upper()} model...")
    if config['model']['type'] == 'cnn':
        model = SimpleCNN()
    else:
        model = SimpleMLP(input_size=config['data']['target_length'])
    
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['model']['learning_rate'])
    criterion = nn.MSELoss()

    # 4. Training Loop
    best_val_loss = float('inf')
    print("\n--- Training Started ---")
    
    for epoch in range(config['model']['epochs']):
        model.train()
        total_train_loss = 0
        
        # Batch Printing
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), torch.log10(y).to(device)
            
            optimizer.zero_grad()
            preds = model(x).squeeze()
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
            # Print every 50 batches
            if i % 50 == 0:
                print(f"  Epoch [{epoch+1}/{config['model']['epochs']}] | Batch [{i}/{len(train_loader)}] | Loss: {loss.item():.4f}")

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), torch.log10(y).to(device)
                preds = model(x).squeeze()
                total_val_loss += criterion(preds, y).item()
        
        avg_train = total_train_loss / len(train_loader)
        avg_val = total_val_loss / len(val_loader)
        
        print(f"✅ End of Epoch {epoch+1}: Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), f"{config['paths']['checkpoint_dir']}/best_model.pth")
            print(f"⭐ New best model saved with Val Loss: {best_val_loss:.4f}")

    # 5. Final Evaluation
    print("\n--- Final Evaluation on Test Set ---")
    evaluate_model(model, test_loader, device)

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_truth = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            preds = model(x).squeeze().cpu().numpy()
            all_preds.extend(preds)
            all_truth.extend(np.log10(y.numpy()))

    plt.figure(figsize=(8, 6))
    plt.scatter(all_truth, all_preds, alpha=0.5, s=10)
    plt.plot([min(all_truth), max(all_truth)], [min(all_truth), max(all_truth)], 'r--')
    plt.xlabel("True Log10(nu_max)")
    plt.ylabel("Predicted Log10(nu_max)")
    plt.title("Baseline Model Performance")
    plt.savefig("./Baseline/results/performance_plot.png")
    print("📈 Test evaluation plot saved to: results/performance_plot.png")

if __name__ == "__main__":
    train()