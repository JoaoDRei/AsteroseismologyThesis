from logging import config
import os
import yaml
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from utils import AstroBaselineDataset
from models import SimpleMLP, SimpleCNN, SimpleTransformer
import datetime
import shutil

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
    elif config['model']['type'] == 'mlp':
        model = SimpleMLP(input_size=full_ds.target_length)
    elif config['model']['type'] == 'transformer':
        model = SimpleTransformer(seq_length=config['data']['seq_length'])
    else:
        raise ValueError("Unknown model type")
    
    
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['model']['learning_rate'])
    #adaptive learning rate:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    loss_name = config['model'].get('loss_function', 'mse')
    if loss_name == 'mse':
        criterion = nn.MSELoss()
    elif loss_name == 'l1':
        criterion = nn.L1Loss()
    elif loss_name == 'huber':
        criterion = nn.SmoothL1Loss()
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")

    # 4. Training Loop
    best_val_loss = float('inf')
    print("\n--- Training Started ---")
    
    for epoch in range(config['model']['epochs']):
        model.train()
        total_train_loss = 0
        
        # Batch Printing
        for i, (x, y, _) in enumerate(train_loader):
            x, y = x.to(device), torch.log10(y + 1e-8).to(device)
            
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
            for x, y, _ in val_loader:
                x, y = x.to(device), torch.log10(y + 1e-8).to(device)
                preds = model(x).squeeze()
                total_val_loss += criterion(preds, y).item()
        
        if len(train_loader) == 0:
            raise ValueError("Empty training loader")
        avg_train = total_train_loss / len(train_loader)
        avg_val = total_val_loss / len(val_loader)
        
        print(f"✅ End of Epoch {epoch+1}: Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

        # Update the learning rate based on validation loss
        scheduler.step(avg_val)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current Learning Rate: {current_lr}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), f"{config['paths']['checkpoint_dir']}/best_model.pth")
            print(f"⭐ New best model saved with Val Loss: {best_val_loss:.4f}")

    # 5. Final Evaluation
    print("\n--- Final Evaluation on Test Set ---")
    model.load_state_dict(torch.load(f"{config['paths']['checkpoint_dir']}/best_model.pth"))   
    evaluate_model(model, test_loader, device, criterion, config['data']['manifest_path'],test_idx)

def evaluate_model(model, test_loader, device, criterion, manifest_path, test_idx):
    
    df=pd.read_csv(manifest_path)
    model.eval()

    all_preds = []
    all_truth = []
    all_kics = []

    total_test_loss = 0
    # Load Configuration
    with open("./Baseline/config.yaml", "r") as f:
        config = yaml.safe_load(f)


    # Create a unique name based on the current time and model type
    exp_name = f"{config['model']['type']}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    exp_dir = f"./Baseline/results/{exp_name}"
    os.makedirs(exp_dir, exist_ok=True)
    
    # Copy the config file into that folder so you have a permanent record
    with open(f"{exp_dir}/config.yaml", "w") as f:
        yaml.dump(config, f)
    
    # copy the model architecture file into the folder as well for reproducibility
    model_source_file = f"./Baseline/models/{config['model']['type']}.py" 
    shutil.copy(model_source_file, f"{exp_dir}/model_architecture.py")
    #copy the best model checkpoint into the folder as well
    shutil.copy(f"{config['paths']['checkpoint_dir']}/best_model.pth", f"{exp_dir}/best_model.pth")


    with torch.no_grad():
        for x, y, kic in test_loader:
            x, y = x.to(device), torch.log10(y + 1e-8).to(device)
            preds = model(x).squeeze()

            # Calculate loss
            loss = criterion(preds, y)
            total_test_loss += loss.item()

            all_preds.extend(preds.cpu().numpy())
            all_truth.extend(y.cpu().numpy())
            all_kics.extend(kic.numpy())
    avg_test_loss = total_test_loss / len(test_loader)

    results_df=pd.DataFrame({
        "kic": all_kics,
        "truth_log": all_truth,
        "pred_log": all_preds,
        })

    merged=results_df.merge(df[['kic','nu_max_hon', 'nu_max']], on='kic', how='left')
    merged['hon_log']=np.log10(merged['nu_max_hon'] + 1e-8)
    merged['truth_log_check']=np.log10(merged['nu_max'] + 1e-8)

    validHon=merged['nu_max_hon'] > 0
    hon_mse=np.mean((merged.loc[validHon,'hon_log'] - merged.loc[validHon,'truth_log'])**2)
    

    # Save the loss metric to a file
    with open(f"{exp_dir}/metrics.txt", "w") as f:
        f.write(f"Final Test Loss (MSE): {avg_test_loss:.4f}\n")
        f.write(f"HON MSE: {hon_mse:.4f}\n")

    savepath = f"{exp_dir}/performance_plot.png"
    plt.figure(figsize=(8, 6))
    plt.scatter(merged['truth_log'], merged['pred_log'],alpha=0.5, s=10, label="Model", color='blue')
    plt.scatter(merged.loc[validHon, 'truth_log'],merged.loc[validHon, 'hon_log'],alpha=0.3, s=5, label="Hon 2019", color='orange')
    plt.plot([min(all_truth), max(all_truth)], [min(all_truth), max(all_truth)], 'r--')
    plt.xlabel("True Log10(nu_max)")
    plt.ylabel("Predicted Log10(nu_max)")
    plt.title(f"{config['data']['mode'].upper()} {config['model']['type'].upper()} {config['data']['filter_duration']}  Model Performance")
    plt.legend()
    plt.savefig(savepath)
    print(f"📈 Test evaluation plot saved to: {savepath}")
    
if __name__ == "__main__":
    train()