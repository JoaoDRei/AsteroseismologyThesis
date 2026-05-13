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

    w_nu  = config['model']['loss_weights']['nu']
    w_dnu = config['model']['loss_weights']['dnu'] 
    
    for epoch in range(config['model']['epochs']):
        model.train()
        total_train_loss = 0
        
        # Batch Printing
        for i, (x, y, _) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            y_nu=torch.log10(y[:,0]+1e-8)
            y_dnu=torch.log10(y[:,1]+1e-3)


            optimizer.zero_grad()
            preds = model(x) #no squeeze here since we have 2 outputs now
            loss_nu = criterion(preds[:,0], y_nu)
            loss_dnu = criterion(preds[:,1], y_dnu)

            loss = w_nu * loss_nu + w_dnu * loss_dnu
            
           

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
                x, y = x.to(device), y.to(device)
                y_nu=torch.log10(y[:,0]+1e-8)
                y_dnu=torch.log10(y[:,1]+1e-3)
                preds = model(x)
                loss_nu = criterion(preds[:,0], y_nu)
                loss_dnu = criterion(preds[:,1], y_dnu)
 
                loss = w_nu * loss_nu + w_dnu * loss_dnu    
                total_val_loss += loss.item()
        
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
    


    all_preds_nu = []
    all_preds_dnu = []
    all_truth_nu = []
    all_truth_dnu = []
    all_kics = []

    total_test_loss = 0
    # Load Configuration
    with open("./Baseline/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    df=pd.read_csv(manifest_path)
    if config['data'].get('filter_duration'):
        df = df[df['duration'] == config['data']['filter_duration']].reset_index(drop=True)
    model.eval()
    print("Rows:", len(df))
    print("Unique KICs:", df['kic'].nunique())

    w_nu  = config['model']['loss_weights']['nu']
    w_dnu = config['model']['loss_weights']['dnu']

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
            x, y = x.to(device), y.to(device)

            y_nu=torch.log10(y[:,0]+1e-8)
            y_dnu=torch.log10(y[:,1]+1e-3)
            preds = model(x)
            preds_np=preds.cpu().numpy()
            y_nu_np=y_nu.cpu().numpy()
            y_dnu_np=y_dnu.cpu().numpy()
            loss_nu = criterion(preds[:,0], y_nu)
            loss_dnu = criterion(preds[:,1], y_dnu)

            loss = w_nu * loss_nu + w_dnu * loss_dnu    
            
            total_test_loss += loss.item()

            all_preds_nu.extend(preds_np[:,0])
            all_preds_dnu.extend(preds_np[:,1])
            all_truth_nu.extend(y_nu_np)
            all_truth_dnu.extend(y_dnu_np)
            all_kics.extend(kic.numpy())
    avg_test_loss = total_test_loss / len(test_loader)

    results_df=pd.DataFrame({
        "kic": all_kics,
        "truth_log_nu": all_truth_nu,
        "truth_log_dnu": all_truth_dnu,
        "pred_log_nu": all_preds_nu,
        "pred_log_dnu": all_preds_dnu,
        })

    # FIX: Deduplicate df by kic before merge - each KIC appears 3× in manifest (20d, 55d, 80d)
    # The merge was creating 3 rows per prediction, inflating error metrics
    df_dedup = df[['kic','nu_max_hon', 'nu_max', 'nu_max_syd', 'delta_nu', 'delta_nu_syd']].drop_duplicates(subset=['kic'], keep='first')
    merged=results_df.merge(df_dedup, on='kic', how='left')

    merged['hon_log']=np.log10(merged['nu_max_hon'] + 1e-8)
    merged['syd_log']=np.log10(merged['nu_max_syd'] + 1e-8)
    merged['truth_log_check']=np.log10(merged['nu_max'] + 1e-8)
    merged['delta_nu_syd_log']=np.log10(merged['delta_nu_syd'] + 1e-3)
    
    merged['residual_nu']=merged['pred_log_nu'] - merged['truth_log_nu']
    merged['residual_dnu']=merged['pred_log_dnu'] - merged['truth_log_dnu']
    
    merged['hon_residual'] = merged['hon_log'] - merged['truth_log_nu']
    validHon=merged['nu_max_hon'] > 0
    hon_mse=np.mean((merged.loc[validHon,'hon_log'] - merged.loc[validHon,'truth_log_nu'])**2)
    validSyd=merged['nu_max_syd'] > 0
    valid_dnu_syd=merged['delta_nu_syd'] > 0

    check = merged[['pred_log_nu', 'truth_log_nu', 'syd_log']].head(20)
    print(check)

    # =========================
    # ERROR CALCULATIONS (MSE & RMSE)
    # =========================
    
    # --- Model Errors ---
    model_nu_mse = np.mean((merged['pred_log_nu'] - merged['truth_log_nu'])**2)
    model_nu_mse_syd_subset = np.mean(
    (merged.loc[validSyd,'pred_log_nu'] - merged.loc[validSyd,'truth_log_nu'])**2
    )
    model_dnu_mse = np.mean((merged['pred_log_dnu'] - merged['truth_log_dnu'])**2)
    model_dnu_mse_syd_subset = np.mean(
    (merged.loc[valid_dnu_syd,'pred_log_dnu'] - merged.loc[valid_dnu_syd,'truth_log_dnu'])**2
    )
    model_nu_rmse = np.sqrt(model_nu_mse)
    model_dnu_rmse = np.sqrt(model_dnu_mse)
    model_combined_error = w_nu * model_nu_mse + w_dnu * model_dnu_mse  # Same as training loss

    # --- HON Errors (nu_max only) ---
    hon_mse = np.mean((merged.loc[validHon,'hon_log'] - merged.loc[validHon,'truth_log_nu'])**2)
    hon_rmse = np.sqrt(hon_mse)

    # --- SYD Errors ---
    syd_nu_mse = np.mean((merged.loc[validSyd,'syd_log'] - merged.loc[validSyd,'truth_log_nu'])**2)
    syd_nu_rmse = np.sqrt(syd_nu_mse)
    
    syd_dnu_mse = np.mean((merged.loc[valid_dnu_syd,'delta_nu_syd_log'] - merged.loc[valid_dnu_syd,'truth_log_dnu'])**2)
    syd_dnu_rmse = np.sqrt(syd_dnu_mse)
    
    # Combined error using same weights as model for fair comparison
    syd_combined_error = w_nu * syd_nu_mse + w_dnu * syd_dnu_mse
    
    # Calculate residuals for SYD
    merged['syd_residual'] = merged['syd_log'] - merged['truth_log_nu']
    merged['dnu_syd_residual'] = merged['delta_nu_syd_log'] - merged['truth_log_dnu']
    
    print("\n--- Residual Diagnostics ---")
    print(f"Mean nu residual: {merged['residual_nu'].mean():.4f}")
    print(f"Std nu residual: {merged['residual_nu'].std():.4f}")
    print(f"Median nu residual: {merged['residual_nu'].median():.4f}")

    print(f"Mean dnu residual: {merged['residual_dnu'].mean():.4f}")
    print(f"Std dnu residual: {merged['residual_dnu'].std():.4f}")
    print(f"Median dnu residual: {merged['residual_dnu'].median():.4f}")
    
    print("\n" + "="*70)
    print("ERROR METRICS SUMMARY (Log Space)")
    print("="*70)
    
    print("\nDATA AVAILABILITY:")
    print(f"  Total stars: {len(merged)}")
    print(f"  HON data: {validHon.sum()} ({100*validHon.sum()/len(merged):.1f}%)")
    print(f"  SYD data: {validSyd.sum()} ({100*validSyd.sum()/len(merged):.1f}%)")
    
    print("\nNu_max Errors (MSE | RMSE):")
    print(f"  Model:    {model_nu_mse:.4f} | {model_nu_rmse:.4f}")
    print(f"  HON:      {hon_mse:.4f} | {hon_rmse:.4f}")
    print(f"  SYD:      {syd_nu_mse:.4f} | {syd_nu_rmse:.4f}")
    print(f"  SYD (nu subset): {model_nu_mse_syd_subset:.4f}")
    
    print("\nDelta_nu Errors (MSE | RMSE):")
    print(f"  Model:    {model_dnu_mse:.4f} | {model_dnu_rmse:.4f}")
    print(f"  SYD:      {syd_dnu_mse:.4f} | {syd_dnu_rmse:.4f}")
    print(f"  SYD (dnu subset): {model_dnu_mse_syd_subset:.4f}")

    print(f"\nCombined Error (w_nu*MSE_nu + w_dnu*MSE_dnu):")
    print(f"  Model:    {model_combined_error:.4f}")
    print(f"  SYD:      {syd_combined_error:.4f}")
    print(f"  (Weights: w_nu={w_nu}, w_dnu={w_dnu})")
    print("="*70)
    # Save the loss metric to a file
    with open(f"{exp_dir}/metrics.txt", "w") as f:
        f.write("=" * 70 + "\n")
        f.write("COMPREHENSIVE MODEL & BASELINE COMPARISON\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("DATA AVAILABILITY:\n")
        f.write(f"  Total test stars: {len(merged)}\n")
        f.write(f"  HON data: {validHon.sum()} ({100*validHon.sum()/len(merged):.1f}%)\n")
        f.write(f"  SYD nu_max data: {validSyd.sum()} ({100*validSyd.sum()/len(merged):.1f}%)\n")
        f.write(f"  SYD delta_nu data: {valid_dnu_syd.sum()} ({100*valid_dnu_syd.sum()/len(merged):.1f}%)\n\n")
        
        f.write("ERROR METRICS (Log Space - MSE & RMSE):\n")
        f.write("-" * 70 + "\n")
        
        f.write(f"{'Method':<20} {'nu_max MSE':<15} {'nu_max RMSE':<15} {'Combined':<15}\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Model':<20} {model_nu_mse:<15.6f} {model_nu_rmse:<15.6f} {model_combined_error:<15.6f}\n")
        f.write(f"{'HON 2019':<20} {hon_mse:<15.6f} {hon_rmse:<15.6f} {'N/A (nu only)':<15}\n")
        f.write(f"{'SYD':<20} {syd_nu_mse:<15.6f} {syd_nu_rmse:<15.6f} {syd_combined_error:<15.6f}\n\n")
        
        f.write("DELTA_NU ERRORS (for methods that have it):\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Method':<20} {'delta_nu MSE':<15} {'delta_nu RMSE':<15}\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Model':<20} {model_dnu_mse:<15.6f} {model_dnu_rmse:<15.6f}\n")
        f.write(f"{'SYD':<20} {syd_dnu_mse:<15.6f} {syd_dnu_rmse:<15.6f}\n\n")
        
        f.write("NOTES:\n")
        f.write(f"  - All errors in log10 space\n")
        f.write(f"  - Model loss weights: w_nu={w_nu}, w_dnu={w_dnu}\n")
        f.write(f"  - Combined error = w_nu*MSE_nu + w_dnu*MSE_dnu\n")
        f.write(f"  - HON only provides nu_max (no delta_nu)\n")
        f.write(f"  - Test Loss (Weighted MSE): {avg_test_loss:.4f}\n")

    savepath1 = f"{exp_dir}/performance_plot.png"
    plt.figure(figsize=(10, 8))
    plt.scatter(merged['truth_log_nu'], merged['pred_log_nu'],alpha=0.5, s=10, label="Model", color='blue')
    plt.scatter(merged.loc[validHon, 'truth_log_nu'],merged.loc[validHon, 'hon_log'],alpha=0.3, s=5, label="Hon 2019", color='orange')
    plt.scatter(merged.loc[validSyd, 'truth_log_nu'],merged.loc[validSyd, 'syd_log'],alpha=0.3, s=5, label="SYD", color='green')
    plt.plot([min(all_truth_nu), max(all_truth_nu)], [min(all_truth_nu), max(all_truth_nu)], 'r--', linewidth=2, label='Perfect prediction')
    plt.xlabel("True Log10(nu_max)")
    plt.ylabel("Predicted Log10(nu_max)")
    plt.title(f"{config['data']['mode'].upper()} {config['model']['type'].upper()} {config['data']['filter_duration']}  Model Performance")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(savepath1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📈 Test evaluation plot saved to: {savepath1}")
    
    savepath2= f"{exp_dir}/residuals_plot.png"
    plt.figure(figsize=(8, 6))
    plt.scatter(merged['truth_log_nu'], merged['residual_nu'], alpha=0.5, s=10, color='purple')
    plt.xlabel("True Log10(nu_max)")
    plt.ylabel("Residual (Pred-Truth)")
    plt.title("Residuals vs True nu_max")
    plt.grid(True, alpha=0.3)
    plt.savefig(savepath2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📈 Residual plot saved to: {savepath2}")

    savepath3= f"{exp_dir}/residual_hist.png"
    plt.figure(figsize=(8, 6))
    plt.hist(merged['residual_nu'], bins=50, alpha=0.7, color='purple')
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.title("Residual Distribution")
    plt.grid(True, alpha=0.3, axis='y')
    plt.savefig(savepath3, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📈 Residual histogram saved to: {savepath3}")

    savepath4= f"{exp_dir}/residuals_comparison.png"
    plt.figure(figsize=(10,7))
    plt.scatter(merged['truth_log_nu'], merged['residual_nu'], alpha=0.4, s=10, label="Model")
    plt.scatter(merged['truth_log_nu'], merged['hon_residual'], alpha=0.4, s=10, label="Hon")
    plt.scatter(merged.loc[validSyd, 'truth_log_nu'], merged.loc[validSyd, 'syd_residual'], alpha=0.4, s=10, label="SYD")
    plt.axhline(0, color='black', linestyle='--', linewidth=1.5)
    plt.legend()
    plt.xlabel("True Log10(nu_max)")
    plt.ylabel("Residual")
    plt.title("Residual Comparison (All Methods)")
    plt.grid(True, alpha=0.3)
    plt.savefig(savepath4, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📈 Residual comparison plot saved to: {savepath4}")

    bins = np.linspace(merged['truth_log_nu'].min(), merged['truth_log_nu'].max(), 10)
    merged['bin'] = pd.cut(merged['truth_log_nu'], bins)

    bin_stats = merged.groupby('bin',observed=True)['residual_nu'].agg(['mean', 'std', 'count'])

    print("\n--- Residuals by nu_max bin ---")
    print(bin_stats)


    savepath5 = f"{exp_dir}/performance_plot_dnu.png"
    plt.figure(figsize=(10, 8))
    plt.scatter(merged['truth_log_dnu'], merged['pred_log_dnu'],alpha=0.5, s=10, label="Model", color='blue')
    plt.scatter(merged.loc[valid_dnu_syd, 'truth_log_dnu'], merged.loc[valid_dnu_syd, 'delta_nu_syd_log'], 
                alpha=0.3, s=5, label="SYD", color='green')
    plt.plot([min(all_truth_dnu), max(all_truth_dnu)], [min(all_truth_dnu), max(all_truth_dnu)], 'r--', linewidth=2, label='Perfect prediction')
    plt.xlabel("True Log10(Δν)")
    plt.ylabel("Predicted Log10(Δν)")
    plt.title(f"{config['data']['mode'].upper()} {config['model']['type'].upper()} {config['data']['filter_duration']}  Model Performance for Δν")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(savepath5, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📈 Test evaluation plot saved to: {savepath5}")
    

    savepath6= f"{exp_dir}/residuals_plot_dnu.png"
    plt.figure(figsize=(8, 6))
    plt.scatter(merged['truth_log_dnu'], merged['residual_dnu'], alpha=0.5, s=10, color='purple', label='Model')
    plt.scatter(merged['truth_log_dnu'], merged['dnu_syd_residual'], alpha=0.5, s=10, color='green')
    plt.xlabel("True Log10(Δν)")
    plt.ylabel("Residual (Pred-Truth)")
    plt.title("Residuals vs True Δν")
    plt.grid(True, alpha=0.3)
    plt.savefig(savepath6, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📈 Residual plot saved to: {savepath6}")

    savepath7= f"{exp_dir}/residual_hist_dnu.png"
    plt.figure(figsize=(8, 6))
    plt.hist(merged['residual_dnu'], bins=50, alpha=0.7, color='purple')
    plt.xlabel("Residual for Δν")
    plt.ylabel("Count")
    plt.title("Residual Distribution for Δν")
    plt.grid(True, alpha=0.3, axis='y')
    plt.savefig(savepath7, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📈 Residual histogram saved to: {savepath7}")

    bins_denu = np.linspace(merged['truth_log_dnu'].min(), merged['truth_log_dnu'].max(), 10)
    merged['bin_dnu'] = pd.cut(merged['truth_log_dnu'], bins_denu)

    bin_statsdnu = merged.groupby('bin_dnu',observed=True)['residual_dnu'].agg(['mean', 'std', 'count'])

    print("\n--- Residuals by Δν bin ---")
    print(bin_statsdnu)

if __name__ == "__main__":
    train()