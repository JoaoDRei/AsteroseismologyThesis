from logging import config
import os
import yaml
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from utils import AstroBaselineDataset, SyntheticPSDPeakDataset
from models import SimpleMLP, SimpleCNN, SimpleTransformer
import datetime
import shutil
import seaborn as sns

def train(config_path="./Baseline/config.yaml"):
    # 1. Load Configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # If the main config only signals synthetic mode, switch to the synthetic config.
    synthetic = config.get("debug", {}).get("synthetic", False)
    if synthetic:
        synthetic_config_path = config.get("debug", {}).get("synthetic_config", "./config_synthetic.yaml")
        # Resolve relative paths against the main config file.
        base_dir = os.path.dirname(os.path.abspath(config_path))
        synthetic_config_path = os.path.join(base_dir, synthetic_config_path) if not os.path.isabs(synthetic_config_path) else synthetic_config_path
        if os.path.abspath(synthetic_config_path) != os.path.abspath(config_path):
            with open(synthetic_config_path, "r") as f:
                config = yaml.safe_load(f)
            config.setdefault("debug", {})["synthetic"] = True

    #load loss factory
    def get_loss_fn(name):
        if name == "mse":
            return nn.MSELoss()
        elif name == "l1":
            return nn.L1Loss()
        elif name == "huber":
            return nn.SmoothL1Loss()
        elif name == "bce":
            return nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unknown loss: {name}")

    if synthetic:
        targets_list = ["peak_loc"]
    else:
        targets_list = config["model"]["targets"]
    
    loss_fns = {
        t: get_loss_fn(config['model']['loss_functions'][t])
        for t in targets_list
    }
    loss_weights = config['model']['loss_weights']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Starting training on device: {device}")
    os.makedirs(config['paths']['checkpoint_dir'], exist_ok=True)

    
    # 2. Data Loading & Splitting
    if not synthetic:
        print(f"📂 Loading manifest from {config['data']['manifest_path']}...")
        df = pd.read_csv(config['data']['manifest_path'])
        if "class" in config["model"]["targets"]:

            def normalize_ev(v):
                if pd.isna(v):
                    return None
                v = str(v).strip().upper()
                return v if v in ["RGB", "RC"] else None

            df["evolstate_norm"] = df["evolstate"].apply(normalize_ev)

            before = len(df)
            df = df[df["evolstate_norm"].notnull()].reset_index(drop=True)
            after = len(df)

            print(f"⚠️ Dropped {before - after} invalid evolstate rows")
            
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

        def check_class_balance(df, name):
            print(f"\n{name} distribution:")
            print(df["evolstate_norm"].value_counts(normalize=True))

        train_df = df[df['kic'].isin(train_kics)]
        val_df   = df[df['kic'].isin(val_kics)]
        test_df  = df[df['kic'].isin(test_kics)]

        check_class_balance(train_df, "Train")
        check_class_balance(val_df, "Validation")
        check_class_balance(test_df, "Test")
    
    if synthetic:
        full_ds = SyntheticPSDPeakDataset(n_samples=2000, length=config['data']['target_length'])
        indices = np.arange(len(full_ds))
        np.random.seed(42)
        np.random.shuffle(indices)

        n_train = int(0.8 * len(indices))
        n_val = int(0.1 * len(indices))

        train_idx = indices[:n_train].tolist()
        val_idx = indices[n_train:n_train + n_val].tolist()
        test_idx = indices[n_train + n_val:].tolist()
    else:
        full_ds = AstroBaselineDataset(df, mode=config['data']['mode'],
                                    target_length=config['data']['target_length'])

        train_idx = df[df['kic'].isin(train_kics)].index.tolist()
        val_idx = df[df['kic'].isin(val_kics)].index.tolist()
        test_idx = df[df['kic'].isin(test_kics)].index.tolist()

    print(f"📊 Dataset Split: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    num_workers = config['data'].get('num_workers', 0)
    train_loader = DataLoader(Subset(full_ds, train_idx), batch_size=config['data']['batch_size'], shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(Subset(full_ds, val_idx), batch_size=config['data']['batch_size'], shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(Subset(full_ds, test_idx), batch_size=config['data']['batch_size'], shuffle=False, num_workers=num_workers)

    # 3. Model Initialization
    print(f"🏗️  Initializing {config['model']['type'].upper()} model...")
    if config['model']['type'] == 'cnn':
        model = SimpleCNN(output_dim=len(targets_list))
    elif config['model']['type'] == 'mlp':
        model = SimpleMLP(input_size=full_ds.target_length, output_dim=len(targets_list))
    elif config['model']['type'] == 'transformer':
        model = SimpleTransformer(seq_length=config['data']['seq_length'], output_dim=len(targets_list))
    else:
        raise ValueError("Unknown model type")
    
    
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['model']['learning_rate'])
    #adaptive learning rate:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    # 4. Training Loop

    synthetic = config.get("debug", {}).get("synthetic", False)

    if synthetic:
        targets_list = ["peak_loc"]
    else:
        targets_list = config["model"]["targets"]
    idx_nu = targets_list.index("nu_max") if "nu_max" in targets_list else None
    idx_dnu = targets_list.index("delta_nu") if "delta_nu" in targets_list else None    


    best_val_loss = float('inf')
    print("\n--- Training Started ---")

    # safe lookup for weights in case some targets are not present
    w_nu  = config['model']['loss_weights'].get('nu_max', 0.0)
    w_dnu = config['model']['loss_weights'].get('delta_nu', 0.0)
    
    for epoch in range(config['model']['epochs']):
        model.train()
        total_train_loss = 0
        
        # Batch Printing
        for i, (x, y, _) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            preds = model(x) #no squeeze here since we have 2 outputs now
            loss = 0
            

            for idx, t in enumerate(targets_list):
                pred = preds[:, idx]
                truth = y[:, idx]

                loss_t = loss_fns[t](pred, truth)
                loss += loss_weights[t] * loss_t
           

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
                preds = model(x)
                loss = 0
                for idx, t in enumerate(targets_list):
                    pred = preds[:, idx]
                    truth = y[:, idx]

                    loss_t = loss_fns[t](pred, truth)
                    loss += loss_weights[t] * loss_t
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
    evaluate_model(model, test_loader, device, config, config['data']['manifest_path'], test_idx)

def evaluate_model(model, test_loader, device, config, manifest_path, test_idx):
    


    # Load Configuration
    #with #pen(config_path, "r") as f:
        ##config = yaml.safe_load(f)


        #load loss factory
    def get_loss_fn(name):
        if name == "mse":
            return nn.MSELoss()
        elif name == "l1":
            return nn.L1Loss()
        elif name == "huber":
            return nn.SmoothL1Loss()
        elif name == "bce":
            return nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unknown loss: {name}")

    synthetic = config.get("debug", {}).get("synthetic", False)

    if synthetic:
        targets_list = ["peak_loc"]
    else:
        targets_list = config["model"]["targets"]
    
    
    loss_fns = {
        t: get_loss_fn(config['model']['loss_functions'][t])
        for t in targets_list
    }
    loss_weights = config['model']['loss_weights'] 


    all_preds = {t: [] for t in targets_list}
    all_truth = {t: [] for t in targets_list}
    all_kics = []

    total_test_loss = 0




    
    idx_nu = targets_list.index("nu_max") if "nu_max" in targets_list else None
    idx_dnu = targets_list.index("delta_nu") if "delta_nu" in targets_list else None    


    if synthetic:
        import pandas as _pd
        df = _pd.DataFrame({'kic': test_idx, 'nu_max': [np.nan] * len(test_idx)})
    else:
        df = pd.read_csv(manifest_path)
        if config['data'].get('filter_duration'):
            df = df[df['duration'] == config['data']['filter_duration']].reset_index(drop=True)
    model.eval()
    print("Rows:", len(df))
    print("Unique KICs:", df['kic'].nunique())

    # safe lookup for weights in case some targets are not present
    w_nu  = config['model']['loss_weights'].get('nu_max', 0.0)
    w_dnu = config['model']['loss_weights'].get('delta_nu', 0.0)

    # Create a unique name based on the current time and model type
    if synthetic:
        exp_name = f"synth_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        exp_name = f"{config['model']['type']}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    exp_dir = f"./Baseline/results/{exp_name}"
    os.makedirs(exp_dir, exist_ok=True)
    
    # Copy the config file into that folder so you have a permanent record
    with open(f"{exp_dir}/config.yaml", "w") as f:
        yaml.dump(config, f)
    
    # copy the model architecture file into the folder as well for reproducibility
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_source_file = os.path.join(base_dir, "models", f"{config['model']['type']}.py")
    shutil.copy(model_source_file, f"{exp_dir}/model_architecture.py")
    #copy the best model checkpoint into the folder as well
    shutil.copy(f"{config['paths']['checkpoint_dir']}/best_model.pth", f"{exp_dir}/best_model.pth")
    
    # copy the synthetic config file if in synthetic mode
    if synthetic:
        synthetic_config_path = config.get("debug", {}).get("synthetic_config", "./config_synthetic.yaml")
        # Resolve relative paths against the baseline directory
        synthetic_config_path = os.path.join(base_dir, synthetic_config_path) if not os.path.isabs(synthetic_config_path) else synthetic_config_path
        if os.path.exists(synthetic_config_path):
            shutil.copy(synthetic_config_path, f"{exp_dir}/config_synthetic.yaml")


    with torch.no_grad():
        for x, y, kic in test_loader:
            x, y = x.to(device), y.to(device)


            preds = model(x)
            preds_np=preds.cpu().numpy()
            
            loss = 0
            for idx, t in enumerate(targets_list):
                pred = preds[:, idx]
                truth = y[:, idx]

                loss_t = loss_fns[t](pred, truth)
                loss += loss_weights[t] * loss_t                       
            total_test_loss += loss.item()

            for idx, t in enumerate(targets_list):
                all_preds[t].extend(preds_np[:, idx])
                all_truth[t].extend(y[:, idx].cpu().numpy())
            all_kics.extend(kic.numpy())
    avg_test_loss = total_test_loss / len(test_loader)
    
    cm = None
    if "class" in targets_list:
        from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

        logits = np.array(all_preds["class"])
        truth_class = np.array(all_truth["class"])

        # Convert logits → probabilities → class labels
        probs = 1 / (1 + np.exp(-logits))
        preds_class = (probs > 0.5).astype(int)

        # --- Metrics ---
        acc = accuracy_score(truth_class, preds_class)
        cm = confusion_matrix(truth_class, preds_class)
        report = classification_report(truth_class, preds_class, target_names=["RGB", "RC"])

        print("\n--- Classification Metrics ---")
        print(f"Accuracy: {acc:.4f}")
        print("\nConfusion Matrix:")
        print(cm)
        print("\nClassification Report:")
        print(report)

    #Build database dynamically for all targets
    results_dict = {"kic": all_kics}

    for t in targets_list:
        if t == "class":
            results_dict["truth_class"] = all_truth[t]
            results_dict["logits_class"] = all_preds[t]
        else:
            results_dict[f"truth_log_{t}"] = all_truth[t]
            results_dict[f"pred_log_{t}"] = all_preds[t]

    results_df = pd.DataFrame(results_dict)
    if "class" in targets_list:
        probs = 1 / (1 + np.exp(-np.array(all_preds["class"])))
        results_df["prob_class"] = probs
        results_df["pred_class"] = (probs > 0.5).astype(int)
    # FIX: Deduplicate df by kic before merge - each KIC appears 3× in manifest (20d, 55d, 80d)
    # The merge was creating 3 rows per prediction, inflating error metrics
    # Select only columns that actually exist in the manifest to avoid KeyErrors
    columns_needed = ['kic']
    if 'teff' in df.columns:
        columns_needed.append('teff')
    for c in ['nu_max_hon', 'nu_max', 'nu_max_syd', 'nu_max_a2z', 'nu_max_dia', 'delta_nu', 'delta_nu_syd', 'delta_nu_a2z', 'delta_nu_dia']:
        if c in df.columns:
            columns_needed.append(c)

    df_dedup = df[columns_needed].drop_duplicates(subset=['kic'], keep='first')
    merged = results_df.merge(df_dedup, on='kic', how='left')

    # Compute log-columns only when the source columns exist
    if 'nu_max_hon' in merged.columns:
        merged['hon_log'] = np.log10(merged['nu_max_hon'] + 1e-8)
    if 'nu_max_syd' in merged.columns:
        merged['syd_log'] = np.log10(merged['nu_max_syd'] + 1e-8)
    if 'nu_max_a2z' in merged.columns:
        merged['a2z_log'] = np.log10(merged['nu_max_a2z'] + 1e-8)
    if 'nu_max_dia' in merged.columns:
        merged['dia_log'] = np.log10(merged['nu_max_dia'] + 1e-8)
    if 'nu_max' in merged.columns:
        merged['truth_log_check'] = np.log10(merged['nu_max'] + 1e-8)
    if 'delta_nu_syd' in merged.columns:
        merged['delta_nu_syd_log'] = np.log10(merged['delta_nu_syd'] + 1e-3)
    if 'delta_nu_a2z' in merged.columns:
        merged['delta_nu_a2z_log'] = np.log10(merged['delta_nu_a2z'] + 1e-3)
    if 'delta_nu_dia' in merged.columns:
        merged['delta_nu_dia_log'] = np.log10(merged['delta_nu_dia'] + 1e-3)

    # Residuals for model targets (only for targets present in targets_list)
    for t in targets_list:
        if t in ["nu_max", "delta_nu"]:
            pred_col = f"pred_log_{t}"
            truth_col = f"truth_log_{t}"
            if pred_col in merged.columns and truth_col in merged.columns:
                merged[f"residual_{t}"] = merged[pred_col] - merged[truth_col]

    # HON / SYD diagnostics only if their columns exist
    if 'hon_log' in merged.columns and 'truth_log_nu_max' in merged.columns:
        merged['hon_residual'] = merged['hon_log'] - merged['truth_log_nu_max']
        validHon = merged['nu_max_hon'] > 0 if 'nu_max_hon' in merged.columns else pd.Series(False, index=merged.index)
        hon_mse = np.mean((merged.loc[validHon,'hon_log'] - merged.loc[validHon,'truth_log_nu_max'])**2) if validHon.any() else np.nan
    else:
        validHon = pd.Series(False, index=merged.index)
        hon_mse = np.nan

    validSyd = (merged['nu_max_syd'] > 0) if 'nu_max_syd' in merged.columns else pd.Series(False, index=merged.index)
    valid_dnu_syd = (merged['delta_nu_syd'] > 0) if 'delta_nu_syd' in merged.columns else pd.Series(False, index=merged.index)

    validA2Z = (merged['nu_max_a2z'] > 0) if 'nu_max_a2z' in merged.columns else pd.Series(False, index=merged.index)
    valid_dnu_a2z = (merged['delta_nu_a2z'] > 0) if 'delta_nu_a2z' in merged.columns else pd.Series(False, index=merged.index)

    validDia = (merged['nu_max_dia'] > 0) if 'nu_max_dia' in merged.columns else pd.Series(False, index=merged.index)
    valid_dnu_dia = (merged['delta_nu_dia'] > 0) if 'delta_nu_dia' in merged.columns else pd.Series(False, index=merged.index)

    if "nu_max" in targets_list and "pred_log_nu_max" in merged.columns:
        check_cols = [c for c in ['pred_log_nu_max', 'truth_log_nu_max', 'syd_log'] if c in merged.columns]
        print(merged[check_cols].head(20))

    
    if "nu_max" in targets_list:
    
        # =========================
        # ERROR CALCULATIONS (MSE & RMSE)
        # =========================
        
        # --- Model Errors ---
        model_nu_mse = np.mean((merged['pred_log_nu_max'] - merged['truth_log_nu_max'])**2)
        model_nu_mse_syd_subset = np.mean(
        (merged.loc[validSyd,'pred_log_nu_max'] - merged.loc[validSyd,'truth_log_nu_max'])**2
        ) if validSyd.any() else np.nan
        model_nu_mse_a2z_subset = np.mean(
        (merged.loc[validA2Z,'pred_log_nu_max'] - merged.loc[validA2Z,'truth_log_nu_max'])**2
        ) if validA2Z.any() else np.nan
        model_nu_mse_dia_subset = np.mean(
        (merged.loc[validDia,'pred_log_nu_max'] - merged.loc[validDia,'truth_log_nu_max'])**2
        ) if validDia.any() else np.nan
        model_dnu_mse = np.mean((merged['pred_log_delta_nu'] - merged['truth_log_delta_nu'])**2) if "delta_nu" in targets_list else np.nan
        model_dnu_mse_syd_subset = np.mean(
        (merged.loc[valid_dnu_syd,'pred_log_delta_nu'] - merged.loc[valid_dnu_syd,'truth_log_delta_nu'])**2
        ) if ("delta_nu" in targets_list and valid_dnu_syd.any()) else np.nan
        model_dnu_mse_a2z_subset = np.mean(
        (merged.loc[valid_dnu_a2z,'pred_log_delta_nu'] - merged.loc[valid_dnu_a2z,'truth_log_delta_nu'])**2
        ) if ("delta_nu" in targets_list and valid_dnu_a2z.any()) else np.nan
        model_dnu_mse_dia_subset = np.mean(
        (merged.loc[valid_dnu_dia,'pred_log_delta_nu'] - merged.loc[valid_dnu_dia,'truth_log_delta_nu'])**2
        ) if ("delta_nu" in targets_list and valid_dnu_dia.any()) else np.nan
        model_nu_rmse = np.sqrt(model_nu_mse)
        model_dnu_rmse = np.sqrt(model_dnu_mse) if not np.isnan(model_dnu_mse) else np.nan
        model_combined_error = w_nu * model_nu_mse + (w_dnu * model_dnu_mse if not np.isnan(model_dnu_mse) else 0)

        # --- HON Errors (nu_max only) ---
        hon_mse = np.mean((merged.loc[validHon,'hon_log'] - merged.loc[validHon,'truth_log_nu_max'])**2) if validHon.any() else np.nan
        hon_rmse = np.sqrt(hon_mse) if not np.isnan(hon_mse) else np.nan

        # --- SYD Errors ---
        syd_nu_mse = np.mean((merged.loc[validSyd,'syd_log'] - merged.loc[validSyd,'truth_log_nu_max'])**2) if validSyd.any() else np.nan
        syd_nu_rmse = np.sqrt(syd_nu_mse) if not np.isnan(syd_nu_mse) else np.nan

        syd_dnu_mse = np.mean((merged.loc[valid_dnu_syd,'delta_nu_syd_log'] - merged.loc[valid_dnu_syd,'truth_log_delta_nu'])**2) if ("delta_nu" in targets_list and valid_dnu_syd.any()) else np.nan
        syd_dnu_rmse = np.sqrt(syd_dnu_mse) if not np.isnan(syd_dnu_mse) else np.nan

        # A2Z errors
        a2z_nu_mse = np.mean((merged.loc[validA2Z,'a2z_log'] - merged.loc[validA2Z,'truth_log_nu_max'])**2) if validA2Z.any() else np.nan
        a2z_nu_rmse = np.sqrt(a2z_nu_mse) if not np.isnan(a2z_nu_mse) else np.nan
        a2z_dnu_mse = np.mean((merged.loc[valid_dnu_a2z,'delta_nu_a2z_log'] - merged.loc[valid_dnu_a2z,'truth_log_delta_nu'])**2) if ("delta_nu" in targets_list and valid_dnu_a2z.any()) else np.nan
        a2z_dnu_rmse = np.sqrt(a2z_dnu_mse) if not np.isnan(a2z_dnu_mse) else np.nan

        # DIA errors
        dia_nu_mse = np.mean((merged.loc[validDia,'dia_log'] - merged.loc[validDia,'truth_log_nu_max'])**2) if validDia.any() else np.nan
        dia_nu_rmse = np.sqrt(dia_nu_mse) if not np.isnan(dia_nu_mse) else np.nan
        dia_dnu_mse = np.mean((merged.loc[valid_dnu_dia,'delta_nu_dia_log'] - merged.loc[valid_dnu_dia,'truth_log_delta_nu'])**2) if ("delta_nu" in targets_list and valid_dnu_dia.any()) else np.nan
        dia_dnu_rmse = np.sqrt(dia_dnu_mse) if not np.isnan(dia_dnu_mse) else np.nan

        # Combined error using same weights as model for fair comparison
        syd_combined_error = w_nu * syd_nu_mse + (w_dnu * syd_dnu_mse if not np.isnan(syd_dnu_mse) else 0)
        a2z_combined_error = w_nu * a2z_nu_mse + (w_dnu * a2z_dnu_mse if not np.isnan(a2z_dnu_mse) else 0)
        dia_combined_error = w_nu * dia_nu_mse + (w_dnu * dia_dnu_mse if not np.isnan(dia_dnu_mse) else 0)
        
        # Calculate residuals for SYD (only if columns exist)
        if 'syd_log' in merged.columns and 'truth_log_nu_max' in merged.columns:
            merged['syd_residual'] = merged['syd_log'] - merged['truth_log_nu_max']
        if 'delta_nu_syd_log' in merged.columns and 'truth_log_delta_nu' in merged.columns:
            merged['dnu_syd_residual'] = merged['delta_nu_syd_log'] - merged['truth_log_delta_nu']
        if 'a2z_log' in merged.columns and 'truth_log_nu_max' in merged.columns:
            merged['a2z_residual'] = merged['a2z_log'] - merged['truth_log_nu_max']
        if 'delta_nu_a2z_log' in merged.columns and 'truth_log_delta_nu' in merged.columns:
            merged['dnu_a2z_residual'] = merged['delta_nu_a2z_log'] - merged['truth_log_delta_nu']
        if 'dia_log' in merged.columns and 'truth_log_nu_max' in merged.columns:
            merged['dia_residual'] = merged['dia_log'] - merged['truth_log_nu_max']
        if 'delta_nu_dia_log' in merged.columns and 'truth_log_delta_nu' in merged.columns:
            merged['dnu_dia_residual'] = merged['delta_nu_dia_log'] - merged['truth_log_delta_nu']
        
        print("\n--- Residual Diagnostics ---")
        if 'residual_nu_max' in merged.columns:
            print(f"Mean nu residual: {merged['residual_nu_max'].mean():.4f}")
            print(f"Std nu residual: {merged['residual_nu_max'].std():.4f}")
            print(f"Median nu residual: {merged['residual_nu_max'].median():.4f}")

        if 'residual_delta_nu' in merged.columns:
            print(f"Mean dnu residual: {merged['residual_delta_nu'].mean():.4f}")
            print(f"Std dnu residual: {merged['residual_delta_nu'].std():.4f}")
            print(f"Median dnu residual: {merged['residual_delta_nu'].median():.4f}")
        
        print("\n" + "="*70)
        print("ERROR METRICS SUMMARY (Log Space)")
        print("="*70)
        
        print("\nDATA AVAILABILITY:")
        print(f"  Total stars: {len(merged)}")
        print(f"  HON data: {validHon.sum()} ({100*validHon.sum()/len(merged):.1f}%)")
        print(f"  SYD data: {validSyd.sum()} ({100*validSyd.sum()/len(merged):.1f}%)")
        print(f"  A2Z data: {validA2Z.sum()} ({100*validA2Z.sum()/len(merged):.1f}%)")
        print(f"  DIA data: {validDia.sum()} ({100*validDia.sum()/len(merged):.1f}%)")
        
        print("\nNu_max Errors (MSE | RMSE):")
        print(f"  Model:    {model_nu_mse:.4f} | {model_nu_rmse:.4f}")
        print(f"  HON:      {hon_mse:.4f} | {hon_rmse:.4f}")
        print(f"  SYD:      {syd_nu_mse:.4f} | {syd_nu_rmse:.4f}")
        print(f"  SYD (nu subset): {model_nu_mse_syd_subset:.4f}")
        print(f"  A2Z:      {a2z_nu_mse:.4f} | {a2z_nu_rmse:.4f}")
        print(f"  A2Z (nu subset): {model_nu_mse_a2z_subset:.4f}")
        print(f"  DIA:      {dia_nu_mse:.4f} | {dia_nu_rmse:.4f}")
        print(f"  DIA (nu subset): {model_nu_mse_dia_subset:.4f}")
        
        if "delta_nu" in targets_list:
            print("\nDelta_nu Errors (MSE | RMSE):")
            print(f"  Model:    {model_dnu_mse:.4f} | {model_dnu_rmse:.4f}")
            print(f"  SYD:      {syd_dnu_mse:.4f} | {syd_dnu_rmse:.4f}")
            print(f"  SYD (dnu subset): {model_dnu_mse_syd_subset:.4f}")
            print(f"  A2Z:      {a2z_dnu_mse:.4f} | {a2z_dnu_rmse:.4f}")
            print(f"  A2Z (dnu subset): {model_dnu_mse_a2z_subset:.4f}")
            print(f"  DIA:      {dia_dnu_mse:.4f} | {dia_dnu_rmse:.4f}")
            print(f"  DIA (dnu subset): {model_dnu_mse_dia_subset:.4f}")

            print(f"\nCombined Error (w_nu*MSE_nu + w_dnu*MSE_dnu):")
            print(f"  Model:    {model_combined_error:.4f}")
            print(f"  SYD:      {syd_combined_error:.4f}")
            print(f"  A2Z:      {a2z_combined_error:.4f}")
            print(f"  DIA:      {dia_combined_error:.4f}")
            print(f"  (Weights: w_nu={w_nu}, w_dnu={w_dnu})")
        print("="*70)
        # Save the loss metric to a file
    with open(f"{exp_dir}/metrics.txt", "w") as f:
        if "nu_max" in targets_list and "delta_nu" in targets_list:
            f.write("=" * 70 + "\n")
            f.write("COMPREHENSIVE MODEL & BASELINE COMPARISON\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("DATA AVAILABILITY:\n")
            f.write(f"  Total test stars: {len(merged)}\n")
            f.write(f"  HON data: {validHon.sum()} ({100*validHon.sum()/len(merged):.1f}%)\n")
            f.write(f"  SYD nu_max data: {validSyd.sum()} ({100*validSyd.sum()/len(merged):.1f}%)\n")
            f.write(f"  SYD delta_nu data: {valid_dnu_syd.sum()} ({100*valid_dnu_syd.sum()/len(merged):.1f}%)\n")
            f.write(f"  A2Z nu_max data: {validA2Z.sum()} ({100*validA2Z.sum()/len(merged):.1f}%)\n")
            f.write(f"  A2Z delta_nu data: {valid_dnu_a2z.sum()} ({100*valid_dnu_a2z.sum()/len(merged):.1f}%)\n")
            f.write(f"  DIA nu_max data: {validDia.sum()} ({100*validDia.sum()/len(merged):.1f}%)\n")
            f.write(f"  DIA delta_nu data: {valid_dnu_dia.sum()} ({100*valid_dnu_dia.sum()/len(merged):.1f}%)\n\n")
            
            f.write("ERROR METRICS (Log Space - MSE & RMSE):\n")
            f.write("-" * 70 + "\n")
            
            f.write(f"{'Method':<20} {'nu_max MSE':<15} {'nu_max RMSE':<15} {'Combined':<15}\n")
            f.write("-" * 70 + "\n")
            f.write(f"{'Model':<20} {model_nu_mse:<15.6f} {model_nu_rmse:<15.6f} {model_combined_error:<15.6f}\n")
            f.write(f"{'HON 2019':<20} {hon_mse:<15.6f} {hon_rmse:<15.6f} {'N/A (nu only)':<15}\n")
            f.write(f"{'SYD':<20} {syd_nu_mse:<15.6f} {syd_nu_rmse:<15.6f} {syd_combined_error:<15.6f}\n")
            f.write(f"{'A2Z':<20} {a2z_nu_mse:<15.6f} {a2z_nu_rmse:<15.6f} {a2z_combined_error:<15.6f}\n")
            f.write(f"{'DIA':<20} {dia_nu_mse:<15.6f} {dia_nu_rmse:<15.6f} {dia_combined_error:<15.6f}\n\n")
            
            f.write("DELTA_NU ERRORS (for methods that have it):\n")
            f.write("-" * 70 + "\n")
            f.write(f"{'Method':<20} {'delta_nu MSE':<15} {'delta_nu RMSE':<15}\n")
            f.write("-" * 70 + "\n")
            f.write(f"{'Model':<20} {model_dnu_mse:<15.6f} {model_dnu_rmse:<15.6f}\n")
            f.write(f"{'SYD':<20} {syd_dnu_mse:<15.6f} {syd_dnu_rmse:<15.6f}\n")
            f.write(f"{'A2Z':<20} {a2z_dnu_mse:<15.6f} {a2z_dnu_rmse:<15.6f}\n")
            f.write(f"{'DIA':<20} {dia_dnu_mse:<15.6f} {dia_dnu_rmse:<15.6f}\n\n")
            
            f.write("NOTES:\n")
            f.write(f"  - All errors in log10 space\n")
            f.write(f"  - Model loss weights: w_nu={w_nu}, w_dnu={w_dnu}\n")
            f.write(f"  - Combined error = w_nu*MSE_nu + w_dnu*MSE_dnu\n")
            f.write(f"  - HON only provides nu_max (no delta_nu)\n")
            f.write(f"  - Test Loss (Weighted MSE): {avg_test_loss:.4f}\n")
        if "class" in targets_list:
            f.write("\n" + "="*70 + "\n")
            f.write("CLASSIFICATION METRICS\n")
            f.write("="*70 + "\n\n")

            f.write(f"Accuracy: {acc:.4f}\n\n")

            f.write("Confusion Matrix:\n")
            f.write(f"{cm}\n\n")

            f.write("Classification Report:\n")
            f.write(report + "\n")
    if "nu_max" in targets_list:
        savepath1 = f"{exp_dir}/performance_plot.png"
        min_val=merged['truth_log_nu_max'].min()
        max_val=merged['truth_log_nu_max'].max()
        plt.figure(figsize=(10, 8))
        plt.scatter(merged['truth_log_nu_max'], merged['pred_log_nu_max'],alpha=0.5, s=10, label="Model", color='blue')
        plt.scatter(merged.loc[validHon, 'truth_log_nu_max'],merged.loc[validHon, 'hon_log'],alpha=0.3, s=5, label="Hon 2019", color='orange')
        plt.scatter(merged.loc[validSyd, 'truth_log_nu_max'],merged.loc[validSyd, 'syd_log'],alpha=0.3, s=5, label="SYD", color='green')
        plt.scatter(merged.loc[validA2Z, 'truth_log_nu_max'],merged.loc[validA2Z, 'a2z_log'],alpha=0.3, s=5, label="A2Z", color='magenta')
        plt.scatter(merged.loc[validDia, 'truth_log_nu_max'],merged.loc[validDia, 'dia_log'],alpha=0.3, s=5, label="DIA", color='cyan')
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
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
        plt.scatter(merged['truth_log_nu_max'], merged['residual_nu_max'], alpha=0.5, s=10, color='purple')
        plt.xlabel("True Log10(nu_max)")
        plt.ylabel("Residual (Pred-Truth)")
        plt.title("Residuals vs True nu_max")
        plt.grid(True, alpha=0.3)
        plt.savefig(savepath2, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📈 Residual plot saved to: {savepath2}")

        savepath3= f"{exp_dir}/residual_hist.png"
        plt.figure(figsize=(8, 6))
        plt.hist(merged['residual_nu_max'], bins=50, alpha=0.7, color='purple')
        plt.xlabel("Residual")
        plt.ylabel("Count")
        plt.title("Residual Distribution")
        plt.grid(True, alpha=0.3, axis='y')
        plt.savefig(savepath3, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📈 Residual histogram saved to: {savepath3}")

        savepath4= f"{exp_dir}/residuals_comparison.png"
        plt.figure(figsize=(10,7))
        plt.scatter(merged['truth_log_nu_max'], merged['residual_nu_max'], alpha=0.4, s=10, label="Model")
        plt.scatter(merged['truth_log_nu_max'], merged['hon_residual'], alpha=0.4, s=10, label="Hon")
        plt.scatter(merged.loc[validSyd, 'truth_log_nu_max'], merged.loc[validSyd, 'syd_residual'], alpha=0.4, s=10, label="SYD")
        plt.scatter(merged.loc[validA2Z, 'truth_log_nu_max'], merged.loc[validA2Z, 'a2z_residual'], alpha=0.4, s=10, label="A2Z")
        plt.scatter(merged.loc[validDia, 'truth_log_nu_max'], merged.loc[validDia, 'dia_residual'], alpha=0.4, s=10, label="DIA")
        plt.axhline(0, color='black', linestyle='--', linewidth=1.5)
        plt.legend()
        plt.xlabel("True Log10(nu_max)")
        plt.ylabel("Residual")
        plt.title("Residual Comparison (All Methods)")
        plt.grid(True, alpha=0.3)
        plt.savefig(savepath4, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📈 Residual comparison plot saved to: {savepath4}")

        bins = np.linspace(merged['truth_log_nu_max'].min(), merged['truth_log_nu_max'].max(), 10)
        merged['bin'] = pd.cut(merged['truth_log_nu_max'], bins)

        bin_stats = merged.groupby('bin',observed=True)['residual_nu_max'].agg(['mean', 'std', 'count'])

        print("\n--- Residuals by nu_max bin ---")
        print(bin_stats)


    if "delta_nu" in targets_list:
        savepath5 = f"{exp_dir}/performance_plot_dnu.png"
        min_val_dnu=merged['truth_log_delta_nu'].min()
        max_val_dnu=merged['truth_log_delta_nu'].max()
        plt.figure(figsize=(10, 8))
        plt.scatter(merged['truth_log_delta_nu'], merged['pred_log_delta_nu'],alpha=0.5, s=10, label="Model", color='blue')
        plt.scatter(merged.loc[valid_dnu_syd, 'truth_log_delta_nu'], merged.loc[valid_dnu_syd, 'delta_nu_syd_log'], 
                    alpha=0.3, s=5, label="SYD", color='green')
        plt.scatter(merged.loc[valid_dnu_a2z, 'truth_log_delta_nu'], merged.loc[valid_dnu_a2z, 'delta_nu_a2z_log'], 
                    alpha=0.3, s=5, label="A2Z", color='magenta')
        plt.scatter(merged.loc[valid_dnu_dia, 'truth_log_delta_nu'], merged.loc[valid_dnu_dia, 'delta_nu_dia_log'], 
                    alpha=0.3, s=5, label="DIA", color='cyan')
        plt.plot([min_val_dnu, max_val_dnu], [min_val_dnu, max_val_dnu], 'r--', linewidth=2, label='Perfect prediction')
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
        plt.scatter(merged['truth_log_delta_nu'], merged['residual_delta_nu'], alpha=0.5, s=10, color='purple', label='Model')
        plt.scatter(merged['truth_log_delta_nu'], merged['dnu_syd_residual'], alpha=0.5, s=10, color='green', label='SYD')
        plt.scatter(merged.loc[valid_dnu_a2z, 'truth_log_delta_nu'], merged.loc[valid_dnu_a2z, 'dnu_a2z_residual'], alpha=0.5, s=10, color='magenta', label='A2Z')
        plt.scatter(merged.loc[valid_dnu_dia, 'truth_log_delta_nu'], merged.loc[valid_dnu_dia, 'dnu_dia_residual'], alpha=0.5, s=10, color='cyan', label='DIA')
        plt.xlabel("True Log10(Δν)")
        plt.ylabel("Residual (Pred-Truth)")
        plt.title("Residuals vs True Δν")
        plt.grid(True, alpha=0.3)
        plt.savefig(savepath6, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📈 Residual plot saved to: {savepath6}")

        savepath7= f"{exp_dir}/residual_hist_dnu.png"
        plt.figure(figsize=(8, 6))
        plt.hist(merged['residual_delta_nu'], bins=50, alpha=0.7, color='purple')
        plt.xlabel("Residual for Δν")
        plt.ylabel("Count")
        plt.title("Residual Distribution for Δν")
        plt.grid(True, alpha=0.3, axis='y')
        plt.savefig(savepath7, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📈 Residual histogram saved to: {savepath7}")

        bins_denu = np.linspace(merged['truth_log_delta_nu'].min(), merged['truth_log_delta_nu'].max(), 10)
        merged['bin_dnu'] = pd.cut(merged['truth_log_delta_nu'], bins_denu)

        bin_statsdnu = merged.groupby('bin_dnu',observed=True)['residual_delta_nu'].agg(['mean', 'std', 'count'])

        print("\n--- Residuals by Δν bin ---")
        print(bin_statsdnu)



    if "nu_max" in targets_list and "delta_nu" in targets_list:

        # =========================
        # ASTEROSEISMIC PARAMETERS (only when model predicted nu_max and delta_nu)
        # =========================
        NU_MAX_SUN = 3090.0
        DNU_SUN = 135.1
        TEFF_SUN = 5777.0
        LOGG_SUN = 4.44

        if ('pred_log_nu_max' in merged.columns) and ('pred_log_delta_nu' in merged.columns):
            # Convert from log10 back to linear scale
            merged["nu_max_pred"] = 10**merged["pred_log_nu_max"]
            merged["dnu_pred"] = 10**merged["pred_log_delta_nu"]

            # ---- HANDLE T_eff ----
            if "teff" in merged.columns:
                Teff = merged["teff"]
            else:
                print("⚠️ No T_eff found; skipping some derived quantities")
                Teff = None

            # ---- Scaling relations ----
            if Teff is not None:
                # Radius
                merged["R_pred"] = (
                    (merged["nu_max_pred"] / NU_MAX_SUN)
                    * (merged["dnu_pred"] / DNU_SUN) ** (-2)
                    * (Teff / TEFF_SUN) ** 0.5
                )

                # Mass
                merged["M_pred"] = (
                    (merged["nu_max_pred"] / NU_MAX_SUN) ** 3
                    * (merged["dnu_pred"] / DNU_SUN) ** (-4)
                    * (Teff / TEFF_SUN) ** 1.5
                )

                # log g
                merged["logg_pred"] = (
                    LOGG_SUN
                    + np.log10(merged["nu_max_pred"] / NU_MAX_SUN)
                    + 0.5 * np.log10(Teff / TEFF_SUN)
                )

            # Prepare output columns depending on what was computed
            output_cols = ["kic"]
            output_cols.append("nu_max_pred")
            output_cols.append("dnu_pred")
            if 'M_pred' in merged.columns:
                output_cols.extend(["M_pred", "R_pred", "logg_pred"])

            output_df = merged[output_cols]
            csv_path = f"{exp_dir}/asteroseismic_parameters.csv"
            output_df.to_csv(csv_path, index=False)
            print(f"⭐ Asteroseismic parameters saved to: {csv_path}")
        else:
            print("⚠️ Skipping asteroseismic parameter computation: model did not predict both nu_max and delta_nu")
    if "class" in targets_list:
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=["RGB", "RC"],
                    yticklabels=["RGB", "RC"])
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.savefig(f"{exp_dir}/confusion_matrix.png", dpi=150, bbox_inches='tight')
        plt.close()

        print(f"📈 Confusion matrix saved to: {exp_dir}/confusion_matrix.png")
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./Baseline/config.yaml", help="Path to config yaml")
    args = parser.parse_args()

    train(config_path=args.config)