"""
BiLSTM Binary Classification Inference Script

Load trained BiLSTM model and run binary classification predictions on test data.
Outputs predictions with probabilities to CSV file.

Based on BiLSTM_Classification.ipynb implementation.
Paper: Kononov et al., "Prediction of Technical State of Mechanical Systems 
       Based on Interpretive Neural Network Model", Sensors 2023, 23(4), 1892
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import argparse
import json
import pickle
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Model Architecture (same as notebook)
# ============================================================================

class BiLSTMClassifierFixed(nn.Module):
    """
    BiLSTM model for binary classification - FIXED LENGTH VERSION
    Architecture from paper Table 1
    """
    def __init__(self, input_size, hidden_sizes=[64, 32], fc_sizes=[16, 8], dropout=0.2):
        super(BiLSTMClassifierFixed, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        
        # BiLSTM Layer 1: input_size -> 64 (bidirectional -> 128)
        self.bilstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_sizes[0],
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # BiLSTM Layer 2: 128 -> 32 (bidirectional -> 64)
        self.bilstm2 = nn.LSTM(
            input_size=hidden_sizes[0] * 2,
            hidden_size=hidden_sizes[1],
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # FC Layer 1: 64 -> 16
        self.fc1 = nn.Linear(hidden_sizes[1] * 2, fc_sizes[0])
        self.dropout1 = nn.Dropout(dropout)
        
        # FC Layer 2: 16 -> 8
        self.fc2 = nn.Linear(fc_sizes[0], fc_sizes[1])
        self.dropout2 = nn.Dropout(dropout)
        
        # Output Layer: 8 -> 1
        self.fc_out = nn.Linear(fc_sizes[1], 1)
        
    def forward(self, x):
        """
        Forward pass for FIXED-LENGTH sequences
        
        Args:
            x: Input tensor (batch, seq_len, num_sensors)
        
        Returns:
            output: Binary classification logits (batch, 1)
        """
        # BiLSTM Layer 1
        output, (h1, c1) = self.bilstm1(x)
        
        # BiLSTM Layer 2
        output, (h2, c2) = self.bilstm2(output)
        
        # Get last output (many-to-one)
        last_output = output[:, -1, :]  # (batch, hidden*2)
        
        # FC Layer 1 + ReLU + Dropout
        x = self.fc1(last_output)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # FC Layer 2 + ReLU + Dropout
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # Output Layer
        output = self.fc_out(x)  # (batch, 1)
        
        return output


# ============================================================================
# Data Loading & Preprocessing
# ============================================================================

def load_cmapss_data(filepath):
    """Load C-MAPSS dataset"""
    columns = ['unit', 'cycle', 'setting_1', 'setting_2', 'setting_3'] + \
              [f's{i}' for i in range(1, 22)]
    df = pd.read_csv(filepath, sep='\s+', header=None, names=columns)
    return df


def normalize_sensors(df, stats, sensor_cols):
    """Apply min-max normalization using training statistics"""
    df = df.copy()
    for col in sensor_cols:
        min_val = stats[col]['min']
        max_val = stats[col]['max']
        range_val = max_val - min_val
        if range_val > 1e-6:
            df[col] = (df[col] - min_val) / range_val
        else:
            df[col] = 0.0
    return df


def create_sliding_windows(df, sensor_cols, window_size, step=1):
    """
    Create sliding windows for inference
    
    Args:
        df: DataFrame with sensor data
        sensor_cols: List of sensor column names
        window_size: Size of sliding window (21 from paper)
        step: Step size for sliding window
    
    Returns:
        windows: Array of sequences (num_windows, window_size, num_sensors)
        window_info: DataFrame with window metadata
    """
    windows = []
    window_info = []
    
    for unit_id in sorted(df['unit'].unique()):
        unit_data = df[df['unit'] == unit_id].sort_values('cycle')
        sensor_values = unit_data[sensor_cols].values
        total_cycles = len(unit_data)
        
        if total_cycles < window_size:
            # Pad short sequences
            pad_length = window_size - total_cycles
            pad = np.repeat(sensor_values[:1], pad_length, axis=0)
            window = np.concatenate([pad, sensor_values], axis=0)
            
            windows.append(window)
            window_info.append({
                'unit_id': unit_id,
                'window_idx': 0,
                'start_cycle': 1,
                'end_cycle': total_cycles,
                'total_cycles': total_cycles,
                'is_padded': True
            })
        else:
            # Create sliding windows
            for start_idx in range(0, total_cycles - window_size + 1, step):
                end_idx = start_idx + window_size
                window = sensor_values[start_idx:end_idx]
                
                windows.append(window)
                window_info.append({
                    'unit_id': unit_id,
                    'window_idx': start_idx // step,
                    'start_cycle': start_idx + 1,
                    'end_cycle': end_idx,
                    'total_cycles': total_cycles,
                    'is_padded': False
                })
    
    windows = np.array(windows, dtype=np.float32)
    window_info_df = pd.DataFrame(window_info)
    
    return windows, window_info_df


def create_last_window_per_unit(df, sensor_cols, window_size):
    """
    Create only the last window per unit (most recent state)
    Useful for final RUL prediction per engine
    
    Args:
        df: DataFrame with sensor data
        sensor_cols: List of sensor column names
        window_size: Size of window (21 from paper)
    
    Returns:
        windows: Array of sequences (num_units, window_size, num_sensors)
        unit_info: DataFrame with unit metadata
    """
    windows = []
    unit_info = []
    
    for unit_id in sorted(df['unit'].unique()):
        unit_data = df[df['unit'] == unit_id].sort_values('cycle')
        sensor_values = unit_data[sensor_cols].values
        total_cycles = len(unit_data)
        
        if total_cycles >= window_size:
            window = sensor_values[-window_size:]
        else:
            # Pad with first reading
            pad_length = window_size - total_cycles
            pad = np.repeat(sensor_values[:1], pad_length, axis=0)
            window = np.concatenate([pad, sensor_values], axis=0)
        
        windows.append(window)
        unit_info.append({
            'unit_id': unit_id,
            'last_cycle': total_cycles,
            'is_padded': total_cycles < window_size
        })
    
    windows = np.array(windows, dtype=np.float32)
    unit_info_df = pd.DataFrame(unit_info)
    
    return windows, unit_info_df


# ============================================================================
# Inference
# ============================================================================

def run_inference(checkpoint_path, data_dir, test_file, output_path, 
                  device='cuda', mode='last_window', threshold=None):
    """
    Run inference on test data
    
    Args:
        checkpoint_path: Path to model checkpoint (.pt file)
        data_dir: Directory containing normalization.json
        test_file: Path to test data file
        output_path: Path to save predictions CSV
        device: 'cuda' or 'cpu'
        mode: 'last_window' (one prediction per unit) or 'all_windows' (sliding window)
        threshold: Classification threshold (default: from checkpoint or 0.5)
    """
    
    print("=" * 80)
    print("BiLSTM BINARY CLASSIFICATION INFERENCE")
    print("=" * 80)
    
    # Load checkpoint
    print(f"\nLoading checkpoint from: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        if isinstance(e, pickle.UnpicklingError) or 'Weights only load failed' in str(e):
            print("Weights-only load failed. Retrying with weights_only=False.")
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            except TypeError:
                raise
        else:
            raise
    
    config = checkpoint.get('config', {})
    
    # Model configuration
    WINDOW_SIZE = config.get('sliding_window', 21)
    FORECAST_HORIZON = config.get('forecast_horizon', 30)
    
    # Determine optimal threshold
    if threshold is None:
        # Try to load from results.json if available
        results_path = Path(data_dir) / 'results.json'
        if results_path.exists():
            with open(results_path, 'r') as f:
                results = json.load(f)
                threshold = results.get('threshold', {}).get('optimal_threshold', 0.5)
                print(f"Loaded optimal threshold from results: {threshold:.4f}")
        else:
            threshold = 0.5
            print(f"Using default threshold: {threshold}")
    
    print(f"\nModel configuration:")
    print(f"  Forecast horizon: {FORECAST_HORIZON} cycles")
    print(f"  Window size: {WINDOW_SIZE}")
    print(f"  Classification threshold: {threshold:.4f}")
    print(f"  Inference mode: {mode}")
    
    # Initialize model
    print("\nInitializing model...")
    SENSORS = [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]
    SENSOR_COLS = [f's{i}' for i in SENSORS]
    
    model = BiLSTMClassifierFixed(
        input_size=len(SENSOR_COLS),
        hidden_sizes=[64, 32],
        fc_sizes=[16, 8],
        dropout=0.2
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    print(f"Input features: {len(SENSOR_COLS)} sensors")
    print(f"Loaded from epoch: {checkpoint.get('epoch', 'unknown')}")
    if 'best_f1' in checkpoint:
        print(f"Best validation F1: {checkpoint['best_f1']:.4f}")
    
    # Load normalization stats
    norm_stats_path = Path(data_dir) / 'normalization.json'
    print(f"\nLoading normalization stats from: {norm_stats_path}")
    
    if not norm_stats_path.exists():
        raise FileNotFoundError(
            f"Normalization stats not found: {norm_stats_path}\n"
            f"Please ensure you have the normalization.json file from training."
        )
    
    with open(norm_stats_path, 'r') as f:
        norm_stats = json.load(f)
    
    # Load test data
    print(f"\nLoading test data from: {test_file}")
    test_df = load_cmapss_data(test_file)
    print(f"Test data: {len(test_df):,} records, {test_df['unit'].nunique()} units")
    
    # Normalize
    print("\nNormalizing test data...")
    test_df = normalize_sensors(test_df, norm_stats, SENSOR_COLS)
    
    # Create windows based on mode
    print(f"\nCreating windows (mode: {mode})...")
    
    if mode == 'last_window':
        X_test, info_df = create_last_window_per_unit(test_df, SENSOR_COLS, WINDOW_SIZE)
        print(f"Windows created: {len(X_test)} (one per unit)")
    elif mode == 'all_windows':
        X_test, info_df = create_sliding_windows(test_df, SENSOR_COLS, WINDOW_SIZE, step=1)
        print(f"Windows created: {len(X_test):,} (sliding window)")
    else:
        raise ValueError(f"Invalid mode: {mode}. Choose 'last_window' or 'all_windows'")
    
    print(f"Window shape: {X_test.shape}")
    
    # Run inference
    print("\nRunning inference...")
    predictions = []
    probabilities = []
    
    batch_size = 256
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch = X_test[i:i+batch_size]
            x = torch.from_numpy(batch).to(device)
            
            # Get logits
            logits = model(x).squeeze()
            
            # Convert to probabilities
            probs = torch.sigmoid(logits)
            
            # Apply threshold
            preds = (probs >= threshold).long()
            
            probabilities.extend(probs.cpu().numpy())
            predictions.extend(preds.cpu().numpy())
    
    # Add predictions to info dataframe
    info_df['probability'] = probabilities
    info_df['predicted_class'] = predictions
    info_df['prediction'] = info_df['predicted_class'].map({
        0: 'Safe (>30 cycles)',
        1: f'Failure Risk (≤{FORECAST_HORIZON} cycles)'
    })
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("INFERENCE COMPLETED")
    print("=" * 80)
    print(f"Total predictions: {len(info_df):,}")
    
    if mode == 'last_window':
        print(f"\nPer-Unit Classification Results:")
        print(f"  Units predicted as SAFE (Class 0): {(info_df['predicted_class']==0).sum()}")
        print(f"  Units predicted as FAILURE RISK (Class 1): {(info_df['predicted_class']==1).sum()}")
    else:
        print(f"\nWindow Classification Results:")
        print(f"  Windows predicted as SAFE (Class 0): {(info_df['predicted_class']==0).sum():,}")
        print(f"  Windows predicted as FAILURE RISK (Class 1): {(info_df['predicted_class']==1).sum():,}")
    
    print(f"\nProbability Statistics:")
    print(f"  Mean probability: {info_df['probability'].mean():.4f}")
    print(f"  Std probability:  {info_df['probability'].std():.4f}")
    print(f"  Min probability:  {info_df['probability'].min():.4f}")
    print(f"  Max probability:  {info_df['probability'].max():.4f}")
    
    print("=" * 80)
    
    # Save predictions
    print(f"\nSaving predictions to: {output_path}")
    info_df.to_csv(output_path, index=False, float_format='%.6f')
    
    print(f"\nPredictions saved successfully!")
    print(f"\nFirst 10 predictions:")
    display_cols = ['unit_id', 'probability', 'predicted_class', 'prediction']
    if mode == 'all_windows':
        display_cols = ['unit_id', 'window_idx', 'end_cycle'] + display_cols[1:]
    print(info_df[display_cols].head(10).to_string(index=False))
    
    # Additional analysis for last_window mode
    if mode == 'last_window':
        print("\n" + "=" * 80)
        print("HIGH RISK UNITS (Predicted Failure within 30 cycles)")
        print("=" * 80)
        high_risk = info_df[info_df['predicted_class'] == 1].sort_values('probability', ascending=False)
        if len(high_risk) > 0:
            print(f"\nTotal high-risk units: {len(high_risk)}")
            print("\nTop 10 highest risk units:")
            print(high_risk[['unit_id', 'last_cycle', 'probability', 'prediction']].head(10).to_string(index=False))
        else:
            print("\nNo high-risk units detected.")
    
    return info_df


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='BiLSTM Binary Classification Inference - Predict engine failure risk',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict on test set (one prediction per unit)
  python bilstm_inference.py --test_file datasets/test_FD001.txt
  
  # Use all sliding windows for detailed analysis
  python bilstm_inference.py --test_file datasets/test_FD001.txt --mode all_windows
  
  # Custom threshold
  python bilstm_inference.py --test_file datasets/test_FD001.txt --threshold 0.45
  
  # CPU inference
  python bilstm_inference.py --test_file datasets/test_FD001.txt --device cpu
        """
    )
    
    parser.add_argument('--checkpoint', type=str, default='runs/FD001/checkpoints/best_model_fixed.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--test_file', type=str, required=True,
                        help='Path to test data file (e.g., datasets/test_FD001.txt)')
    parser.add_argument('--data_dir', type=str, default='runs/FD001',
                        help='Directory containing normalization.json and results.json')
    parser.add_argument('--output', type=str, default='predictions.csv',
                        help='Output CSV file path')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run inference on (cuda or cpu)')
    parser.add_argument('--mode', type=str, default='last_window',
                        choices=['last_window', 'all_windows'],
                        help='Inference mode: last_window (one per unit) or all_windows (sliding)')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Classification threshold (default: from results.json or 0.5)')
    
    args = parser.parse_args()
    
    # Auto-detect device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Run inference
    results = run_inference(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        test_file=args.test_file,
        output_path=args.output,
        device=args.device,
        mode=args.mode,
        threshold=args.threshold
    )
    
    return results


if __name__ == '__main__':
    main()
