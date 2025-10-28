"""
Inference Script for CNN-Transformer Fault Classification Model

This script automatically loads the latest trained model from runs/ directory
and performs predictions on new CMAPSS data, outputting results in JSON format.

Features:
- 🤖 Auto-loads latest model from runs/ directory
- 📊 Outputs comprehensive JSON predictions
- 🔧 Automatic preprocessing with saved scaler
- 📈 Includes confidence scores and engine-level summaries

Usage:
    # Auto-load latest model
    python inference.py --input_file data/test_FD001.txt
    
    # Specify a specific run
    python inference.py --input_file data/test_FD001.txt --run_dir runs/run_20251026_123456

"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import argparse
import pickle
from pathlib import Path
from datetime import datetime
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# MODEL DEFINITION (Must match training architecture)
# ============================================================================

class CNNTransformerClassifier(nn.Module):
    """
    CNN-Transformer model for fault classification
    
    Architecture:
        Input (batch, seq_len, features)
        ↓
        CNN Block 1: Conv1D(64) + ReLU + MaxPool + Dropout
        ↓
        CNN Block 2: Conv1D(128) + ReLU + MaxPool + Dropout
        ↓
        Projection: Linear to d_model (128)
        ↓
        Transformer Encoder: N layers with Multi-Head Attention
        ↓
        Global Average Pooling
        ↓
        Dense Block: FC(256) → FC(128) → FC(num_classes)
        ↓
        Output: Class probabilities
    """
    
    def __init__(self, config, num_features, num_classes=2):
        super(CNNTransformerClassifier, self).__init__()
        
        cnn_cfg = config['model']['cnn']
        trans_cfg = config['model']['transformer']
        dense_cfg = config['model']['dense']
        
        # === CNN Feature Extraction ===
        self.conv1 = nn.Conv1d(
            in_channels=num_features,
            out_channels=cnn_cfg['conv1_filters'],
            kernel_size=cnn_cfg['conv1_kernel_size'],
            padding='same'
        )
        self.pool1 = nn.MaxPool1d(kernel_size=cnn_cfg['pool1_size'])
        self.dropout1 = nn.Dropout(trans_cfg['dropout'])
        
        self.conv2 = nn.Conv1d(
            in_channels=cnn_cfg['conv1_filters'],
            out_channels=cnn_cfg['conv2_filters'],
            kernel_size=cnn_cfg['conv2_kernel_size'],
            padding='same'
        )
        self.pool2 = nn.MaxPool1d(kernel_size=cnn_cfg['pool2_size'])
        self.dropout2 = nn.Dropout(trans_cfg['dropout'])
        
        self.projection = nn.Linear(cnn_cfg['conv2_filters'], trans_cfg['d_model'])
        
        # === Transformer Encoder ===
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=trans_cfg['d_model'],
            nhead=trans_cfg['num_heads'],
            dim_feedforward=trans_cfg['d_ff'],
            dropout=trans_cfg['dropout'],
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=trans_cfg['num_layers']
        )
        
        # === Classification Head ===
        self.fc1 = nn.Linear(trans_cfg['d_model'], dense_cfg['units'][0])
        self.dropout3 = nn.Dropout(trans_cfg['dropout'])
        
        self.fc2 = nn.Linear(dense_cfg['units'][0], dense_cfg['units'][1])
        self.dropout4 = nn.Dropout(trans_cfg['dropout'])
        
        self.fc3 = nn.Linear(dense_cfg['units'][1], num_classes)
        
    def forward(self, x):
        # CNN expects (batch, features, seq_len)
        x = x.transpose(1, 2)
        
        # CNN Block 1
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # CNN Block 2
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Back to (batch, seq_len, features) for Transformer
        x = x.transpose(1, 2)
        
        # Project to d_model
        x = self.projection(x)
        
        # Transformer Encoder
        x = self.transformer(x)
        
        # Global Average Pooling
        x = torch.mean(x, dim=1)
        
        # Dense Classification Head
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout4(x)
        
        x = self.fc3(x)
        
        return x


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def find_latest_run(runs_dir='runs'):
    """
    Find the most recent run directory
    
    Args:
        runs_dir: Base directory containing run folders
        
    Returns:
        Path to latest run directory
    """
    runs_path = Path(runs_dir)
    if not runs_path.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")
    
    run_folders = sorted([d for d in runs_path.iterdir() if d.is_dir() and d.name.startswith('run_')])
    
    if not run_folders:
        raise FileNotFoundError(f"No run folders found in {runs_dir}")
    
    latest_run = run_folders[-1]  # Most recent by timestamp
    return latest_run


def load_run_info(run_dir):
    """
    Load run information from run directory
    
    Args:
        run_dir: Path to run directory
        
    Returns:
        Dictionary with run info
    """
    run_info_path = Path(run_dir) / 'run_info.json'
    training_summary_path = Path(run_dir) / 'results' / 'training_summary.json'
    
    info = {}
    if run_info_path.exists():
        with open(run_info_path, 'r') as f:
            info.update(json.load(f))
    
    if training_summary_path.exists():
        with open(training_summary_path, 'r') as f:
            info['training_summary'] = json.load(f)
    
    return info


# ============================================================================
# INFERENCE CLASS
# ============================================================================

class FaultClassifierInference:
    """
    Inference pipeline for CMAPSS fault classification
    Automatically loads latest model from runs/ directory
    """
    
    def __init__(self, 
                 run_dir=None,
                 device=None):
        """
        Initialize inference pipeline
        
        Args:
            run_dir: Path to specific run directory. If None, uses latest from runs/
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        print("="*80)
        print("🚀 INITIALIZING FAULT CLASSIFIER INFERENCE PIPELINE")
        print("="*80)
        
        # Find run directory
        if run_dir is None:
            print("\n📂 Auto-detecting latest model...")
            run_dir = find_latest_run()
        else:
            run_dir = Path(run_dir)
        
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        
        self.run_dir = run_dir
        print(f"   ✅ Using run: {run_dir.name}")
        
        # Load run info
        self.run_info = load_run_info(run_dir)
        if 'training_summary' in self.run_info:
            ts = self.run_info['training_summary']
            print(f"   📊 Training: Epoch {ts['best_epoch']}, Val Acc: {ts['best_metrics']['val_acc']*100:.2f}%")
        
        # Set paths
        self.config_path = run_dir / 'config.json'
        self.model_path = run_dir / 'models' / 'best_model.pth'
        self.scaler_path = run_dir / 'models' / 'scaler.pkl'
        
        # Verify files exist
        for name, path in [('Config', self.config_path), 
                           ('Model', self.model_path), 
                           ('Scaler', self.scaler_path)]:
            if not path.exists():
                raise FileNotFoundError(f"{name} not found: {path}")
        
        # Device configuration
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"\n📱 Device: {self.device}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
        
        # Load configuration
        print(f"\n📂 Loading configuration from: {self.config_path.name}")
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
        print("   ✅ Config loaded")
        
        # Extract key parameters
        self.window_size = self.config['preprocessing']['window_size']
        self.rul_threshold = self.config['dataset']['rul_threshold']
        
        # Define column names
        self.column_names = ['unit', 'time'] + \
                           [f'op_setting_{i}' for i in range(1, 4)] + \
                           [f'sensor_{i}' for i in range(1, 22)]
        
        # Define feature columns
        op_settings = ['op_setting_1', 'op_setting_2', 'op_setting_3']
        if 'selected_sensors' in self.config['dataset'] and self.config['dataset']['selected_sensors']:
            selected_sensors = [f'sensor_{i}' for i in self.config['dataset']['selected_sensors']]
        else:
            # Default: remove constant sensors
            selected_sensors = [f'sensor_{i}' for i in [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]]
        
        self.feature_cols = op_settings + selected_sensors
        self.num_features = len(self.feature_cols)
        
        print(f"\n🔧 Configuration:")
        print(f"   Window size: {self.window_size} cycles")
        print(f"   RUL threshold: {self.rul_threshold} cycles")
        print(f"   Features: {self.num_features} ({len(op_settings)} op_settings + {len(selected_sensors)} sensors)")
        
        # Load scaler
        print(f"\n📊 Loading scaler from: {self.scaler_path.name}")
        with open(self.scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        print("   ✅ Scaler loaded")
        
        # Load model
        print(f"\n🧠 Loading model from: {self.model_path.name}")
        self.model = CNNTransformerClassifier(
            self.config, 
            self.num_features, 
            num_classes=2
        ).to(self.device)
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print("   ✅ Model loaded and set to evaluation mode")
        print(f"   📈 Model was trained until epoch {checkpoint['epoch']+1}")
        print(f"   📊 Best validation accuracy: {checkpoint['val_acc']*100:.2f}%")
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"   🔢 Total parameters: {total_params:,}")
        
        print("\n" + "="*80)
        print("✅ INFERENCE PIPELINE READY")
        print("="*80 + "\n")
    
    def load_data(self, input_file):
        """
        Load data from input file (same format as CMAPSS test data)
        
        Args:
            input_file: Path to input data file
        
        Returns:
            DataFrame with loaded data
        """
        print(f"📂 Loading data from: {input_file}")
        
        # Read data
        df = pd.read_csv(input_file, sep='\s+', header=None, names=self.column_names)
        
        print(f"   ✅ Loaded {df.shape[0]} rows from {df['unit'].nunique()} engines")
        print(f"   📊 Columns: {df.shape[1]}")
        print(f"   🔢 Engine IDs: {df['unit'].min()} to {df['unit'].max()}")
        
        return df
    
    def preprocess_data(self, df):
        """
        Preprocess data: normalize features and create sliding windows
        
        Args:
            df: Input DataFrame
        
        Returns:
            X: numpy array of windows (num_samples, window_size, num_features)
            window_info: DataFrame with metadata about each window
        """
        print("\n🔄 Preprocessing data...")
        
        # Create a copy
        df_norm = df.copy()
        
        # Normalize features using fitted scaler
        print("   Normalizing features...")
        df_norm[self.feature_cols] = self.scaler.transform(df[self.feature_cols])
        
        # Create sliding windows
        print(f"   Creating sliding windows (size={self.window_size})...")
        X = []
        window_info = []
        
        for unit_id in tqdm(df_norm['unit'].unique(), desc='   Processing engines'):
            unit_data = df_norm[df_norm['unit'] == unit_id].sort_values('time').reset_index(drop=True)
            
            features = unit_data[self.feature_cols].values
            
            # Sliding window with step=1
            for i in range(len(features) - self.window_size + 1):
                X.append(features[i:i+self.window_size])
                
                # Store metadata
                window_info.append({
                    'unit': unit_id,
                    'window_id': i,
                    'start_cycle': int(unit_data.iloc[i]['time']),
                    'end_cycle': int(unit_data.iloc[i+self.window_size-1]['time']),
                    'last_cycle': int(unit_data.iloc[i+self.window_size-1]['time'])
                })
        
        X = np.array(X)
        window_info_df = pd.DataFrame(window_info)
        
        print(f"   ✅ Created {X.shape[0]} windows")
        print(f"   📊 Shape: {X.shape} (samples, timesteps, features)")
        
        return X, window_info_df
    
    def predict(self, X, batch_size=64):
        """
        Perform inference on preprocessed windows
        
        Args:
            X: numpy array of windows
            batch_size: Batch size for inference
        
        Returns:
            predictions: numpy array of predicted classes (0 or 1)
            probabilities: numpy array of class probabilities (num_samples, 2)
            confidence: numpy array of prediction confidence (max probability)
        """
        print("\n🔮 Performing predictions...")
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X)
        
        # Create DataLoader
        from torch.utils.data import TensorDataset, DataLoader
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Inference
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for (batch_x,) in tqdm(loader, desc='   Predicting'):
                batch_x = batch_x.to(self.device)
                
                # Forward pass
                outputs = self.model(batch_x)
                probs = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        predictions = np.array(all_preds)
        probabilities = np.array(all_probs)
        confidence = np.max(probabilities, axis=1)
        
        print(f"   ✅ Predictions completed")
        print(f"   📊 Results:")
        print(f"      Normal (0):   {(predictions == 0).sum():6,} ({(predictions == 0).sum()/len(predictions)*100:5.2f}%)")
        print(f"      Degraded (1): {(predictions == 1).sum():6,} ({(predictions == 1).sum()/len(predictions)*100:5.2f}%)")
        print(f"   💪 Average confidence: {confidence.mean():.4f}")
        
        return predictions, probabilities, confidence
    
    def predict_from_file(self, input_file, output_file=None, batch_size=64):
        """
        End-to-end prediction from input file
        
        Args:
            input_file: Path to input data file
            output_file: Path to save predictions (optional)
            batch_size: Batch size for inference
        
        Returns:
            results_df: DataFrame with predictions and metadata
        """
        print("\n" + "="*80)
        print("🎯 STARTING PREDICTION PIPELINE")
        print("="*80)
        
        # Load data
        df = self.load_data(input_file)
        
        # Preprocess
        X, window_info_df = self.preprocess_data(df)
        
        # Predict
        predictions, probabilities, confidence = self.predict(X, batch_size)
        
        # Combine results
        results_df = window_info_df.copy()
        results_df['prediction'] = predictions
        results_df['prediction_label'] = results_df['prediction'].map({
            0: 'Normal',
            1: 'Degraded'
        })
        results_df['prob_normal'] = probabilities[:, 0]
        results_df['prob_degraded'] = probabilities[:, 1]
        results_df['confidence'] = confidence
        
        print("\n📊 PREDICTION SUMMARY")
        print("="*80)
        print(f"\nPer-Engine Results:")
        
        engine_summary = results_df.groupby('unit').agg({
            'prediction': ['count', lambda x: (x == 0).sum(), lambda x: (x == 1).sum()],
            'confidence': 'mean'
        }).round(4)
        engine_summary.columns = ['Total Windows', 'Normal Windows', 'Degraded Windows', 'Avg Confidence']
        
        for idx, row in engine_summary.iterrows():
            degraded_pct = row['Degraded Windows'] / row['Total Windows'] * 100
            status = "🟢 Healthy" if degraded_pct < 20 else "🟡 Warning" if degraded_pct < 50 else "🔴 Degraded"
            print(f"\nEngine {idx:3d}: {status}")
            print(f"  Windows: {int(row['Total Windows'])} total, {int(row['Normal Windows'])} normal, {int(row['Degraded Windows'])} degraded ({degraded_pct:.1f}%)")
            print(f"  Avg confidence: {row['Avg Confidence']:.4f}")
        
        # Save results to JSON
        if output_file:
            # Prepare JSON output
            output_data = {
                'metadata': {
                    'timestamp': pd.Timestamp.now().isoformat(),
                    'input_file': input_file,
                    'model_path': self.model_path,
                    'config_path': self.config_path,
                    'scaler_path': self.scaler_path,
                    'window_size': int(self.window_size),
                    'rul_threshold': int(self.rul_threshold),
                    'num_features': int(self.num_features)
                },
                'summary': {
                    'total_predictions': int(len(predictions)),
                    'total_engines': int(results_df['unit'].nunique()),
                    'normal_count': int((predictions == 0).sum()),
                    'degraded_count': int((predictions == 1).sum()),
                    'normal_percentage': float((predictions == 0).sum() / len(predictions) * 100),
                    'degraded_percentage': float((predictions == 1).sum() / len(predictions) * 100),
                    'average_confidence': float(confidence.mean()),
                    'min_confidence': float(confidence.min()),
                    'max_confidence': float(confidence.max())
                },
                'predictions': [],
                'engine_summary': []
            }
            
            # Add per-window predictions
            for _, row in results_df.iterrows():
                output_data['predictions'].append({
                    'unit': int(row['unit']),
                    'window_id': int(row['window_id']),
                    'start_cycle': int(row['start_cycle']),
                    'end_cycle': int(row['end_cycle']),
                    'last_cycle': int(row['last_cycle']),
                    'prediction': int(row['prediction']),
                    'prediction_label': row['prediction_label'],
                    'probabilities': {
                        'normal': float(row['prob_normal']),
                        'degraded': float(row['prob_degraded'])
                    },
                    'confidence': float(row['confidence'])
                })
            
            # Add engine summary
            for idx, row in engine_summary.iterrows():
                unit_data = results_df[results_df['unit'] == idx]
                latest_pred = unit_data.iloc[-1]
                
                output_data['engine_summary'].append({
                    'unit': int(idx),
                    'total_windows': int(row['Total Windows']),
                    'normal_windows': int(row['Normal Windows']),
                    'degraded_windows': int(row['Degraded Windows']),
                    'degraded_percentage': float(row['Degraded Windows'] / row['Total Windows'] * 100),
                    'avg_confidence': float(row['Avg Confidence']),
                    'latest_prediction': {
                        'cycle': int(latest_pred['last_cycle']),
                        'label': latest_pred['prediction_label'],
                        'probability_degraded': float(latest_pred['prob_degraded']),
                        'confidence': float(latest_pred['confidence'])
                    }
                })
            
            # Save to JSON
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"\n💾 Predictions saved to: {output_file}")
        
        print("\n" + "="*80)
        print("✅ PREDICTION PIPELINE COMPLETED")
        print("="*80 + "\n")
        
        return results_df


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """
    Main function for command-line usage
    """
    parser = argparse.ArgumentParser(
        description='CNN-Transformer Fault Classification Inference - Auto-loads latest model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Auto-load latest model and predict
    python inference.py --input_file data/test_FD001.txt
    
    # Specify a specific run
    python inference.py --input_file data/test_FD001.txt --run_dir runs/run_20251026_123456
    
    # Custom output file
    python inference.py --input_file data/test_FD001.txt --output_file my_predictions.json
    
    # Use GPU
    python inference.py --input_file data/test_FD001.txt --device cuda
        """
    )
    
    parser.add_argument('--input_file', type=str, required=True,
                       help='Path to input data file (same format as CMAPSS test data)')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Path to save predictions (default: predictions_TIMESTAMP.json)')
    parser.add_argument('--run_dir', type=str, default=None,
                       help='Path to specific run directory (default: auto-detect latest from runs/)')
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu'],
                       help='Device to use (default: auto-detect)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for inference (default: 64)')
    
    args = parser.parse_args()
    
    # Default output file with timestamp
    if args.output_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_file = f'predictions_{timestamp}.json'
    
    # Initialize inference pipeline
    inference = FaultClassifierInference(
        run_dir=args.run_dir,
        device=args.device
    )
    
    # Run prediction
    results_df = inference.predict_from_file(
        input_file=args.input_file,
        output_file=args.output_file,
        batch_size=args.batch_size
    )
    
    return results_df


if __name__ == '__main__':
    main()
