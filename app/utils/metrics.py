"""Utility functions for metrics calculation"""

from typing import Dict, List
import numpy as np


def calculate_summary_statistics(values: List[float]) -> Dict[str, float]:
    """Calculate summary statistics for a list of values"""
    if not values:
        return {
            'mean': 0.0,
            'median': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0
        }
    
    return {
        'mean': float(np.mean(values)),
        'median': float(np.median(values)),
        'std': float(np.std(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values))
    }


def aggregate_drift_scores(feature_drifts: Dict[str, Dict]) -> Dict[str, Dict]:
    """Aggregate drift scores across features"""
    aggregated = {}
    
    for feature, drift_info in feature_drifts.items():
        score = drift_info.get('score', 0.0)
        
        if feature not in aggregated:
            aggregated[feature] = {
                'scores': [],
                'mean_drift': 0.0,
                'max_drift': 0.0
            }
        
        aggregated[feature]['scores'].append(score)
    
    # Calculate mean and max
    for feature, data in aggregated.items():
        data['mean_drift'] = float(np.mean(data['scores']))
        data['max_drift'] = float(np.max(data['scores']))
        del data['scores']  # Remove raw scores from output
    
    return aggregated
