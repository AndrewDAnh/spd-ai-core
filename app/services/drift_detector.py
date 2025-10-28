import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
from app.core.logging_config import logger


class DriftDetector:
    """Service for detecting data drift"""
    
    def __init__(self, threshold: float = 0.2):
        self.threshold = threshold
    
    def detect_drift(
        self,
        reference_data: List[Dict[str, float]],
        current_data: List[Dict[str, float]]
    ) -> Dict:
        """
        Detect drift between reference and current data
        
        Args:
            reference_data: Baseline/reference data
            current_data: Current production data
            
        Returns:
            Dictionary with drift results
        """
        try:
            # Convert to DataFrames
            ref_df = pd.DataFrame(reference_data)
            cur_df = pd.DataFrame(current_data)
            
            # Ensure same columns
            common_cols = list(set(ref_df.columns) & set(cur_df.columns))
            
            feature_drifts = {}
            drift_scores = []
            
            for col in common_cols:
                # Determine if column is numerical or categorical
                if self._is_numerical(ref_df[col]):
                    drift_info = self._detect_numerical_drift(
                        ref_df[col].values,
                        cur_df[col].values
                    )
                else:
                    drift_info = self._detect_categorical_drift(
                        ref_df[col].values,
                        cur_df[col].values
                    )
                
                feature_drifts[col] = drift_info
                drift_scores.append(drift_info['score'])
            
            # Calculate overall drift score
            overall_drift_score = np.mean(drift_scores) if drift_scores else 0.0
            drift_detected = overall_drift_score > self.threshold
            
            return {
                'drift_detected': drift_detected,
                'overall_drift_score': float(overall_drift_score),
                'feature_drifts': feature_drifts,
                'threshold': self.threshold
            }
            
        except Exception as e:
            logger.error(f"Error in drift detection: {str(e)}")
            raise
    
    def _is_numerical(self, series: pd.Series) -> bool:
        """Check if series is numerical"""
        return pd.api.types.is_numeric_dtype(series)
    
    def _detect_numerical_drift(
        self,
        reference: np.ndarray,
        current: np.ndarray
    ) -> Dict:
        """
        Detect drift for numerical features using KS test and PSI
        """
        # Kolmogorov-Smirnov test
        ks_statistic, p_value = stats.ks_2samp(reference, current)
        
        # Population Stability Index (PSI)
        psi_score = self._calculate_psi(reference, current)
        
        # Use PSI as primary score for numerical features
        score = psi_score
        
        # Determine status
        if score > 0.2:
            status = "high_drift"
        elif score > 0.1:
            status = "medium_drift"
        else:
            status = "no_drift"
        
        return {
            'score': float(score),
            'method': 'psi',
            'status': status,
            'p_value': float(p_value),
            'ks_statistic': float(ks_statistic)
        }
    
    def _calculate_psi(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        bins: int = 10
    ) -> float:
        """
        Calculate Population Stability Index (PSI)
        
        PSI formula: sum((current_pct - reference_pct) * ln(current_pct / reference_pct))
        """
        # Create bins based on reference data
        min_val = min(reference.min(), current.min())
        max_val = max(reference.max(), current.max())
        
        # Handle edge case where all values are the same
        if min_val == max_val:
            return 0.0
        
        bin_edges = np.linspace(min_val, max_val, bins + 1)
        
        # Calculate distributions
        ref_hist, _ = np.histogram(reference, bins=bin_edges)
        cur_hist, _ = np.histogram(current, bins=bin_edges)
        
        # Convert to percentages
        ref_pct = ref_hist / len(reference)
        cur_pct = cur_hist / len(current)
        
        # Avoid division by zero
        ref_pct = np.where(ref_pct == 0, 0.0001, ref_pct)
        cur_pct = np.where(cur_pct == 0, 0.0001, cur_pct)
        
        # Calculate PSI
        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        
        return float(abs(psi))
    
    def _detect_categorical_drift(
        self,
        reference: np.ndarray,
        current: np.ndarray
    ) -> Dict:
        """
        Detect drift for categorical features using Chi-square test
        """
        # Get unique categories
        categories = np.unique(np.concatenate([reference, current]))
        
        # Calculate distributions
        ref_counts = pd.Series(reference).value_counts()
        cur_counts = pd.Series(current).value_counts()
        
        # Align categories
        ref_freq = [ref_counts.get(cat, 0) for cat in categories]
        cur_freq = [cur_counts.get(cat, 0) for cat in categories]
        
        # Chi-square test
        try:
            chi2_stat, p_value = stats.chisquare(cur_freq, ref_freq)
            score = chi2_stat / (len(reference) + len(current))
        except:
            score = 0.0
            p_value = 1.0
        
        # Determine status
        if p_value < 0.01:
            status = "high_drift"
        elif p_value < 0.05:
            status = "medium_drift"
        else:
            status = "no_drift"
        
        return {
            'score': float(score),
            'method': 'chi_square',
            'status': status,
            'p_value': float(p_value)
        }

