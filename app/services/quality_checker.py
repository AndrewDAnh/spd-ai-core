import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from app.core.logging_config import logger


class QualityChecker:
    """Service for checking data quality"""
    
    def __init__(self, outlier_sensitivity: str = "medium"):
        self.outlier_sensitivity = outlier_sensitivity
        self._set_sensitivity_params()
    
    def _set_sensitivity_params(self):
        """Set parameters based on sensitivity level"""
        sensitivity_map = {
            "low": {"iqr_multiplier": 3.0, "z_threshold": 4.0},
            "medium": {"iqr_multiplier": 1.5, "z_threshold": 3.0},
            "high": {"iqr_multiplier": 1.0, "z_threshold": 2.5}
        }
        params = sensitivity_map.get(self.outlier_sensitivity, sensitivity_map["medium"])
        self.iqr_multiplier = params["iqr_multiplier"]
        self.z_threshold = params["z_threshold"]
    
    def check_quality(
        self,
        data: List[Dict[str, float]],
        expected_schema: List[str] = None
    ) -> Dict:
        """
        Perform comprehensive quality checks on data
        
        Args:
            data: Data to check
            expected_schema: Expected column names
            
        Returns:
            Dictionary with quality check results
        """
        try:
            df = pd.DataFrame(data)
            
            issues = []
            
            # Schema validation
            if expected_schema:
                schema_issues = self._check_schema(df, expected_schema)
                issues.extend(schema_issues)
            
            # Missing values check
            missing_issues = self._check_missing_values(df)
            issues.extend(missing_issues)
            
            # Outliers check
            outlier_issues = self._check_outliers(df)
            issues.extend(outlier_issues)
            
            # Range validation
            range_issues = self._check_ranges(df)
            issues.extend(range_issues)
            
            # Summarize
            quality_passed = len([i for i in issues if i['severity'] == 'high']) == 0
            
            summary = {
                'total_issues': len(issues),
                'high_severity': len([i for i in issues if i['severity'] == 'high']),
                'medium_severity': len([i for i in issues if i['severity'] == 'medium']),
                'low_severity': len([i for i in issues if i['severity'] == 'low'])
            }
            
            return {
                'quality_passed': quality_passed,
                'issues': issues,
                'summary': summary
            }
            
        except Exception as e:
            logger.error(f"Error in quality check: {str(e)}")
            raise
    
    def _check_schema(self, df: pd.DataFrame, expected_schema: List[str]) -> List[Dict]:
        """Check if data matches expected schema"""
        issues = []
        
        actual_cols = set(df.columns)
        expected_cols = set(expected_schema)
        
        # Missing columns
        missing_cols = expected_cols - actual_cols
        if missing_cols:
            issues.append({
                'feature': 'schema',
                'type': 'missing_columns',
                'severity': 'high',
                'details': f"Missing columns: {', '.join(missing_cols)}"
            })
        
        # Extra columns
        extra_cols = actual_cols - expected_cols
        if extra_cols:
            issues.append({
                'feature': 'schema',
                'type': 'extra_columns',
                'severity': 'low',
                'details': f"Extra columns: {', '.join(extra_cols)}"
            })
        
        return issues
    
    def _check_missing_values(self, df: pd.DataFrame) -> List[Dict]:
        """Check for missing values"""
        issues = []
        
        for col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                missing_pct = (missing_count / len(df)) * 100
                
                # Determine severity
                if missing_pct > 20:
                    severity = 'high'
                elif missing_pct > 10:
                    severity = 'medium'
                else:
                    severity = 'low'
                
                issues.append({
                    'feature': col,
                    'type': 'missing_values',
                    'severity': severity,
                    'details': f"{missing_pct:.1f}% missing ({missing_count}/{len(df)})"
                })
        
        return issues
    
    def _check_outliers(self, df: pd.DataFrame) -> List[Dict]:
        """Check for outliers using IQR and Z-score methods"""
        issues = []
        
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                # IQR method
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - self.iqr_multiplier * IQR
                upper_bound = Q3 + self.iqr_multiplier * IQR
                
                outliers_iqr = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                
                # Z-score method
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                outliers_z = (z_scores > self.z_threshold).sum()
                
                # Use the more conservative estimate
                outliers = max(outliers_iqr, outliers_z)
                
                if outliers > 0:
                    outlier_pct = (outliers / len(df)) * 100
                    
                    # Determine severity
                    if outlier_pct > 10:
                        severity = 'medium'
                    else:
                        severity = 'low'
                    
                    issues.append({
                        'feature': col,
                        'type': 'outliers',
                        'severity': severity,
                        'details': f"{outliers} outliers detected ({outlier_pct:.1f}%)"
                    })
        
        return issues
    
    def _check_ranges(self, df: pd.DataFrame) -> List[Dict]:
        """Check for unrealistic value ranges"""
        issues = []
        
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                # Check for negative values where they shouldn't exist
                # (This is domain-specific; for sensors, negative values might be suspicious)
                if (df[col] < 0).any():
                    negative_count = (df[col] < 0).sum()
                    issues.append({
                        'feature': col,
                        'type': 'negative_values',
                        'severity': 'low',
                        'details': f"{negative_count} negative values found"
                    })
                
                # Check for constant values (no variation)
                if df[col].std() == 0:
                    issues.append({
                        'feature': col,
                        'type': 'constant_values',
                        'severity': 'medium',
                        'details': "Feature has constant values (no variation)"
                    })
        
        return issues


# Import stats for zscore
from scipy import stats
