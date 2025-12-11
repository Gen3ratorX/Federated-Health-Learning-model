"""
Data preprocessing utilities
Handles missing values, outliers, and feature engineering
"""

import pandas as pd
import numpy as np
from typing import List, Optional


class DataPreprocessor:
    """
    Handles data cleaning and preprocessing
    """
    
    def __init__(self):
        self.outlier_thresholds = {
            'age': (18, 90),
            'bmi': (15, 45),
            'bp_systolic': (90, 200),
            'bp_diastolic': (60, 130),
            'cholesterol': (120, 300),
            'glucose': (70, 200),
            'heart_rate': (50, 120)
        }
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values using appropriate strategies
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with imputed values
        """
        df_clean = df.copy()
        
        # For numerical features: fill with median
        numerical_cols = ['age', 'bmi', 'bp_systolic', 'bp_diastolic', 
                         'cholesterol', 'glucose', 'heart_rate']
        
        for col in numerical_cols:
            if col in df_clean.columns:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
        
        # For binary features: fill with mode
        binary_cols = ['smoking', 'diabetes', 'family_history']
        
        for col in binary_cols:
            if col in df_clean.columns:
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
        
        return df_clean
    
    def remove_outliers(self, df: pd.DataFrame, method: str = 'clip') -> pd.DataFrame:
        """
        Handle outliers using clipping or removal
        
        Args:
            df: Input DataFrame
            method: 'clip' to clip values, 'remove' to remove rows
        
        Returns:
            DataFrame with outliers handled
        """
        df_clean = df.copy()
        
        if method == 'clip':
            # Clip values to acceptable ranges
            for col, (min_val, max_val) in self.outlier_thresholds.items():
                if col in df_clean.columns:
                    df_clean[col] = df_clean[col].clip(min_val, max_val)
        
        elif method == 'remove':
            # Remove rows with outliers
            mask = pd.Series([True] * len(df_clean))
            
            for col, (min_val, max_val) in self.outlier_thresholds.items():
                if col in df_clean.columns:
                    mask &= (df_clean[col] >= min_val) & (df_clean[col] <= max_val)
            
            df_clean = df_clean[mask]
        
        return df_clean
    
    def add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add engineered features (optional - for improved model performance)
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with additional features
        """
        df_enhanced = df.copy()
        
        # Mean arterial pressure (MAP)
        if 'bp_systolic' in df.columns and 'bp_diastolic' in df.columns:
            df_enhanced['map'] = (df['bp_systolic'] + 2 * df['bp_diastolic']) / 3
        
        # Pulse pressure
        if 'bp_systolic' in df.columns and 'bp_diastolic' in df.columns:
            df_enhanced['pulse_pressure'] = df['bp_systolic'] - df['bp_diastolic']
        
        # Age groups (for categorical analysis)
        if 'age' in df.columns:
            df_enhanced['age_group'] = pd.cut(
                df['age'],
                bins=[0, 40, 60, 100],
                labels=['young', 'middle', 'senior']
            )
        
        # BMI categories
        if 'bmi' in df.columns:
            df_enhanced['bmi_category'] = pd.cut(
                df['bmi'],
                bins=[0, 18.5, 25, 30, 100],
                labels=['underweight', 'normal', 'overweight', 'obese']
            )
        
        return df_enhanced
    
    def preprocess(
        self,
        df: pd.DataFrame,
        handle_missing: bool = True,
        handle_outliers: bool = True,
        add_features: bool = False
    ) -> pd.DataFrame:
        """
        Full preprocessing pipeline
        
        Args:
            df: Input DataFrame
            handle_missing: Whether to impute missing values
            handle_outliers: Whether to handle outliers
            add_features: Whether to add derived features
        
        Returns:
            Preprocessed DataFrame
        """
        df_processed = df.copy()
        
        if handle_missing:
            df_processed = self.handle_missing_values(df_processed)
        
        if handle_outliers:
            df_processed = self.remove_outliers(df_processed, method='clip')
        
        if add_features:
            df_processed = self.add_derived_features(df_processed)
        
        return df_processed


if __name__ == "__main__":
    """Test the preprocessor"""
    print("Testing DataPreprocessor...")
    
    # Create sample data with issues
    sample_data = pd.DataFrame({
        'age': [45, 200, 35, np.nan, 55],  # Outlier and missing
        'bmi': [25.5, 50, 22.1, 28.3, np.nan],  # Outlier and missing
        'bp_systolic': [120, 250, 110, 135, 140],  # Outlier
        'bp_diastolic': [80, 95, 70, 85, 90],
        'cholesterol': [180, 350, 200, 220, np.nan],  # Outlier and missing
        'glucose': [95, 100, 180, 105, 110],
        'heart_rate': [70, 150, 68, 72, 75],  # Outlier
        'smoking': [0, 1, 0, np.nan, 1],  # Missing
        'diabetes': [0, 0, 1, 1, 0],
        'family_history': [1, 0, 1, 0, np.nan]  # Missing
    })
    
    print("\nOriginal data:")
    print(sample_data)
    print(f"\nMissing values:\n{sample_data.isnull().sum()}")
    
    # Preprocess
    preprocessor = DataPreprocessor()
    cleaned_data = preprocessor.preprocess(sample_data)
    
    print("\n\nCleaned data:")
    print(cleaned_data)
    print(f"\nMissing values after cleaning:\n{cleaned_data.isnull().sum()}")
    
    print("\n✅ Preprocessor test completed!")