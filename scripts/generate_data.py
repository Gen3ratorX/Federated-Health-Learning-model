"""
Synthetic Hospital Data Generator for Federated Learning POC
Generates Non-IID patient datasets for 3 hospitals with different distributions
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json

# Set random seed for reproducibility
np.random.seed(42)


def generate_patient_features(n_samples, hospital_id, distribution_type='non_iid'):
    """
    Generate synthetic patient features with hospital-specific distributions
    
    Args:
        n_samples: Number of patients to generate
        hospital_id: Hospital identifier (1, 2, or 3)
        distribution_type: 'iid' or 'non_iid'
    
    Returns:
        DataFrame with patient features
    """
    
    # Base distributions (affected by hospital_id for Non-IID)
    if distribution_type == 'non_iid':
        # Hospital 1: Urban hospital - younger population, more smokers
        # Hospital 2: Suburban hospital - middle-aged, balanced
        # Hospital 3: Rural hospital - older population, more diabetes
        
        age_params = {
            1: {'mean': 45, 'std': 15},  # Younger
            2: {'mean': 55, 'std': 12},  # Middle-aged
            3: {'mean': 65, 'std': 10}   # Older
        }
        
        smoking_rates = {1: 0.35, 2: 0.25, 3: 0.20}
        diabetes_rates = {1: 0.15, 2: 0.20, 3: 0.30}
        
    else:  # IID - same distribution across all hospitals
        age_params = {1: {'mean': 55, 'std': 15}, 2: {'mean': 55, 'std': 15}, 3: {'mean': 55, 'std': 15}}
        smoking_rates = {1: 0.25, 2: 0.25, 3: 0.25}
        diabetes_rates = {1: 0.20, 2: 0.20, 3: 0.20}
    
    # Generate features
    age = np.clip(
        np.random.normal(age_params[hospital_id]['mean'], age_params[hospital_id]['std'], n_samples),
        18, 90
    ).astype(int)
    
    # BMI correlates with age
    bmi = np.clip(
        22 + (age - 50) * 0.1 + np.random.normal(0, 4, n_samples),
        15, 45
    )
    
    # Blood pressure increases with age and BMI
    bp_systolic = np.clip(
        100 + (age - 40) * 0.5 + (bmi - 25) * 0.8 + np.random.normal(0, 15, n_samples),
        90, 200
    ).astype(int)
    
    bp_diastolic = np.clip(
        65 + (age - 40) * 0.3 + (bmi - 25) * 0.5 + np.random.normal(0, 10, n_samples),
        60, 130
    ).astype(int)
    
    # Cholesterol increases with age
    cholesterol = np.clip(
        170 + (age - 40) * 0.8 + np.random.normal(0, 30, n_samples),
        120, 300
    ).astype(int)
    
    # Glucose (affected by diabetes status)
    smoking = np.random.binomial(1, smoking_rates[hospital_id], n_samples)
    diabetes = np.random.binomial(1, diabetes_rates[hospital_id], n_samples)
    
    glucose = np.clip(
        90 + diabetes * 40 + np.random.normal(0, 20, n_samples),
        70, 200
    ).astype(int)
    
    # Heart rate
    heart_rate = np.clip(
        70 + np.random.normal(0, 12, n_samples),
        50, 120
    ).astype(int)
    
    # Family history (30% of population)
    family_history = np.random.binomial(1, 0.3, n_samples)
    
    # Create DataFrame
    df = pd.DataFrame({
        'patient_id': [f'H{hospital_id}_P{i:05d}' for i in range(n_samples)],
        'age': age,
        'bmi': bmi.round(1),
        'bp_systolic': bp_systolic,
        'bp_diastolic': bp_diastolic,
        'cholesterol': cholesterol,
        'glucose': glucose,
        'heart_rate': heart_rate,
        'smoking': smoking,
        'diabetes': diabetes,
        'family_history': family_history
    })
    
    return df


def calculate_risk_score(df):
    """
    Calculate cardiovascular risk score based on clinical features
    
    Risk factors:
    - Age > 60: +2 points
    - BMI > 30: +1.5 points
    - High BP (>140/90): +2 points
    - High cholesterol (>240): +1.5 points
    - High glucose (>125): +2 points
    - Smoking: +2 points
    - Diabetes: +2.5 points
    - Family history: +1 point
    
    At-Risk if score >= 5
    """
    
    risk_score = np.zeros(len(df))
    
    # Age factor
    risk_score += (df['age'] > 60) * 2.0
    
    # BMI factor
    risk_score += (df['bmi'] > 30) * 1.5
    
    # Blood pressure
    risk_score += ((df['bp_systolic'] > 140) | (df['bp_diastolic'] > 90)) * 2.0
    
    # Cholesterol
    risk_score += (df['cholesterol'] > 240) * 1.5
    
    # Glucose
    risk_score += (df['glucose'] > 125) * 2.0
    
    # Smoking
    risk_score += df['smoking'] * 2.0
    
    # Diabetes
    risk_score += df['diabetes'] * 2.5
    
    # Family history
    risk_score += df['family_history'] * 1.0
    
    # Binary classification: At-Risk if score >= 5
    df['risk_score'] = risk_score
    df['at_risk'] = (risk_score >= 5).astype(int)
    
    return df


def generate_hospital_dataset(hospital_id, n_samples, distribution_type='non_iid', output_dir='data'):
    """
    Generate complete dataset for a single hospital
    """
    print(f"\n🏥 Generating data for Hospital {hospital_id}...")
    
    # Generate features
    df = generate_patient_features(n_samples, hospital_id, distribution_type)
    
    # Calculate risk
    df = calculate_risk_score(df)
    
    # Statistics
    at_risk_pct = (df['at_risk'].sum() / len(df)) * 100
    avg_age = df['age'].mean()
    smoking_pct = (df['smoking'].sum() / len(df)) * 100
    diabetes_pct = (df['diabetes'].sum() / len(df)) * 100
    
    print(f"  📊 Patients: {n_samples}")
    print(f"  📊 At-Risk: {df['at_risk'].sum()} ({at_risk_pct:.1f}%)")
    print(f"  📊 Avg Age: {avg_age:.1f} years")
    print(f"  📊 Smokers: {smoking_pct:.1f}%")
    print(f"  📊 Diabetes: {diabetes_pct:.1f}%")
    
    # Save to CSV
    output_path = Path(output_dir) / f'hospital_{hospital_id}' / 'patient_data.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"  ✅ Saved to: {output_path}")
    
    # Save metadata
    metadata = {
        'hospital_id': hospital_id,
        'n_samples': n_samples,
        'distribution_type': distribution_type,
        'at_risk_count': int(df['at_risk'].sum()),
        'at_risk_percentage': float(at_risk_pct),
        'avg_age': float(avg_age),
        'smoking_percentage': float(smoking_pct),
        'diabetes_percentage': float(diabetes_pct),
        'features': list(df.columns)
    }
    
    metadata_path = Path(output_dir) / f'hospital_{hospital_id}' / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return df, metadata


def main():
    """
    Generate datasets for all hospitals
    """
    print("=" * 60)
    print("🏥 FEDERATED LEARNING POC - DATA GENERATION")
    print("=" * 60)
    
    # Configuration
    hospitals = {
        1: {'name': 'Urban Medical Center', 'samples': 1000},
        2: {'name': 'Suburban General Hospital', 'samples': 800},
        3: {'name': 'Rural Community Hospital', 'samples': 600}
    }
    
    distribution_type = 'non_iid'  # Change to 'iid' for uniform distribution
    
    print(f"\n📋 Configuration:")
    print(f"  - Distribution: {distribution_type.upper()}")
    print(f"  - Total Hospitals: {len(hospitals)}")
    print(f"  - Total Patients: {sum(h['samples'] for h in hospitals.values())}")
    
    # Generate datasets
    all_data = []
    all_metadata = []
    
    for hospital_id, config in hospitals.items():
        df, metadata = generate_hospital_dataset(
            hospital_id=hospital_id,
            n_samples=config['samples'],
            distribution_type=distribution_type,
            output_dir='data'
        )
        all_data.append(df)
        all_metadata.append(metadata)
    
    # Generate combined statistics
    print("\n" + "=" * 60)
    print("📊 OVERALL STATISTICS")
    print("=" * 60)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    print(f"\n  Total Patients: {len(combined_df)}")
    print(f"  Total At-Risk: {combined_df['at_risk'].sum()} ({(combined_df['at_risk'].sum() / len(combined_df)) * 100:.1f}%)")
    print(f"  Age Range: {combined_df['age'].min()}-{combined_df['age'].max()} years")
    print(f"  Avg BMI: {combined_df['bmi'].mean():.1f}")
    
    # Class distribution per hospital
    print("\n  Class Distribution by Hospital:")
    for i, metadata in enumerate(all_metadata, 1):
        print(f"    Hospital {i}: {metadata['at_risk_count']}/{metadata['n_samples']} at-risk ({metadata['at_risk_percentage']:.1f}%)")
    
    # Save combined metadata
    combined_metadata = {
        'total_patients': len(combined_df),
        'distribution_type': distribution_type,
        'hospitals': all_metadata
    }
    
    metadata_path = Path('data') / 'combined_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(combined_metadata, f, indent=2)
    
    print(f"\n✅ Data generation complete!")
    print(f"📁 Files saved in: data/hospital_{{1,2,3}}/patient_data.csv")
    print("=" * 60)


if __name__ == "__main__":
    main()