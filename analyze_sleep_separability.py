import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

def main():
    file_path = '/Users/dcaric/Working/pythonWorking/PaolaFinalWork/pilot_files/all_task_ready.csv'
    df = pd.read_csv(file_path)
    
    exclude_cols = ['Subject_ID', 'Task', 'Sleep_Hours', 'Gender']
    feature_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
    
    # Define target
    df['Target'] = (df['Sleep_Hours'] > 6).astype(int)
    
    group1 = df[df['Target'] == 0][feature_cols]
    group2 = df[df['Target'] == 1][feature_cols]
    
    # 1. Check Class Imbalance and Subject distribution
    print("--- Dataset Composition ---")
    print(f"Group 1 (<= 6h): {len(group1)} rows, {df[df['Target'] == 0]['Subject_ID'].nunique()} unique subjects")
    print(f"Group 2 (> 6h): {len(group2)} rows, {df[df['Target'] == 1]['Subject_ID'].nunique()} unique subjects")
    
    # 2. Calculate Cohen's d for feature separability
    # Cohen's d = (Mean1 - Mean2) / Pooled_SD
    mean1 = group1.mean()
    mean2 = group2.mean()
    var1 = group1.var()
    var2 = group2.var()
    n1 = len(group1)
    n2 = len(group2)
    
    pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    cohens_d = ((mean1 - mean2) / pooled_sd).abs()
    
    print("\n--- Feature Separability (Cohen's d) ---")
    print("Interpretation: ~0.2 is Small, ~0.5 is Medium, >0.8 is Large")
    print(f"Max Effect Size: {cohens_d.max():.4f} (Feature: {cohens_d.idxmax()})")
    print(f"Average Effect Size across all features: {cohens_d.mean():.4f}")
    
    # Print the top 5 features with highest effect size
    print("Top 5 features with highest effect size:")
    print(cohens_d.sort_values(ascending=False).head(5))

    # 3. Quick Model Test (Random Forest)
    X = df[feature_cols].fillna(0)
    y = df['Target']
    
    # GroupKFold by Subject would be proper, but let's do a quick random split first
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    clf = RandomForestClassifier(random_state=42, n_estimators=50)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    print("\n--- Quick Random Forest Model Performance ---")
    print(classification_report(y_test, y_pred, target_names=['<= 6h (Bad)', '> 6h (Good)']))

if __name__ == '__main__':
    main()
