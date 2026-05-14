import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

def main():
    file_path = '/Users/dcaric/Working/pythonWorking/PaolaFinalWork/pilot_files/all_task_ready.csv'
    df = pd.read_csv(file_path)
    
    # Define target: 0 for <=6h (Bad), 1 for >6h (Good)
    df['Target'] = (df['Sleep_Hours'] > 6).astype(int)
    
    # 1. Undersample the majority class (>6h) to balance the dataset
    bad_sleep = df[df['Target'] == 0]
    good_sleep = df[df['Target'] == 1]
    
    # Randomly sample 'good_sleep' to match the number of 'bad_sleep' rows
    good_sleep_sampled = good_sleep.sample(n=len(bad_sleep), random_state=42)
    
    # Combine back into a single balanced dataframe
    balanced_df = pd.concat([bad_sleep, good_sleep_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print("--- Balanced Dataset Composition ---")
    print(f"Group 1 (<= 6h): {len(balanced_df[balanced_df['Target'] == 0])} rows")
    print(f"Group 2 (> 6h) [Undersampled]: {len(balanced_df[balanced_df['Target'] == 1])} rows")
    
    exclude_cols = ['Subject_ID', 'Task', 'Sleep_Hours', 'Gender', 'Target']
    feature_cols = [col for col in balanced_df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(balanced_df[col])]
    
    X = balanced_df[feature_cols].fillna(0)
    y = balanced_df['Target']
    groups = balanced_df['Subject_ID']
    
    # 2. Strict Group Split to prevent Data Leakage (Subjects in Test are NEVER in Train)
    # This forces the model to actually learn sleep patterns, not memorize subjects.
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Train model
    clf = RandomForestClassifier(random_state=42, n_estimators=100)
    clf.fit(X_train, y_train)
    
    # Predict
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\n--- Model Performance on UNSEEN Subjects (True Generalization) ---")
    print(f"Accuracy: {acc:.2f} (50% is random guessing)")
    print("\nClassification Report:")
    # Some targets might be missing in test set if the group split was unlucky, but we'll try to map them.
    # To avoid errors if a class is missing:
    target_names = ['<= 6h (Bad)', '> 6h (Good)']
    labels = np.unique(y_test)
    names = [target_names[i] for i in labels]
    print(classification_report(y_test, y_pred, labels=labels, target_names=names))

if __name__ == '__main__':
    main()
