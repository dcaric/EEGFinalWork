import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

def main():
    print("Loading data...")
    input_file = 'pilot_files/all_task_ready.csv'
    output_dir = 'results/subject_normalized_analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(input_file)
    
    # Balance subjects: equal number of subjects with <=6h and >6h
    subject_sleep = df.groupby('Subject_ID')['Sleep_Hours'].first()
    bad_subjects = subject_sleep[subject_sleep <= 6].index.tolist()
    good_subjects = subject_sleep[subject_sleep > 6].index.tolist()
    
    # Sample good subjects to match the number of bad subjects
    np.random.seed(42)
    sampled_good_subjects = np.random.choice(good_subjects, size=len(bad_subjects), replace=False).tolist()
    
    balanced_subjects = bad_subjects + sampled_good_subjects
    df = df[df['Subject_ID'].isin(balanced_subjects)].copy().reset_index(drop=True)
    
    print(f"Balanced dataset: {len(bad_subjects)} subjects with <=6h, {len(sampled_good_subjects)} subjects with >6h.")
    
    exclude_cols = ['Subject_ID', 'Task', 'Sleep_Hours', 'Gender']
    feature_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
    
    # Subject-wise normalization (Z-score)
    print("Applying subject-wise normalization...")
    def normalize_subject(group):
        # Subtract mean and divide by std, if std is 0, return 0
        mean = group.mean()
        std = group.std()
        if std == 0 or pd.isna(std):
            return group * 0
        return (group - mean) / std
    
    # Create normalized features dataframe
    df_norm = df.copy()
    df_norm[feature_cols] = df.groupby('Subject_ID')[feature_cols].transform(normalize_subject)
    
    X = df_norm[feature_cols].fillna(0)
    y = df_norm['Sleep_Hours']
    groups = df_norm['Subject_ID']
    
    print("Running GroupKFold regression...")
    gkf = GroupKFold(n_splits=5)
    
    df_preds = df_norm[['Subject_ID', 'Task', 'Sleep_Hours', 'Gender']].copy()
    df_preds['Predicted_Sleep_Hours'] = np.nan
    df_preds['Fold'] = -1
    
    rf = RandomForestRegressor(random_state=42, n_estimators=100)
    dummy = DummyRegressor(strategy='mean')
    
    rf_preds_all = []
    dummy_preds_all = []
    y_true_all = []
    
    fold_idx = 1
    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Train RF
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        
        # Train Dummy
        dummy.fit(X_train, y_train)
        dummy_pred = dummy.predict(X_test)
        
        rf_preds_all.extend(rf_pred)
        dummy_preds_all.extend(dummy_pred)
        y_true_all.extend(y_test)
        
        df_preds.loc[test_idx, 'Predicted_Sleep_Hours'] = rf_pred
        df_preds.loc[test_idx, 'Fold'] = fold_idx
        fold_idx += 1

    # Save predictions
    df_preds.to_csv(os.path.join(output_dir, 'all_task_grouped_regression_predictions.csv'), index=False)
    
    # Calculate metrics
    rf_rmse = np.sqrt(mean_squared_error(y_true_all, rf_preds_all))
    rf_mae = mean_absolute_error(y_true_all, rf_preds_all)
    rf_r2 = r2_score(y_true_all, rf_preds_all)
    
    dummy_rmse = np.sqrt(mean_squared_error(y_true_all, dummy_preds_all))
    dummy_r2 = r2_score(y_true_all, dummy_preds_all)
    
    print("Generating regression scatter plot...")
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='Sleep_Hours', y='Predicted_Sleep_Hours', data=df_preds, alpha=0.5)
    
    # Perfect prediction line
    min_val = min(df_preds['Sleep_Hours'].min(), df_preds['Predicted_Sleep_Hours'].min())
    max_val = max(df_preds['Sleep_Hours'].max(), df_preds['Predicted_Sleep_Hours'].max())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
    
    plt.title('Actual vs Predicted Sleep Hours (Subject Normalized)')
    plt.xlabel('Actual Sleep Hours')
    plt.ylabel('Predicted Sleep Hours')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'grouped_regression_scatter.png'))
    plt.close()
    
    print("Performing KMeans clustering...")
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X)
    df_norm['Cluster'] = clusters
    
    # Save subject clusters
    df_norm[['Subject_ID', 'Task', 'Sleep_Hours', 'Cluster']].to_csv(os.path.join(output_dir, 'subject_clusters.csv'), index=False)
    
    # Cluster summary
    cluster_summary = df_norm.groupby('Cluster').agg(
        N_Subjects=('Subject_ID', 'nunique'),
        Mean_Sleep_Hours=('Sleep_Hours', 'mean'),
        Std_Sleep_Hours=('Sleep_Hours', 'std'),
        Min_Sleep_Hours=('Sleep_Hours', 'min'),
        Max_Sleep_Hours=('Sleep_Hours', 'max')
    ).reset_index()
    
    cluster_summary.to_csv(os.path.join(output_dir, 'cluster_sleep_summary.csv'), index=False)
    
    print("Generating cluster boxplot...")
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Cluster', y='Sleep_Hours', data=df_norm)
    plt.title('Sleep Hours Distribution per Cluster')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cluster_sleep_hours.png'))
    plt.close()
    
    print("Writing summary file...")
    summary_text = f"""Input file: {input_file}
Rows used: {len(df)}
Subjects used: {df['Subject_ID'].nunique()}
Tasks used: {df['Task'].nunique()}
Numeric features used: {len(feature_cols)}

Subject-Wise Normalization Applied.

Grouped regression results (5-fold GroupKFold by Subject_ID):
Dummy RMSE: {dummy_rmse:.4f}
RF RMSE: {rf_rmse:.4f}
RF MAE: {rf_mae:.4f}
Dummy R2: {dummy_r2:.4f}
RF R2: {rf_r2:.4f}

Clustering summary (k=3):
{cluster_summary.to_string(index=False)}
"""
    
    with open(os.path.join(output_dir, 'summary.txt'), 'w') as f:
        f.write(summary_text)

    print(f"All outputs saved to {output_dir}")

if __name__ == '__main__':
    main()
