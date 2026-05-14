import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def main():
    file_path = '/Users/dcaric/Working/pythonWorking/PaolaFinalWork/pilot_files/all_task_ready.csv'
    df = pd.read_csv(file_path)
    
    # Filter only numeric columns for similarity calculation
    # Exclude non-feature columns
    exclude_cols = ['Subject_ID', 'Task', 'Sleep_Hours', 'Gender']
    feature_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
    
    # Group the data
    group1 = df[df['Sleep_Hours'] <= 6][feature_cols]
    group2 = df[df['Sleep_Hours'] > 6][feature_cols]
    
    print(f"Group 1 (Sleep_Hours <= 6): {len(group1)} rows")
    print(f"Group 2 (Sleep_Hours > 6): {len(group2)} rows")
    
    # Method 1: Variance
    # Lower variance indicates points are closer to each other (more similar)
    # We use Coefficient of Variation (std / mean) to account for different scales
    cv_group1 = (group1.std() / group1.mean().replace(0, np.nan)).abs().mean()
    cv_group2 = (group2.std() / group2.mean().replace(0, np.nan)).abs().mean()
    
    print("\n--- Similarity via Coefficient of Variation (lower means more similar) ---")
    print(f"Group 1 Mean CV: {cv_group1:.4f}")
    print(f"Group 2 Mean CV: {cv_group2:.4f}")
    
    # Method 2: Mean Pairwise Cosine Similarity
    # Higher value indicates more similar (closer to 1.0)
    # Since cosine_similarity is O(n^2), if rows are too many, we take a random sample
    def get_mean_cosine_sim(data, sample_size=1000):
        if len(data) > sample_size:
            data = data.sample(sample_size, random_state=42)
        # Normalize data (cosine similarity is sensitive to scale if not centered, but it measures angle)
        # Handle NaN if any
        data = data.fillna(0)
        sim_matrix = cosine_similarity(data)
        # Extract upper triangle without diagonal
        iu1 = np.triu_indices(sim_matrix.shape[0], k=1)
        return sim_matrix[iu1].mean()

    mean_sim_1 = get_mean_cosine_sim(group1)
    mean_sim_2 = get_mean_cosine_sim(group2)
    
    print("\n--- Similarity via Mean Pairwise Cosine Similarity (higher means more similar) ---")
    print(f"Group 1 Mean Cosine Similarity: {mean_sim_1:.4f}")
    print(f"Group 2 Mean Cosine Similarity: {mean_sim_2:.4f}")

if __name__ == '__main__':
    main()
