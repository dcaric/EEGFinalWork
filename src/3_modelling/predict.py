"""
Predict sleep hours (and category) for one new subject — thesis defense demo.

HOW TO USE:
  python predict.py <path_to_subject_csv> [--experiment exp1_fft_only]

The subject CSV needs one row with the same feature columns used during training.
Subject_ID and Sleep_Hours columns are ignored if present.

EXAMPLE:
  python predict.py subject_07.csv
  python predict.py subject_07.csv --experiment exp2_fft_gender

Run train_final_model.py first to create the model files.
"""

import argparse
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

DEFAULT_EXPERIMENT = 'exp1_fft_only'
BINS   = [0, 6, 8, 99]
LABELS = ['Short (<6h)', 'Normal (6-8h)', 'Long (>8h)']

BASE_DIR  = Path(__file__).parent.parent.parent
MODEL_DIR = BASE_DIR / 'models'


def load_models(experiment):
    reg_path  = MODEL_DIR / f'{experiment}_regression_model.pkl'
    clf_path  = MODEL_DIR / f'{experiment}_classification_model.pkl'
    feat_path = MODEL_DIR / f'{experiment}_features.txt'

    if not reg_path.exists():
        raise FileNotFoundError(
            f"Model not found: {reg_path}\n"
            f"Run train_final_model.py first with EXPERIMENT = '{experiment}'"
        )
    with open(reg_path, 'rb') as f:
        reg = pickle.load(f)
    with open(clf_path, 'rb') as f:
        clf = pickle.load(f)

    feature_cols = feat_path.read_text().strip().split('\n')
    return reg, clf, feature_cols


def predict(subject_csv, experiment):
    reg, clf, feature_cols = load_models(experiment)

    df = pd.read_csv(subject_csv)
    df = df.drop(columns=[c for c in ['Subject_ID', 'Sleep_Hours'] if c in df.columns])

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing {len(missing)} features in input CSV. First few: {missing[:3]}")

    X = df[feature_cols].values  # correct column order guaranteed

    pred_hours = reg.predict(X)[0]
    pred_cat   = clf.predict(X)[0]

    # Map category to friendly label
    if pred_cat == 'Short':
        label = 'Short sleeper (under 6h)'
    elif pred_cat == 'Long':
        label = 'Long sleeper (over 8h)'
    else:
        label = 'Normal sleeper (6–8h)'

    print(f"\n{'='*45}")
    print(f"  Experiment : {experiment}")
    print(f"  Input file : {subject_csv}")
    print(f"{'='*45}")
    print(f"  Predicted sleep hours : {pred_hours:.2f} h")
    print(f"  Predicted category    : {label}")
    print(f"{'='*45}\n")

    return pred_hours, pred_cat


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict sleep for one subject.')
    parser.add_argument('subject_csv', help='Path to subject CSV file')
    parser.add_argument('--experiment', default=DEFAULT_EXPERIMENT,
                        help=f'Experiment name (default: {DEFAULT_EXPERIMENT})')
    args = parser.parse_args()
    predict(args.subject_csv, args.experiment)
