import pandas as pd

# Load your parallel results (since it has the most data)
df = pd.read_csv("final_predictions.csv")

print("--- DIAGNOSTIC REPORT ---")
print(f"Total Rows: {len(df)}")

print("\n1. UNIQUE TRUE LABELS (From Folders):")
print(df['true_label'].unique())

print("\n2. UNIQUE PREDICTED LABELS (From AI):")
print(df['predicted_label'].unique())

print("\n3. SAMPLE MISMATCHES (First 10 rows):")
print(df[['true_label', 'predicted_label']].head(10))
