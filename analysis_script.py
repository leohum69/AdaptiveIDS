import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the Results
print("Loading Final Predictions...")
df = pd.read_csv("final_predictions.csv")

# 2. Clean the Labels (Map them to simple categories)
# This fixes the "Other_Activity" vs "Attack" confusion for the report
def clean_label(label):
    if "Benign" in label: return "Benign"
    return "Attack"

df['True_Class'] = df['true_label'].apply(clean_label)
df['Pred_Class'] = df['predicted_label'].apply(clean_label)

# 3. Calculate Metrics
accuracy = accuracy_score(df['True_Class'], df['Pred_Class'])
print(f"\n✅ SYSTEM OVERALL ACCURACY: {accuracy*100:.2f}%")

print("\n--- DETAILED CLASSIFICATION REPORT ---")
print(classification_report(df['True_Class'], df['Pred_Class']))

# 4. Generate Confusion Matrix Plot
cm = confusion_matrix(df['True_Class'], df['Pred_Class'])
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Attack', 'Benign'], yticklabels=['Attack', 'Benign'])
plt.title('Agentic IDS Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('Final_Confusion_Matrix.png')
print("\n📸 Saved 'Final_Confusion_Matrix.png'")
