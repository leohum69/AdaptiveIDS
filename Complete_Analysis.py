import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

# --- CONFIGURATION ---
files = {
    "1. Single Agent": "/home/pcnlab/Models/Baseline/baseline_predictions.csv",
    "2. Sequential": "/home/pcnlab/Models/Baseline/sequential_predictions.csv",
    "3. Parallel (Ours)": "final_predictions.csv"
}

# --- THE BROAD MAPPER ---
def get_broad_category(label):
    s = str(label).lower().strip()
    
    # 1. BENIGN
    if 'benign' in s or 'normal' in s:
        return 'Benign'
        
    # 2. MQTT FAMILY
    if 'mqtt' in s:
        return 'MQTT_Attack'
        
    # 3. VOLUMETRIC FAMILY (TCP/UDP/ICMP/DoS/DDoS)
    if any(x in s for x in ['tcp', 'udp', 'icmp', 'dos', 'ddos', 'volumetric']):
        return 'Volumetric_Attack'
        
    # 4. RECON FAMILY (Scan/Ping)
    if 'recon' in s or 'scan' in s or 'ping' in s:
        return 'Recon_Attack'
        
    # 5. SPOOFING
    if 'arp' in s or 'spoof' in s:
        return 'Spoofing_Attack'
        
    return 'Other'

results = {}

print("🔵 STARTING BROAD CATEGORY ANALYSIS (Thesis Level 1)...")

for system_name, filename in files.items():
    if not os.path.exists(filename):
        continue
        
    try:
        df = pd.read_csv(filename)
        df = df[df['true_label'] != 'true_label'] # Clean headers
        
        # Apply Broad Mapping
        y_true = df['true_label'].apply(get_broad_category)
        y_pred = df['predicted_label'].apply(get_broad_category)
        
        # Calculate Accuracy
        acc = accuracy_score(y_true, y_pred)
        results[system_name] = acc * 100
        print(f"\n📊 {system_name}: {acc*100:.2f}% (Broad Accuracy)")
        
        # Save Broad Confusion Matrix
        plt.figure(figsize=(8, 6))
        labels = sorted(list(set(y_true) | set(y_pred)))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=labels, yticklabels=labels)
        plt.title(f"{system_name}\nBroad Accuracy: {acc*100:.1f}%")
        plt.tight_layout()
        plt.savefig(f"CM_Broad_{system_name.split()[1]}.png")
        
    except Exception as e:
        print(f"Error: {e}")

# --- FINAL BAR CHART ---
if results:
    plt.figure(figsize=(10, 6))
    bars = plt.bar(results.keys(), results.values(), color=['gray', 'orange', 'green'])
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                 f"{bar.get_height():.1f}%", ha='center', fontweight='bold')
    plt.ylim(0, 105)
    plt.title("System Accuracy by Attack Family (Broad Classification)")
    plt.ylabel("Accuracy (%)")
    plt.savefig("Final_Thesis_Broad_Chart.png")
    print("\n✅ SAVED: 'Final_Thesis_Broad_Chart.png'")
