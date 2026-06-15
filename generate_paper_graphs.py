# generate_paper_graphs.py
import pandas as pd, numpy as np, joblib, gc, os
import lightgbm as lgb, xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_curve, auc, precision_recall_curve, 
                             average_precision_score)
import warnings; warnings.filterwarnings("ignore")

def compute_global_bayesian_fusion(y_true, p_mqtt, p_vol, p_recon):
    """
    Continuous Probability Bayesian Logic running across the stitched global stream.
    """
    trusts = {'MQTT': 0.34, 'Volumetric': 0.33, 'Recon': 0.33}
    fused_probs = np.zeros(len(y_true))
    history = {'MQTT': [], 'Volumetric': [], 'Recon': []}
    
    alpha = 0.05 # Learning rate for trust shifts
    
    for i in range(len(y_true)):
        history['MQTT'].append(trusts['MQTT'])
        history['Volumetric'].append(trusts['Volumetric'])
        history['Recon'].append(trusts['Recon'])
        
        pm, pv, pr = p_mqtt[i], p_vol[i], p_recon[i]
        
        # 1. Exact Bayesian Fusion
        p_attack = (trusts['MQTT'] * pm) + (trusts['Volumetric'] * pv) + (trusts['Recon'] * pr)
        p_benign = (trusts['MQTT'] * (1 - pm)) + (trusts['Volumetric'] * (1 - pv)) + (trusts['Recon'] * (1 - pr))
        final_prob = p_attack / (p_attack + p_benign + 1e-9)
        fused_probs[i] = final_prob
        
        # 2. Continuous Probability Trust Update
        true_label = y_true[i]
        score_m = (1.0 - abs(true_label - pm)) - 0.5
        score_v = (1.0 - abs(true_label - pv)) - 0.5
        score_r = (1.0 - abs(true_label - pr)) - 0.5
        
        trusts['MQTT'] += alpha * score_m
        trusts['Volumetric'] += alpha * score_v
        trusts['Recon'] += alpha * score_r
            
        # 3. Normalize Trust Scores
        for k in trusts: trusts[k] = max(0.01, trusts[k])
        total_trust = sum(trusts.values())
        for k in trusts: trusts[k] /= total_trust
            
    return fused_probs, history

def plot_master_trust_graph(history, boundaries, dataset_names):
    """Generates the massive timeline graph showing adaptability."""
    plt.figure(figsize=(15, 6))
    sns.set_theme(style="whitegrid", rc={"axes.edgecolor": "black"})
    
    flows = range(len(history['MQTT']))
    plt.plot(flows, history['MQTT'], label='MQTT Expert Trust', color='#1f77b4', alpha=0.85, linewidth=1.5)
    plt.plot(flows, history['Volumetric'], label='Volumetric Expert Trust', color='#ff7f0e', alpha=0.85, linewidth=1.5)
    plt.plot(flows, history['Recon'], label='Recon Expert Trust', color='#2ca02c', alpha=0.85, linewidth=1.5)
    
    # Draw Vertical Lines for Phase Changes
    prev_b = 0
    for i, b in enumerate(boundaries):
        plt.axvline(x=b, color='black', linestyle='--', alpha=0.6)
        midpoint = prev_b + (b - prev_b) / 2
        plt.text(midpoint, 1.02, f"{dataset_names[i]}\nPhase", ha='center', va='bottom', fontsize=10, fontweight='bold')
        prev_b = b

    plt.xlabel('Global Timeline (Flows Processed)', fontsize=12, fontweight='bold')
    plt.ylabel('Dynamic Trust Weight', fontsize=12, fontweight='bold')
    plt.title('MA-IDS Cross-Domain Robustness: Real-Time Trust Calibration', fontsize=16, fontweight='bold', pad=30)
    plt.legend(loc='lower left')
    plt.ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig("MASTER_Global_Trust_Evolution.png", dpi=300)
    plt.close()

def run_graph_generation():
    print("="*70); print("🚀 INITIATING GLOBAL STREAM GRAPH GENERATION (MEMORY SAFE)"); print("="*70)
    
    try:
        mqtt_model = joblib.load("xgboost_mqtt_intrusion_model.pkl")
        vol_model = lgb.Booster(model_file="lightgbm_global_model.txt")
        recon_model = joblib.load("rf_model.pkl")
        scaler = joblib.load("scaler_agents.pkl"); features = scaler.feature_names_in_
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    datasets = {
        "CIC_IoMT_2024": {"attack": "CICIoMT2024_labelled_CICFlow.csv", "normal": "Benign_test.pcap.csv"},
        "CIC_DDoS": {"attack": "cicddos_mal.csv", "normal": "cicddos_benign.csv"},
        "TON_IOT": {"attack": "TON_IOT_MAL.csv", "normal": "TON_IOT_Benig.csv"},
        "BOT_IOT": {"attack": "BOT_IOT_DDOS.csv", "normal": "botiot_ben.csv"},
        "CIC_IDS": {"attack": "CICIDS_Malicious.csv", "normal": "CICIDS_Benign.csv"}
    }
    
    global_y = []; global_pm = []; global_pv = []; global_pr = []
    boundaries = []; dataset_names = []; current_index = 0
    all_metrics = []

    # 1. READ ALL DATA AND PREDICT IN CHUNKS
    for ds_name, paths in datasets.items():
        print(f"Processing Domain: {ds_name}...")
        try:
            # Memory Safe loading: 3000 rows each -> 6000 rows per dataset
            df_a = pd.read_csv(paths["attack"], low_memory=False, nrows=3000)
            df_n = pd.read_csv(paths["normal"], low_memory=False, nrows=3000)
        except Exception as e: 
            print(f"Skipping {ds_name}: {e}"); continue
            
        df_a['binary_label'] = 1; df_n['binary_label'] = 0
        test_df = pd.concat([df_n, df_a]).sample(frac=1, random_state=42).reset_index(drop=True)
        y_test = test_df['binary_label'].values
        
        f_df = test_df.select_dtypes(include=[np.number])
        if 'binary_label' in f_df.columns: f_df = f_df.drop(columns=['binary_label'])
        f_df = f_df.iloc[:, :len(features)].copy()
        f_df.columns = features
        X_all = f_df.replace([np.inf, -np.inf], np.nan).fillna(0)
        X_xgb = X_all.drop(columns=['src_port', 'dst_port'], errors='ignore')
        X_rf_scaled = scaler.transform(X_all)

        pm = mqtt_model.predict_proba(X_xgb)[:, 1]
        pv = vol_model.predict(X_xgb)
        pr = recon_model.predict_proba(X_rf_scaled)[:, 1]

        global_y.extend(y_test)
        global_pm.extend(pm); global_pv.extend(pv); global_pr.extend(pr)
        
        current_index += len(y_test)
        boundaries.append(current_index)
        dataset_names.append(ds_name)
        
        # Clear memory
        del df_a, df_n, test_df, f_df, X_all, X_xgb, X_rf_scaled; gc.collect()

    print("\n🔄 Running Continuous Global Trust Update...")
    fused_probs, trust_history = compute_global_bayesian_fusion(
        np.array(global_y), np.array(global_pm), np.array(global_pv), np.array(global_pr)
    )
    
    print("📈 Plotting Master Trust Graph...")
    plot_master_trust_graph(trust_history, boundaries, dataset_names)

    # 2. GENERATE INDIVIDUAL GRAPHS PER DATASET (Using sliced global data)
    start_idx = 0
    modern_colors = ['#95a5a6', '#34495e', '#2c3e50', '#d73027'] 
    metrics_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
    
    for i, ds_name in enumerate(dataset_names):
        end_idx = boundaries[i]
        
        y_slice = np.array(global_y[start_idx:end_idx])
        pm_slice = np.array(global_pm[start_idx:end_idx])
        pv_slice = np.array(global_pv[start_idx:end_idx])
        pr_slice = np.array(global_pr[start_idx:end_idx])
        fused_slice = fused_probs[start_idx:end_idx]
        
        agents_data = {
            "MQTT Expert": (pm_slice >= 0.5).astype(int), 
            "Volumetric Expert": (pv_slice >= 0.5).astype(int), 
            "Reconnaissance Expert": (pr_slice >= 0.5).astype(int), 
            "Proposed MA-IDS": (fused_slice >= 0.5).astype(int)
        }
        
        plot_data = {"Accuracy": []}
        
        for agent_name, y_pred in agents_data.items():
            acc = accuracy_score(y_slice, y_pred); prec = precision_score(y_slice, y_pred, zero_division=0)
            rec = recall_score(y_slice, y_pred, zero_division=0); f1 = f1_score(y_slice, y_pred, zero_division=0)
            plot_data["Accuracy"].extend([acc, prec, rec, f1])
            all_metrics.append({"Dataset": ds_name, "Model": agent_name, "Accuracy": round(acc, 4), "Precision": round(prec, 4), "Recall": round(rec, 4), "F1-Score": round(f1, 4)})

        # -- Bar Chart --
        x = np.arange(4); width = 0.2  
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.set_theme(style="whitegrid", rc={"axes.edgecolor": "black"})
        
        for j, agent_name in enumerate(agents_data.keys()):
            scores = plot_data["Accuracy"][j*4:(j+1)*4]
            rects = ax.bar(x + (j - 1.5)*width, scores, width, label=agent_name, color=modern_colors[j], edgecolor='none', alpha=(0.85 if j<3 else 1.0))
            if j == 3:
                for rect in rects: ax.text(rect.get_x() + rect.get_width()/2., rect.get_height() + 0.015, f'{rect.get_height():.2f}', ha='center', va='bottom', fontweight='bold', color='#8b0000', fontsize=11)

        ax.set_ylabel('Performance Score', fontsize=12, fontweight='bold'); ax.set_title(f'Agent Performance Comparison - {ds_name}', fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x); ax.set_xticklabels(metrics_labels, fontsize=12, fontweight='bold')
        ax.set_ylim(0.0, 1.15); ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, frameon=False, fontsize=11)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        plt.tight_layout(); plt.savefig(f"Performance_Bar_{ds_name}.png", dpi=300, bbox_inches='tight'); plt.close()

        # -- ROC Curve --
        fpr, tpr, _ = roc_curve(y_slice, fused_slice); roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6)); sns.set_theme(style="whitegrid", rc={"axes.edgecolor": "black"})
        plt.plot(fpr, tpr, color='#ff7f0e', lw=2, label=f'MA-IDS (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate', fontweight='bold'); plt.ylabel('True Positive Rate', fontweight='bold')
        plt.title(f'Receiver Operating Characteristic (ROC) - {ds_name}', fontweight='bold')
        plt.legend(loc="lower right"); plt.tight_layout(); plt.savefig(f"ROC_{ds_name}.png", dpi=300); plt.close()

        # -- PR Curve --
        precision, recall, _ = precision_recall_curve(y_slice, fused_slice); avg_precision = average_precision_score(y_slice, fused_slice)
        plt.figure(figsize=(8, 6)); sns.set_theme(style="whitegrid", rc={"axes.edgecolor": "black"})
        plt.plot(recall, precision, color='purple', lw=2, label=f'MA-IDS (Avg Precision = {avg_precision:.3f})')
        plt.xlabel('Recall', fontweight='bold'); plt.ylabel('Precision', fontweight='bold')
        plt.title(f'Precision-Recall Curve - {ds_name}', fontweight='bold')
        plt.legend(loc="lower left"); plt.tight_layout(); plt.savefig(f"PR_Curve_{ds_name}.png", dpi=300); plt.close()

	# --- 4. CONFUSION MATRIX (Modernized) ---
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_slice, (fused_slice >= 0.5).astype(int))
        
        plt.figure(figsize=(7, 6))
        # Using a sleek blue color map instead of legacy colors
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                    annot_kws={"size": 14, "weight": "bold"},
                    xticklabels=['Benign', 'Attack'], yticklabels=['Benign', 'Attack'])
        
        plt.xlabel('Predicted Threat', fontsize=12, fontweight='bold')
        plt.ylabel('Actual Threat', fontsize=12, fontweight='bold')
        plt.title(f'Threat Classification Matrix - {ds_name}', fontsize=14, fontweight='bold', pad=15)
        
        plt.tight_layout()
        plt.savefig(f"Confusion_Matrix_{ds_name}.png", dpi=300)
        plt.close()

        start_idx = end_idx

    pd.DataFrame(all_metrics).to_csv("Paper_Metrics_Table.csv", index=False)
    print("\n✅ ALL ASSETS (Master Trust Graph, Per-Domain ROC/PR/Bars) GENERATED SUCCESSFULLY.")

if __name__ == "__main__": 
    run_graph_generation()
