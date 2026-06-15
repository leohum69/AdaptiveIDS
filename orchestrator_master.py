# orchestrator_master.py
import asyncio, pandas as pd, numpy as np, matplotlib, joblib, logging, sys, pickle, time, os, torch, torch.nn as nn
matplotlib.use('Agg')
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report, accuracy_score, f1_score
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import warnings
warnings.filterwarnings("ignore")

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Linear(64, 32), nn.ReLU(), nn.BatchNorm1d(32), nn.Linear(32, encoding_dim))
        self.decoder = nn.Sequential(nn.Linear(encoding_dim, 32), nn.ReLU(), nn.BatchNorm1d(32), nn.Linear(32, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Linear(64, input_dim), nn.Sigmoid())
    def forward(self, x): return self.decoder(self.encoder(x))

class PrintLogger:
    def __init__(self, filename="master_evaluation_output.txt"):
        self.terminal = sys.stdout
        self.logfile = open(filename, "w")
    def write(self, message): self.terminal.write(message); self.logfile.write(message)
    def flush(self): self.terminal.flush(); self.logfile.flush()

sys.stdout = PrintLogger()
logging.basicConfig(filename='master_system_alerts.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def evaluate_dataset(dataset_name: str, attack_csv: str, normal_csv: str, rows_per_class: int = 5000):
    start_time = time.time()
    print(f"\n{'='*60}\n[INIT] Chronological Real-World Stream: {dataset_name}\n{'='*60}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        df_attack = pd.read_csv(attack_csv, low_memory=False, nrows=rows_per_class)
        df_normal = pd.read_csv(normal_csv, low_memory=False, nrows=rows_per_class)
    except FileNotFoundError as e:
        print(f"\n[CRITICAL ERROR] Missing CSV: {e}"); return None
    
    df_attack['binary_label'] = 1; df_normal['binary_label'] = 0
    # FIX: NO SHUFFLE. Concat chronologically to simulate real bursts.
    df_full = pd.concat([df_normal, df_attack]).copy() 
    y_true = df_full['binary_label'].values
    
    with open('ae_model_params.pkl', 'rb') as f: loaded_params = pickle.load(f)
    with open('ae_scaler.pkl', 'rb') as f: loaded_scaler_ae = pickle.load(f)
    loaded_autoencoder = Autoencoder(loaded_params['input_dim'], loaded_params['encoding_dim']).to(device)
    loaded_autoencoder.load_state_dict(torch.load('autoencoder_final.pt', map_location=device))
    loaded_autoencoder.eval()
    
    scaler_agents = joblib.load("scaler_agents.pkl") 
    expected_features = scaler_agents.feature_names_in_ 
    features_df = df_full.select_dtypes(include=[np.number])
    if 'binary_label' in features_df.columns: features_df = features_df.drop(columns=['binary_label'])
    features_df = features_df.iloc[:, :len(expected_features)].copy()
    features_df.columns = expected_features
    features_df = features_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    raw_array = features_df.values
    
    y_probs_fused, y_preds_fused = [], []
    params_mqtt = StdioServerParameters(command="python3", args=["mqtt_agent.py"])
    params_vol = StdioServerParameters(command="python3", args=["volumetric_agent.py"])
    params_recon = StdioServerParameters(command="python3", args=["reconnaissance_agent.py"])

    async with stdio_client(params_mqtt) as (r1, w1), stdio_client(params_vol) as (r2, w2), stdio_client(params_recon) as (r3, w3):
        async with ClientSession(r1, w1) as sess_mqtt, ClientSession(r2, w2) as sess_vol, ClientSession(r3, w3) as sess_recon:
            await sess_mqtt.initialize(); await sess_vol.initialize(); await sess_recon.initialize()

            for idx, raw_features in enumerate(raw_array):
                true_label = y_true[idx]
                t_m = await sess_mqtt.call_tool("get_trust_score", arguments={}); curr_tr1 = float(t_m.content[0].text)
                t_v = await sess_vol.call_tool("get_trust_score", arguments={}); curr_tr2 = float(t_v.content[0].text)
                t_r = await sess_recon.call_tool("get_trust_score", arguments={}); curr_tr3 = float(t_r.content[0].text)

                row_df = pd.DataFrame(raw_features.reshape(1, -1), columns=expected_features)
                new_data_scaled = loaded_scaler_ae.transform(row_df)
                new_data_tensor = torch.tensor(new_data_scaled, dtype=torch.float32).to(device)
                
                with torch.no_grad(): reconstructed = loaded_autoencoder(new_data_tensor).cpu().numpy()
                mse = np.mean(np.square(new_data_scaled - reconstructed), axis=1)[0]
                
                if mse <= loaded_params['threshold_95']:
                    y_probs_fused.append(0.0); y_preds_fused.append(0); continue

                flow_features = raw_features.tolist()
                task_m = sess_mqtt.call_tool("evaluate_threat", arguments={"features": flow_features})
                task_v = sess_vol.call_tool("evaluate_threat", arguments={"features": flow_features})
                task_r = sess_recon.call_tool("evaluate_threat", arguments={"features": flow_features})
                res_m, res_v, res_r = await asyncio.gather(task_m, task_v, task_r)

                p1, p2, p3 = float(res_m.content[0].text), float(res_v.content[0].text), float(res_r.content[0].text)
                fused_prob = ((curr_tr1 * p1) + (curr_tr2 * p2) + (curr_tr3 * p3)) / (curr_tr1 + curr_tr2 + curr_tr3 + 1e-9)
                y_probs_fused.append(fused_prob); y_preds_fused.append(1 if fused_prob >= 0.5 else 0)

                await sess_mqtt.call_tool("update_trust", arguments={"reward": 1 if (1 if p1 >= 0.5 else 0) == true_label else 0})
                await sess_vol.call_tool("update_trust", arguments={"reward": 1 if (1 if p2 >= 0.5 else 0) == true_label else 0})
                await sess_recon.call_tool("update_trust", arguments={"reward": 1 if (1 if p3 >= 0.5 else 0) == true_label else 0})

                pred = 1 if fused_prob >= 0.5 else 0
                if pred != true_label:
                    features_str = ",".join(map(str, flow_features))
                    with open("master_retrain_queue.csv", "a") as f:
                        f.write(f"{dataset_name},{idx},{true_label},{fused_prob},{features_str}\n")

    acc = accuracy_score(y_true, y_preds_fused)
    print(f"[RESULTS] {dataset_name} Complete | Accuracy: {acc*100:.2f}% | Exec Time: {time.time() - start_time:.2f}s")
    return {"Dataset": dataset_name, "Accuracy": f"{acc*100:.2f}%"}

if __name__ == "__main__":
    if os.path.exists("master_retrain_queue.csv"): os.remove("master_retrain_queue.csv")
    datasets = {
        "CIC_IoMT_2024": {"attack": "CICIoMT2024_labelled_CICFlow.csv", "normal": "Benign_test.pcap.csv"},
        "BOT_IOT": {"attack": "BOT_IOT_DDOS.csv", "normal": "botiot_ben.csv"},
        "TON_IOT": {"attack": "TON_IOT_MAL.csv", "normal": "TON_IOT_Benig.csv"},
        "CIC_DDoS": {"attack": "cicddos_mal.csv", "normal": "cicddos_benign.csv"},
        "CIC_IDS": {"attack": "CICIDS_Malicious.csv", "normal": "CICIDS_Benign.csv"}
    }
    results_summary = []
    for ds_name, paths in datasets.items():
        res = asyncio.run(evaluate_dataset(ds_name, paths["attack"], paths["normal"], 2500)) # Quick 5k rows per dataset
        if res: results_summary.append(res)
    if results_summary:
        pd.DataFrame(results_summary).to_csv("Master_Evaluation_Table.csv", index=False)
