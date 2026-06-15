import multiprocessing
from multiprocessing import Process, Queue
import pandas as pd
import numpy as np
import joblib
import time
import os
import glob
import json
import warnings
import sys
import gc

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
TEST_ROOT_PATH = '/home/pcnlab/Dataset/PCAP/test/'
PACKET_LIMIT_PER_FOLDER = 20000  
BATCH_SIZE = 500                # <--- The Speed Secret

# ==========================================
# 1. THE EXPERT WORKER (Batch Mode + Your Logging)
# ==========================================
def expert_worker(agent_name, task_queue, model_path, le_path, features_count):
    # Open files with 'buffering=1' so they write IMMEDIATELY
    pred_log = open("final_predictions.csv", "a", buffering=1) 
    action_log = open("incident_actions.json", "a", buffering=1) 
    aim_log = open("aim_retrain.csv", "a", buffering=1)       
    txt_alert_log = open("security_events.log", "a", buffering=1) 

    print(f"   -> 🟢 {agent_name} Online.")

    try:
        model = joblib.load(model_path)
        le = joblib.load(le_path)
    except:
        print(f"   -> ❌ {agent_name} Failed to load model.")
        return

    # --- DEFINING THE CORRECT FEATURES ---
    # These are the standard features your models were trained on
    STANDARD_FEATURES = ['duration', 'orig_bytes', 'resp_bytes', 'orig_pkts', 'resp_pkts', 
                         'id.resp_p', 'proto', 'service', 'conn_state']

    processed_count = 0

    while True:
        batch_payload = task_queue.get()
        if batch_payload == "STOP": break
        
        packets = batch_payload['data']
        true_label = batch_payload['label']
        
        try:
            # 1. Prepare DataFrame
            df = pd.DataFrame(packets)
            
            # 2. Clean Numbers (Force Strings to 0)
            num_cols = ['duration', 'orig_bytes', 'resp_bytes', 'orig_pkts', 'resp_pkts', 'id.resp_p']
            for c in num_cols:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

            # 3. Clean Categorical
            for c in ['proto', 'service', 'conn_state']:
                if c in df.columns:
                    df[c] = df[c].astype('category').cat.codes

            # --- CRITICAL FIX: SELECT SPECIFIC COLUMNS BY NAME ---
            # Do NOT just take the first N columns (which are strings like UID)
            
            target_cols = []
            
            # A. Try to ask the model what it wants
            if hasattr(model, 'feature_names_in_'):
                target_cols = list(model.feature_names_in_)
            elif hasattr(model, 'feature_name_'):
                target_cols = model.feature_name_
            else:
                # B. Fallback: Use standard list
                target_cols = list(STANDARD_FEATURES)
                if features_count == 8 and 'id.resp_p' in target_cols:
                    target_cols.remove('id.resp_p') # Volumetric usually drops Port

            # C. Ensure columns exist and fill missing with 0
            for col in target_cols:
                if col not in df.columns:
                    df[col] = 0
            
            # D. Create Final Matrix (Strictly ordered, strictly numeric)
            X = df[target_cols]

            # 4. Predict
            pred_idxs = model.predict(X)
            confidences = np.max(model.predict_proba(X), axis=1)
            pred_labels = le.inverse_transform(pred_idxs)

            # 5. Log Results
            for i in range(len(packets)):
                uid = packets[i].get('uid', 'N/A')
                src_ip = packets[i].get('id.orig_h', '0.0.0.0')
                verdict = pred_labels[i]
                conf = confidences[i]
                
                # Write Prediction
                pred_log.write(f"{uid},{true_label},{verdict},{agent_name}\n")
                
                # Incident Response
                if verdict != "Normal":
                    action_record = {
                        "timestamp": time.time(),
                        "uid": uid,
                        "src_ip": src_ip,
                        "threat": verdict,
                        "action": "BLOCK"
                    }
                    action_log.write(json.dumps(action_record) + "\n")
                    txt_alert_log.write(f"ALERT: {verdict} from {src_ip}\n")

                # Retrain Data 
                if conf < 0.60:
                    # Save features as string
                    feats = ",".join(map(str, X.iloc[i].values))
                    aim_log.write(f"{uid},{feats},{true_label},{conf:.4f}\n")
    
            processed_count += len(packets)

        except Exception as e:
            # Print the actual error so we know if it persists
            print(f"⚠️ {agent_name} Batch Error: {e}")
            pass 
    
    pred_log.close(); action_log.close(); aim_log.close(); txt_alert_log.close()
    print(f"   -> 🔴 {agent_name} Finished. Processed: {processed_count}")

# ==========================================
# 2. THE ORCHESTRATOR (Batch Dispatcher)
# ==========================================
class ParallelOrchestrator:
    def __init__(self):
        print("\n🔵 [SYSTEM STARTUP] Booting High-Speed Parallel Engine...")
        
        # Init Files
        with open("final_predictions.csv", "w") as f: f.write("uid,true_label,predicted_label,agent_name\n")
        with open("aim_retrain.csv", "w") as f: f.write("uid,features,true_label,confidence\n")
        with open("incident_actions.json", "w") as f: pass 
        with open("security_events.log", "w") as f: f.write("--- AGENTIC IDS EVENT LOG ---\n")

        # Queues
        self.q_mqtt = Queue(maxsize=100)
        self.q_vol = Queue(maxsize=100)
        self.q_recon = Queue(maxsize=100)
        
        # Processes
        self.p1 = Process(target=expert_worker, args=('MQTT_Expert', self.q_mqtt, 'mqtt_model.pkl', 'mqtt_le.pkl', 9))
        self.p2 = Process(target=expert_worker, args=('Volumetric_Expert', self.q_vol, 'volumetric_model.pkl', 'volumetric_le.pkl', 8))
        self.p3 = Process(target=expert_worker, args=('Recon_Expert', self.q_recon, 'recon_model.pkl', 'recon_le.pkl', 9))
        self.p1.start(); self.p2.start(); self.p3.start()
        
        print("   ├── 🛡️  Loading Supervisor...")
        self.supervisor = joblib.load('supervisor_model.pkl')
        self.supervisor_le = joblib.load('supervisor_le.pkl')
        print("🟢 [AGENTS DEPLOYED] Ready.\n")

    def preprocess_packet(self, packet):
        # Quick pre-calc for Supervisor
        data = packet.copy()
        sup_cols = ['proto', 'service', 'duration', 'orig_bytes', 'resp_bytes', 'conn_state', 'orig_pkts', 'resp_pkts', 'id.resp_p']
        
        df = pd.DataFrame([data])
        for c in sup_cols: 
             if c not in df.columns: df[c] = 0
             else: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
             
        for c in ['proto', 'service', 'conn_state']:
            if c in df.columns: df[c] = df[c].astype('category').cat.codes
            
        return df[sup_cols]

    def process_traffic(self, packet_stream, folder_label):
        # Local Buffers for Batching
        buf_mqtt, buf_vol, buf_recon = [], [], []
        count = 0
        
        for packet in packet_stream:
            count += 1
            try:
                # 1. Supervisor Decision
                df_sup = self.preprocess_packet(packet)
                agent_idx = self.supervisor.predict(df_sup)[0]
                target = self.supervisor_le.inverse_transform([agent_idx])[0]
                
                # 2. Add to Buffer (Not Queue yet)
                if target == 'AGENT_MQTT': buf_mqtt.append(packet)
                elif target == 'AGENT_VOLUMETRIC': buf_vol.append(packet)
                else: buf_recon.append(packet)

                # 3. Flush if Full
                if len(buf_mqtt) >= BATCH_SIZE:
                    self.q_mqtt.put({'data': buf_mqtt, 'label': folder_label})
                    buf_mqtt = []
                if len(buf_vol) >= BATCH_SIZE:
                    self.q_vol.put({'data': buf_vol, 'label': folder_label})
                    buf_vol = []
                if len(buf_recon) >= BATCH_SIZE:
                    self.q_recon.put({'data': buf_recon, 'label': folder_label})
                    buf_recon = []

            except: continue

        # 4. Flush Remaining
        if buf_mqtt: self.q_mqtt.put({'data': buf_mqtt, 'label': folder_label})
        if buf_vol: self.q_vol.put({'data': buf_vol, 'label': folder_label})
        if buf_recon: self.q_recon.put({'data': buf_recon, 'label': folder_label})
        
        return count

    def shutdown(self):
        print("\n🔴 Stopping Agents...")
        self.q_mqtt.put("STOP"); self.q_vol.put("STOP"); self.q_recon.put("STOP")
        self.p1.join(); self.p2.join(); self.p3.join()

# ==========================================
# 3. DATASET LOADER
# ==========================================
def parse_zeek_to_df(file_path, limit=None):
    try:
        with open(file_path, 'r', errors='ignore') as f:
            header = ""
            for line in f:
                if line.startswith('#fields'): header = line; break
            if not header: return None
            cols = header.strip().split('\t')[1:]
            return pd.read_csv(file_path, sep='\t', names=cols, comment='#', low_memory=False, nrows=limit)
    except: return None

if __name__ == "__main__":
    start_real = time.time()
    system = ParallelOrchestrator()
    
    print(f"🚀 STARTING FAST PARALLEL TEST (Limit: {PACKET_LIMIT_PER_FOLDER})")
    folders = sorted(os.listdir(TEST_ROOT_PATH))
    total_processed = 0

    for folder in folders:
        folder_path = os.path.join(TEST_ROOT_PATH, folder)
        if not os.path.isdir(folder_path): continue
        true_label = folder.replace('_test', '').replace('_train', '')
        
        conn_log = os.path.join(folder_path, 'conn.log')
        if not os.path.exists(conn_log): continue
        
        # 1. Load Conn Log (Limited)
        df_conn = parse_zeek_to_df(conn_log, limit=PACKET_LIMIT_PER_FOLDER)
        if df_conn is None or df_conn.empty: continue
        
        # 2. Merge MQTT (The Danger Zone)
        if true_label.startswith('MQTT'):
            mqtt_files = glob.glob(os.path.join(folder_path, 'mqtt*.log'))
            if mqtt_files:
                for mq in mqtt_files:
                    # OPTIMIZATION: Also limit MQTT reading to prevent RAM crash
                    # Read 2x the limit to ensure we find matches, but not millions
                    df_mq = parse_zeek_to_df(mq, limit=PACKET_LIMIT_PER_FOLDER * 5) 
                    
                    if df_mq is not None and 'uid' in df_mq.columns:
                        cols = [c for c in ['uid', 'topic', 'client_id'] if c in df_mq.columns]
                        df_conn = pd.merge(df_conn, df_mq[cols], on='uid', how='left')

        # 3. *** THE FIX: THE HARD CAP ***
        # If the merge caused an explosion (e.g., 5,000 -> 1,000,000), CUT IT NOW.
        if len(df_conn) > PACKET_LIMIT_PER_FOLDER:
            print(f"   ⚠️ Data Explosion Detected! Truncating {len(df_conn)} -> {PACKET_LIMIT_PER_FOLDER}")
            df_conn = df_conn.iloc[:PACKET_LIMIT_PER_FOLDER]

        # 4. Proceed as normal
        packets = df_conn.to_dict('records')
        count = system.process_traffic(packets, true_label)
        total_processed += count
        
        print(f"   -> {folder}: Pushed {count} pkts.")
        del df_conn; del packets; gc.collect()

    system.shutdown()
    
    end_real = time.time()
    real_time = end_real - start_real
    print(f"\n📊 FINAL REPORT")
    print(f"Total Packets: {total_processed}")
    print(f"Time Taken:    {end_real - start_real:.2f} sec")
    print(f"Throughput:      {total_processed/real_time:.2f} pkts/sec")
