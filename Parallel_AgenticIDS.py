import multiprocessing
from multiprocessing import Process, Queue
import pandas as pd
import numpy as np
import joblib
import time
import os
import warnings
import sys

warnings.filterwarnings("ignore")

# ==========================================
# 1. THE WORKER PROCESS (Runs on separate CPU Core)
# ==========================================
def expert_worker(agent_name, task_queue, model_path, le_path, features_count):
    # Setup File Logging for this core
    log_file = open("system_debug.log", "a")
    
    # Load Model
    try:
        model = joblib.load(model_path)
        le = joblib.load(le_path)
    except:
        return

    while True:
        packet_data = task_queue.get()
        if packet_data == "STOP": break

        try:
            # 1. Preprocess: Force Types (THE FIX)
            df = pd.DataFrame([packet_data])
            
            # Numeric conversion (Fixes "object" error)
            num_cols = ['duration', 'orig_bytes', 'resp_bytes', 'orig_pkts', 'resp_pkts', 'id.resp_p', 'id.orig_p']
            for c in num_cols:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

            # Categorical conversion
            cat_cols = ['proto', 'service', 'conn_state']
            for c in cat_cols:
                if c in df.columns:
                    df[c] = df[c].astype('category').cat.codes

            # Drop Port if Volumetric Agent (Feature mismatch fix)
            if features_count == 8 and 'id.resp_p' in df.columns:
                df = df.drop(columns=['id.resp_p'])

            # 2. Predict
            # Ensure we only pass columns the model knows
            # (LightGBM is picky about extra columns)
            # We assume the model object has feature_name_
            if hasattr(model, 'feature_name_'):
                valid_feats = model.feature_name_
                # Create missing cols with 0
                for f in valid_feats:
                    if f not in df.columns: df[f] = 0
                df = df[valid_feats]

            pred_idx = model.predict(df)[0]
            verdict = le.inverse_transform([pred_idx])[0]
            
            # 3. Log to File (NOT Terminal)
            flow_id = packet_data.get('uid', 'unknown')
            log_message = f"[{agent_name}] Flow: {flow_id} -> Verdict: {verdict}\n"
            log_file.write(log_message)
            log_file.flush()

        except Exception as e:
            # Log errors silently to file
            log_file.write(f"[{agent_name} ERROR] {str(e)}\n")
            log_file.flush()
    
    log_file.close()

# ==========================================
# 2. THE MAIN DISTRIBUTOR
# ==========================================
class ParallelOrchestrator:
    def __init__(self):
        print("\n🔵 [SYSTEM STARTUP] Booting Parallel Engine...")
        # Clear previous log
        with open("system_debug.log", "w") as f:
            f.write("--- AGENTIC IDS SYSTEM LOG ---\n")

        self.q_mqtt = Queue()
        self.q_volumetric = Queue()
        self.q_recon = Queue()

        self.p_mqtt = Process(target=expert_worker, args=('MQTT_Expert', self.q_mqtt, 'mqtt_model.pkl', 'mqtt_le.pkl', 9))
        self.p_vol = Process(target=expert_worker, args=('Volumetric_Expert', self.q_volumetric, 'volumetric_model.pkl', 'volumetric_le.pkl', 8))
        self.p_recon = Process(target=expert_worker, args=('Recon_Expert', self.q_recon, 'recon_model.pkl', 'recon_le.pkl', 9))

        self.p_mqtt.start()
        self.p_vol.start()
        self.p_recon.start()
        
        print("   ├── 🛡️  Loading Supervisor...")
        self.supervisor = joblib.load('supervisor_model.pkl')
        self.supervisor_le = joblib.load('supervisor_le.pkl')
        print("🟢 [DISTRIBUTOR ONLINE] System Ready.\n")
        
        # Statistics
        self.stats = {'Total': 0, 'MQTT': 0, 'Volumetric': 0, 'Recon': 0}

    def preprocess_supervisor(self, packet):
        # Quick robust preprocessing
        data = packet.copy()
        
        # Force Numeric Types (Fixes crash)
        for k in ['duration', 'orig_bytes', 'resp_bytes', 'orig_pkts', 'resp_pkts', 'id.resp_p']:
            if k in data:
                try:
                    data[k] = float(data[k])
                except:
                    data[k] = 0.0

        sup_cols = ['proto', 'service', 'duration', 'orig_bytes', 'resp_bytes', 'conn_state', 'orig_pkts', 'resp_pkts', 'id.resp_p']
        
        df = pd.DataFrame([data])
        for c in sup_cols: 
            if c not in df.columns: df[c] = 0
            
        # Cat codes
        for c in ['proto', 'service', 'conn_state']:
            if c in df.columns:
                df[c] = df[c].astype('category').cat.codes
        
        return df[sup_cols]

    def process_traffic(self, packet_stream):
        print(f"🌊 [STREAMING] Processing batch of {len(packet_stream)} packets...")
        print("-" * 60)
        print(f"{'PROCESSED':<15} | {'ROUTING STATS':<40}")
        print("-" * 60)

        for i, packet in enumerate(packet_stream):
            self.stats['Total'] += 1
            
            # Supervisor Decision
            try:
                df_sup = self.preprocess_supervisor(packet)
                agent_idx = self.supervisor.predict(df_sup)[0]
                target_agent = self.supervisor_le.inverse_transform([agent_idx])[0]
            except:
                target_agent = 'AGENT_VOLUMETRIC' # Fallback

            # Async Dispatch
            if target_agent == 'AGENT_MQTT':
                self.q_mqtt.put(packet)
                self.stats['MQTT'] += 1
            elif target_agent == 'AGENT_VOLUMETRIC':
                self.q_volumetric.put(packet)
                self.stats['Volumetric'] += 1
            elif target_agent == 'AGENT_RECON':
                self.q_recon.put(packet)
                self.stats['Recon'] += 1

            # Progress Bar (Update every 1000 packets)
            if i % 1000 == 0 and i > 0:
                sys.stdout.write(f"\r[{i} Pkts] -> MQTT: {self.stats['MQTT']} | Vol: {self.stats['Volumetric']} | Recon: {self.stats['Recon']}")
                sys.stdout.flush()

        print(f"\n\n✅ BATCH COMPLETE. Total Processed: {self.stats['Total']}")
        print(f"📄 Detailed logs saved to 'system_debug.log'")

    def shutdown(self):
        print("\n🔴 Stopping Cores...")
        self.q_mqtt.put("STOP")
        self.q_volumetric.put("STOP")
        self.q_recon.put("STOP")
        self.p_mqtt.join()
        self.p_vol.join()
        self.p_recon.join()
