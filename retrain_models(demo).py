# retrain_models.py
# ─────────────────────────────────────────────────────────────────────────────
# Reads master_retrain_queue.csv, incrementally retrains LightGBM, XGBoost, 
# and Random Forest models. Saves them back to models/ and prints a clear 
# before/after accuracy comparison — perfect for a recorded demo.
#
# Run:  python retrain_models.py
# ─────────────────────────────────────────────────────────────────────────────

import os, sys, shutil, time
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

MODELS_DIR  = "models"
MODELS_BAK  = "models_backup"
QUEUE_FILE  = "master_retrain_queue.csv"
FEATURE_COL_START = 4   # columns 0-3 are: dataset, flow_idx, true_label, fused_prob

# ── Column count in the queue (79 features + 4 meta) ──────────
TOTAL_COLS = 83

def banner(msg):
    print(f"\n{'─'*60}")
    print(f"  {msg}")
    print(f"{'─'*60}")

def load_queue():
    banner("Loading misclassification queue…")
    if not os.path.exists(QUEUE_FILE):
        print(f"[ERROR] {QUEUE_FILE} not found. Run the orchestrator first.")
        sys.exit(1)

    rows = []
    with open(QUEUE_FILE, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= TOTAL_COLS:
                rows.append(parts)

    if not rows:
        print("[INFO] Queue is empty — nothing to retrain on.")
        sys.exit(0)

    meta    = [(r[0], int(r[1]), int(r[2]), float(r[3])) for r in rows]
    X_raw   = np.array([[float(v) for v in r[FEATURE_COL_START:FEATURE_COL_START+79]] for r in rows])
    y_raw   = np.array([int(r[2]) for r in rows])

    print(f"  Loaded {len(rows)} misclassified flows")
    print(f"  Label distribution — Attack: {y_raw.sum()}  |  Benign: {(y_raw==0).sum()}")
    return X_raw, y_raw, meta

def backup_models():
    banner("Backing up current models to models_backup/…")
    os.makedirs(MODELS_BAK, exist_ok=True)
    
    model_files = [
        "lightgbm_global_model.txt", 
        "xgboost_mqtt_intrusion_model.pkl",
        "rf_model.pkl" 
    ]
    
    for fname in model_files:
        src = os.path.join(MODELS_DIR, fname)
        dst = os.path.join(MODELS_BAK, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  ✓ Backed up {fname}")
        else:
            print(f"  ⚠  {fname} not found in {MODELS_DIR} — skipping backup")

def evaluate_lgbm(model, X, y):
    preds = (model.predict(X) >= 0.5).astype(int)
    return accuracy_score(y, preds) * 100

def evaluate_xgb(model, X, y):
    if hasattr(model, 'get_booster'):
        preds = model.predict(X)
    else:
        dmat  = xgb.DMatrix(X, feature_names=model.feature_names)
        probs = model.predict(dmat)
        preds = (probs >= 0.5).astype(int)
    return accuracy_score(y, preds) * 100

def retrain_lightgbm(X, y):
    banner("Retraining LightGBM (Volumetric Agent)…")
    lgbm_path = os.path.join(MODELS_DIR, "lightgbm_global_model.txt")
    if not os.path.exists(lgbm_path):
        print(f"[ERROR] {lgbm_path} not found."); return None

    model = lgb.Booster(model_file=lgbm_path)

    # Slice features to 77 BEFORE doing any evaluation
    X_adj = X[:, 2:] if X.shape[1] == 79 else X
    acc_before = evaluate_lgbm(model, X_adj, y)
    print(f"  Accuracy on queue BEFORE retraining: {acc_before:.2f}%")

    train_data = lgb.Dataset(X_adj, label=y)
    params = {
        "objective":        "binary",
        "metric":           "binary_logloss",
        "learning_rate":    0.03,
        "num_leaves":       31,
        "verbose":         -1,
        "n_jobs":           -1,
    }

    print("  Training 50 additional boosting rounds…")
    t0 = time.time()
    updated_model = lgb.train(
        params,
        train_data,
        num_boost_round=50,
        init_model=model,
        keep_training_booster=True,
    )
    print(f"  Done in {time.time()-t0:.1f}s")

    acc_after = evaluate_lgbm(updated_model, X_adj, y)
    print(f"  Accuracy on queue AFTER  retraining: {acc_after:.2f}%")
    print(f"  Improvement on misclassified flows:  {acc_after - acc_before:+.2f}%")

    updated_model.save_model(lgbm_path)
    print(f"  ✓ Model saved → {lgbm_path}")
    return acc_before, acc_after

def retrain_xgboost(X, y):
    banner("Retraining XGBoost (MQTT Agent)…")
    xgb_path = os.path.join(MODELS_DIR, "xgboost_mqtt_intrusion_model.pkl")
    if not os.path.exists(xgb_path):
        print(f"[ERROR] {xgb_path} not found."); return None

    model = joblib.load(xgb_path)

    # Drop src/dst ports to match mqtt_agent.py alignment BEFORE evaluation
    X_adj = X[:, 2:] if X.shape[1] == 79 else X
    acc_before = evaluate_xgb(model, X_adj, y)
    print(f"  Accuracy on queue BEFORE retraining: {acc_before:.2f}%")

    booster = model.get_booster() if hasattr(model, 'get_booster') else model
    feature_names = booster.feature_names

    dmat = xgb.DMatrix(X_adj, label=y, feature_names=feature_names)
    params = {
        "objective":        "binary:logistic",
        "eval_metric":      "logloss",
        "learning_rate":    0.03,
        "max_depth":        6,
        "subsample":        0.8,
        "verbosity":        0,
    }

    print("  Training 50 additional boosting rounds…")
    t0 = time.time()
    
    updated_booster = xgb.train(
        params,
        dmat,
        num_boost_round=50,
        xgb_model=booster,
    )
    print(f"  Done in {time.time()-t0:.1f}s")

    acc_after = evaluate_xgb(updated_booster, X_adj, y)
    print(f"  Accuracy on queue AFTER  retraining: {acc_after:.2f}%")
    print(f"  Improvement on misclassified flows:  {acc_after - acc_before:+.2f}%")

    if hasattr(model, 'get_booster'):
        model._Booster = updated_booster
        joblib.dump(model, xgb_path)
    else:
        joblib.dump(updated_booster, xgb_path)

    print(f"  ✓ Model saved → {xgb_path}")
    return acc_before, acc_after

def retrain_rf(X, y):
    banner("Retraining Random Forest (Recon Agent)…")
    
    rf_path = os.path.join(MODELS_DIR, "rf_model.pkl")
    if not os.path.exists(rf_path):
        print(f"[ERROR] {rf_path} not found. Check filename."); return None

    model = joblib.load(rf_path)

    # The RF model expects all 79 features, so we pass X directly without slicing
    preds = model.predict(X)
    acc_before = accuracy_score(y, preds) * 100
    print(f"  Accuracy on queue BEFORE retraining: {acc_before:.2f}%")

    print("  Adding 10 new estimators (trees) via warm_start…")
    t0 = time.time()
    
    # Enable warm start and add estimators
    model.warm_start = True
    model.n_estimators += 10
    
    # Fit the new trees to the misclassified data
    model.fit(X, y)
    print(f"  Done in {time.time()-t0:.1f}s")

    # Evaluate new accuracy
    preds_after = model.predict(X)
    acc_after = accuracy_score(y, preds_after) * 100
    print(f"  Accuracy on queue AFTER  retraining: {acc_after:.2f}%")
    print(f"  Improvement on misclassified flows:  {acc_after - acc_before:+.2f}%")

    joblib.dump(model, rf_path)
    print(f"  ✓ Model saved → {rf_path}")
    return acc_before, acc_after

def print_summary(lgbm_result, xgb_result, rf_result):
    banner("RETRAINING COMPLETE — Summary")
    print(f"  {'Model':<30} {'Before':>10}  {'After':>10}  {'Δ':>8}")
    print(f"  {'─'*62}")
    
    if lgbm_result:
        b, a = lgbm_result
        print(f"  {'LightGBM (Volumetric Agent)':<30} {b:>9.2f}%  {a:>9.2f}%  {a-b:>+7.2f}%")
    if xgb_result:
        b, a = xgb_result
        print(f"  {'XGBoost (MQTT Agent)':<30} {b:>9.2f}%  {a:>9.2f}%  {a-b:>+7.2f}%")
    if rf_result:
        b, a = rf_result
        print(f"  {'Random Forest (Recon Agent)':<30} {b:>9.2f}%  {a:>9.2f}%  {a-b:>+7.2f}%")
        
    print(f"\n  Models updated in {MODELS_DIR}/")
    print(f"  Originals backed up in {MODELS_BAK}/")
    print(f"\n  → Run orchestrator_master.py again to validate full-dataset accuracy.")

if __name__ == "__main__":
    print("\n" + "═"*60)
    print("  Agentic AI IDS — Incremental Model Retraining")
    print("  Active Learning Queue → Updated Models")
    print("═"*60)

    X, y, meta = load_queue()
    backup_models()

    lgbm_result = retrain_lightgbm(X, y)
    xgb_result  = retrain_xgboost(X, y)
    rf_result   = retrain_rf(X, y)

    print_summary(lgbm_result, xgb_result, rf_result)
