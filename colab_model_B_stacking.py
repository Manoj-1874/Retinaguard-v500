# ==============================================================================
# MODEL B: STACKING CLASSIFIER / META-LEARNER
# ==============================================================================
# This script extracts the deep mathematical features from our CNN
# and uses them to train a Random Forest and XGBoost Meta-Learner.
# ==============================================================================

import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from tensorflow import keras
import joblib

print("="*60)
print(" STACKING CLASSIFIER (MODEL B) GENERATOR")
print("="*60)

# 1. Load the previously fine-tuned model
print("\n[1] Loading the fine-tuned base model from Drive...")
base_model = keras.models.load_model('/content/drive/MyDrive/finetuned_model.h5', compile=False)

# 2. Chop off the final prediction layer to turn it into a Feature Extractor
# We want the 2048-number array before it makes a decision
feature_extractor = keras.Model(inputs=base_model.inputs, outputs=base_model.layers[-2].output)
print("    [+] Feature Extractor created successfully.")

# 3. Extract Features from Training Data
print("\n[2] Extracting deep mathematical features from Training Images (this takes a moment)...")
X_train_features = feature_extractor.predict(X_train, batch_size=16)
print(f"    [+] Extracted features shape: {X_train_features.shape}")

# 4. Extract Features from Validation Data
print("\n[3] Extracting features from Validation Images...")
X_val_features = feature_extractor.predict(X_val, batch_size=16)

# 5. Train Random Forest Meta-Learner
print("\n[4] Training Random Forest Meta-Learner...")
rf_clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
rf_clf.fit(X_train_features, y_train)

# Evaluate Random Forest
rf_preds = rf_clf.predict(X_val_features)
rf_probs = rf_clf.predict_proba(X_val_features)[:, 1]
rf_acc = accuracy_score(y_val, rf_preds)
rf_auc = roc_auc_score(y_val, rf_probs)
print(f"    [+] Random Forest Accuracy: {rf_acc*100:.2f}%")
print(f"    [+] Random Forest AUC: {rf_auc:.4f}")

# 6. Train XGBoost Meta-Learner
print("\n[5] Training XGBoost Meta-Learner...")
xgb_clf = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_clf.fit(X_train_features, y_train)

# Evaluate XGBoost
xgb_preds = xgb_clf.predict(X_val_features)
xgb_probs = xgb_clf.predict_proba(X_val_features)[:, 1]
xgb_acc = accuracy_score(y_val, xgb_preds)
xgb_auc = roc_auc_score(y_val, xgb_probs)
print(f"    [+] XGBoost Accuracy: {xgb_acc*100:.2f}%")
print(f"    [+] XGBoost AUC: {xgb_auc:.4f}")

# 7. Save the Best Meta-Learner
print("\n[6] Saving the best Meta-Learner to Google Drive...")
if xgb_auc > rf_auc:
    best_model = xgb_clf
    model_name = "XGBoost"
else:
    best_model = rf_clf
    model_name = "Random Forest"

save_path = '/content/drive/MyDrive/meta_learner.pkl'
joblib.dump(best_model, save_path)

print(f"\nSUCCESS! 🚀")
print(f"The best Meta-Learner ({model_name}) has been saved to: {save_path}")
print("We now have Model B!")
