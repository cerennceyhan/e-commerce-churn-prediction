import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
import shap
import warnings
import os
warnings.filterwarnings('ignore')

print("=" * 80)
print("ADIM 2: MODEL EĞİTİMİ (SADECE LLM ÖZELLİKLERİ)")
print("=" * 80)

# Proje kök dizinini bul
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Dosya yollarını göreceli olarak oluştur
data_path = os.path.join(project_root, 'data', 'processed', 'llm_extraction.csv')
output_dir = os.path.join(project_root, 'outputs')

# Output dizini yoksa oluştur
os.makedirs(output_dir, exist_ok=True)

# Veriyi yükle
df = pd.read_csv(data_path, encoding='utf-8-sig')

print(f"\n✅ {len(df)} ürün yüklendi")
print("\n🔧 Özellikler hazırlanıyor...")

llm_features = [
    'fitment_problem',
    'fitment_severity',
    'quality_sentiment',
    'delivery_issue',
    'color_mismatch',
    'fabric_quality_issue',
    'price_value_perception'
]

# Boolean'ları int'e çevir
for col in llm_features:
    if df[col].dtype == 'bool' or df[col].dtype == 'object':
        try:
            if df[col].dtype == 'object':
                df[col] = df[col].map({'True': 1, 'False': 0, True: 1, False: 0})
            else:
                df[col] = df[col].astype(int)
        except:
            df[col] = 0

# NaN doldur
df[llm_features] = df[llm_features].fillna(0)

# X ve y
X = df[llm_features].copy()
y = df['Risk_Class'].copy()

print(f"✅ Toplam özellik sayısı: {len(llm_features)} (SADECE LLM)")
print(f"\n📋 ÖZELLİKLER:")
for feat in llm_features:
    print(f"   ✓ {feat}")

# ============================================================================
# TRAIN-TEST SPLIT
# ============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"\n📊 Veri Bölünmesi:")
print(f"   Train: {len(X_train)} sample")
print(f"   Test:  {len(X_test)} sample")

# ============================================================================
# SMOTE
# ============================================================================
print(f"\n⚖️ SMOTE uygulanıyor...")
smote = SMOTE(random_state=42, k_neighbors=3)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"✅ Train size: {len(X_train)} → {len(X_train_balanced)}")

# ============================================================================
# MODEL EĞİTİMİ
# ============================================================================
print(f"\n🚀 Model eğitimi başlıyor...")

from sklearn.utils.class_weight import compute_sample_weight
sample_weights = compute_sample_weight(
    class_weight='balanced',
    y=y_train_balanced
)

model = XGBClassifier(
    n_estimators=150,
    max_depth=5,
    learning_rate=0.1,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    objective='multi:softmax',
    num_class=3,
    random_state=42,
    eval_metric='mlogloss'
)

model.fit(X_train_balanced, y_train_balanced, sample_weight=sample_weights)
print("✅ Model eğitildi!")

# ============================================================================
# DEĞERLENDİRME
# ============================================================================
print(f"\n📈 Model Değerlendirmesi:")
print("=" * 80)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"\n⭐ Test Set Accuracy: {accuracy:.3f}")
print(f"⭐ Test Set F1-Score: {f1:.3f}")

class_names = ['Healthy', 'Quality Churn', 'Engagement Churn']
print(f"\n📊 CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred, target_names=class_names))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
           xticklabels=class_names,
           yticklabels=class_names,
           cbar_kws={'label': 'Count'})
for i in range(len(class_names)):
    for j in range(len(class_names)):
        plt.text(j + 0.5, i + 0.7, 
                f'({cm_normalized[i, j]:.1%})',
                ha='center', va='center', 
                fontsize=9, color='gray')
plt.title('Confusion Matrix - LLM Features Only', fontsize=16)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
confusion_matrix_path = os.path.join(output_dir, 'confusion_matrix.png')
plt.savefig(confusion_matrix_path, dpi=300)
plt.close()
print(f"\n💾 Confusion matrix kaydedildi: {confusion_matrix_path}")

# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================
print(f"\n🎯 ÖZELLİK ÖNEMLERİ:")
importance_df = pd.DataFrame({
    'feature': llm_features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print(importance_df.to_string(index=False))

plt.figure(figsize=(12, 8))
bars = plt.barh(range(len(importance_df)), importance_df['importance'], color='steelblue')
plt.yticks(range(len(importance_df)), importance_df['feature'])
plt.xlabel('Importance (Gain)', fontsize=12)
plt.title('LLM Feature Importance', fontsize=16)
plt.gca().invert_yaxis()
for i, (bar, val) in enumerate(zip(bars, importance_df['importance'])):
    plt.text(val, i, f' {val:.3f}', va='center', fontsize=9)
plt.tight_layout()
feature_importance_path = os.path.join(output_dir, 'feature_importance.png')
plt.savefig(feature_importance_path, dpi=300)
plt.close()
print(f"💾 Feature importance kaydedildi: {feature_importance_path}")

# ============================================================================
# SHAP
# ============================================================================
print(f"\n🔍 SHAP analizi başlıyor...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test.head(100))

for class_idx, class_name in enumerate(class_names):
    plt.figure(figsize=(12, 8))
    if isinstance(shap_values, list):
        shap_vals_class = shap_values[class_idx]
    else:
        shap_vals_class = shap_values[:, :, class_idx]
    
    shap.summary_plot(
        shap_vals_class,
        X_test.head(100).values,
        feature_names=llm_features,
        show=False,
        max_display=7
    )
    plt.title(f'SHAP - {class_name}', fontsize=16)
    plt.tight_layout()
    shap_path = os.path.join(output_dir, f'shap_{class_idx}_{class_name.replace(" ", "_")}.png')
    plt.savefig(shap_path, dpi=300)
    plt.close()
    print(f"   💾 SHAP {class_name} kaydedildi: {shap_path}")

# ============================================================================
# TAHMİNLER
# ============================================================================
y_pred_proba = model.predict_proba(X_test)
results_df = pd.DataFrame({
    'Ürün': df.loc[X_test.index, 'Ürün'].values,
    'True_Class': y_test.values,
    'Predicted_Class': y_pred,
    'Prob_Healthy': y_pred_proba[:, 0],
    'Prob_Quality_Churn': y_pred_proba[:, 1],
    'Prob_Engagement_Churn': y_pred_proba[:, 2]
})
predictions_path = os.path.join(output_dir, 'predictions.csv')
results_df.to_csv(predictions_path, index=False, encoding='utf-8-sig')
print(f"\n💾 Tahminler kaydedildi: {predictions_path}")

wrong = results_df[results_df['True_Class'] != results_df['Predicted_Class']]
print(f"\n❌ Yanlış Tahmin: {len(wrong)}/{len(results_df)}")

print("\n" + "=" * 80)
print(f"\n📌 SONUÇLAR:")
print(f"   Accuracy: {accuracy:.1%}")
print(f"   F1-Score: {f1:.1%}")