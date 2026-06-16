# ============================================================
#  ADD THIS AS A NEW CELL AT THE BOTTOM OF Diabetes.ipynb
#  Run it once to export all assets for the Flask web app.
# ============================================================

import os
import joblib
import matplotlib
matplotlib.use('Agg')          # non-interactive backend — safe in notebooks too
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# ── 0. Output directories ────────────────────────────────────
os.makedirs('static/images', exist_ok=True)
print("✔ static/images/ directory ready")

# ============================================================
# SECTION A — EDA GRAPHS  (Member 1 visualisations)
# ============================================================

# A1 — Histograms
features = ['age', 'bmi', 'blood_glucose_level', 'HbA1c_level']
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for ax, feat in zip(axes.flatten(), features):
    ax.hist(df[feat].dropna(), bins=25, color='#4C72B0', edgecolor='white', alpha=0.85)
    ax.set_title(feat.replace('_', ' ').title(), fontweight='bold')
    ax.set_xlabel('Value'); ax.set_ylabel('Count')
plt.suptitle('Feature Distributions', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('static/images/eda_histograms.png', dpi=120, bbox_inches='tight')
plt.close()
print("✔ Saved: static/images/eda_histograms.png")

# A2 — Boxplots
fig, axes = plt.subplots(1, 3, figsize=(12, 5))
for ax, feat, color in zip(axes, ['bmi', 'blood_glucose_level', 'age'],
                            ['#4C72B0', '#55A868', '#C44E52']):
    sns.boxplot(y=df[feat], ax=ax, color=color)
    ax.set_title(feat.replace('_', ' ').title(), fontweight='bold')
plt.suptitle('Box Plots — Outlier Detection', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('static/images/eda_boxplots.png', dpi=120, bbox_inches='tight')
plt.close()
print("✔ Saved: static/images/eda_boxplots.png")

# A3 — Correlation heatmap  (use df2 which already has encoding)
corr = df2.corr(numeric_only=True)
fig, ax = plt.subplots(figsize=(11, 7))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5, ax=ax)
ax.set_title('Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('static/images/eda_correlation_heatmap.png', dpi=120, bbox_inches='tight')
plt.close()
print("✔ Saved: static/images/eda_correlation_heatmap.png")

# A4 — Gender vs Diabetes
fig, ax = plt.subplots(figsize=(7, 5))
sns.countplot(x='gender', hue='diabetes', data=df, palette=['#4C72B0', '#DD8452'], ax=ax)
ax.set_title('Gender vs Diabetes', fontsize=13, fontweight='bold')
ax.set_xlabel('Gender'); ax.set_ylabel('Count')
ax.legend(title='Diabetes', labels=['No', 'Yes'])
plt.tight_layout()
plt.savefig('static/images/eda_gender_diabetes.png', dpi=120, bbox_inches='tight')
plt.close()
print("✔ Saved: static/images/eda_gender_diabetes.png")

# A5 — Smoking vs Diabetes
fig, ax = plt.subplots(figsize=(9, 5))
sns.countplot(x='smoking_history', hue='diabetes', data=df, palette=['#4C72B0', '#DD8452'], ax=ax)
ax.set_title('Smoking History vs Diabetes', fontsize=13, fontweight='bold')
ax.set_xlabel('Smoking History'); ax.set_ylabel('Count')
plt.xticks(rotation=40, ha='right')
ax.legend(title='Diabetes', labels=['No', 'Yes'])
plt.tight_layout()
plt.savefig('static/images/eda_smoking_diabetes.png', dpi=120, bbox_inches='tight')
plt.close()
print("✔ Saved: static/images/eda_smoking_diabetes.png")

# ============================================================
# SECTION B — MEMBER 2 — SMOTE class balance graph
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
fig.suptitle('Class Distribution Before & After SMOTE', fontsize=11, fontweight='bold')

axes[0].bar(['No Diabetes', 'Diabetes'], y_train_raw.value_counts().values,
            color=['#4C72B0', '#DD8452'], edgecolor='white')
axes[0].set_title('Before SMOTE'); axes[0].set_ylabel('Count')

axes[1].bar(['No Diabetes', 'Diabetes'],
            [int((pd.Series(y_train_smote) == 0).sum()),
             int((pd.Series(y_train_smote) == 1).sum())],
            color=['#4C72B0', '#DD8452'], edgecolor='white')
axes[1].set_title('After SMOTE')

plt.tight_layout()
plt.savefig('static/images/m2_smote_balance.png', dpi=120, bbox_inches='tight')
plt.close()
print("✔ Saved: static/images/m2_smote_balance.png")

# ============================================================
# SECTION C — MEMBER 3 — Model comparison bar chart
# ============================================================

metrics_to_plot = ['Accuracy', 'Recall', 'F1-Score']
ds_colors = {'A_Unscaled': '#4C72B0', 'B_Standardized': '#55A868', 'C_Normalized': '#C44E52'}
model_names = list(models.keys())
x = np.arange(len(model_names))
bar_w = 0.25

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Model Comparison: Accuracy, Recall & F1 (3 Dataset Versions)',
             fontsize=12, fontweight='bold')

for ax, metric in zip(axes, metrics_to_plot):
    for i, ds in enumerate(datasets):
        vals = [results_df[(results_df['Dataset'] == ds) &
                           (results_df['Model'] == m)][metric].values[0]
                for m in model_names]
        ax.bar(x + i * bar_w, vals, bar_w, label=ds,
               color=ds_colors[ds], edgecolor='white', alpha=0.88)
    ax.set_title(metric, fontweight='bold', fontsize=11)
    ax.set_xticks(x + bar_w)
    ax.set_xticklabels([m.replace(' ', '\n') for m in model_names], fontsize=8)
    ax.set_ylim(0, 1.1); ax.set_ylabel(metric)
    ax.legend(fontsize=7, loc='lower right')
    ax.axhline(0.9, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

plt.tight_layout()
plt.savefig('static/images/m3_model_comparison.png', dpi=120, bbox_inches='tight')
plt.close()
print("✔ Saved: static/images/m3_model_comparison.png")

# ============================================================
# SECTION D — Confusion matrices for best model
# ============================================================

best_model_name = best_row['Model']   # already computed in your notebook
best_ds_name    = best_row['Dataset']

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle(f'Confusion Matrices: {best_model_name} (Best Model by Recall)',
             fontsize=11, fontweight='bold')

for ax, ds_name in zip(axes, datasets):
    cm = all_cms[(ds_name, best_model_name)]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=['No Diabetes', 'Diabetes'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(ds_name, fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('static/images/m3_confusion_matrices.png', dpi=120, bbox_inches='tight')
plt.close()
print("✔ Saved: static/images/m3_confusion_matrices.png")

# ============================================================
# SECTION E — Export best model + scaler as .pkl
# ============================================================
# The best model used scaler_std2 (StandardScaler) on the B_Standardized dataset.
# We re-identify it from results_df to be safe.

best_ds_name    = best_row['Dataset']   # e.g. 'B_Standardized'
best_model_name = best_row['Model']     # e.g. 'Random Forest'

# Map dataset name → (fitted scaler, X_train, X_test)
dataset_scaler_map = {
    'A_Unscaled':     (None,         X_train_A, X_test_A),
    'B_Standardized': (scaler_std2,  X_train_B, X_test_B),
    'C_Normalized':   (scaler_mm2,   X_train_C, X_test_C),
}

best_scaler, best_Xtr, best_Xte = dataset_scaler_map[best_ds_name]

# Re-instantiate and re-fit the best model cleanly on the correct training split
model_map = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'KNN':                 KNeighborsClassifier(n_neighbors=5),
    'Decision Tree':       DecisionTreeClassifier(max_depth=6, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
}
best_clf = model_map[best_model_name]
best_clf.fit(best_Xtr, y_train_smote)
print(f"✔ Best model re-fitted: {best_model_name} on {best_ds_name}")

# Save model
joblib.dump(best_clf, 'best_model.pkl')
print("✔ Saved: best_model.pkl")

# Save scaler  (may be None if A_Unscaled is best — rare for medical data)
if best_scaler is not None:
    joblib.dump(best_scaler, 'scaler.pkl')
    print("✔ Saved: scaler.pkl")
else:
    # Fit a StandardScaler anyway so app.py always works the same way
    from sklearn.preprocessing import StandardScaler
    fallback_scaler = StandardScaler()
    fallback_scaler.fit(X_train_smote)
    joblib.dump(fallback_scaler, 'scaler.pkl')
    print("✔ Saved: scaler.pkl  (identity-fitted fallback; dataset was unscaled)")

# Save the exact feature columns the model was trained on
import json
feature_cols = list(best_Xtr.columns)
with open('feature_columns.json', 'w') as f:
    json.dump(feature_cols, f)
print(f"✔ Saved: feature_columns.json  →  {feature_cols}")

print("\n🎉 All assets exported successfully! You can now run the Flask app.")
