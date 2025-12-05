# ------------------------------------------------------------
# 项目：乳腺癌良恶性二分类（XGBoost 实现）
# 功能：数据加载 → 标准化 → XGBoost建模 → 调参 → 评估 → 可视化
# 适用：Upwork / Kaggle / 教学 / 医疗辅助诊断原型
# ------------------------------------------------------------

# ----------------------------
# 1. 导入库
# ----------------------------
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc, roc_auc_score
)
import xgboost_train as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import randint, uniform

# ----------------------------
# 2. 设置中文字体（如不需要，可删除）
# ----------------------------
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ----------------------------
# 3. 加载数据
# ----------------------------
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names
target_names = data.target_names

print(f"📊 数据集：{data.DESCR.split('..')[0].strip()}")
print(f"样本数: {X.shape[0]}, 特征数: {X.shape[1]}")
print(f"类别分布: 恶性={np.sum(y==0)}, 良性={np.sum(y==1)}\n")

# ----------------------------
# 4. 划分数据集
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------
# 5. 特征标准化（XGBoost 不强制需要，但保留以兼容流程）
# ----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------
# 6. 超参数调优（轻量级随机搜索）
# ----------------------------
param_dist = {
    'n_estimators': randint(50, 300),
    'max_depth': randint(3, 10),
    'learning_rate': uniform(0.01, 0.3),  # 0.01 ~ 0.31
    'subsample': uniform(0.6, 0.4),       # 0.6 ~ 1.0
    'colsample_bytree': uniform(0.6, 0.4)
}

xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss'
)

# 随机搜索（只试 30 组，快速高效）
random_search = RandomizedSearchCV(
    xgb_model,
    param_distributions=param_dist,
    n_iter=30,
    scoring='roc_auc',
    cv=3,  # 3折加速
    random_state=42,
    n_jobs=-1,
    verbose=0
)

print("🔍 正在进行 XGBoost 超参数调优（约需 10~30 秒）...")
random_search.fit(X_train_scaled, y_train)

best_model = random_search.best_estimator_
print(f"✅ 最优参数: {random_search.best_params_}")
print(f"✅ 交叉验证 AUC: {random_search.best_score_:.4f}\n")

# ----------------------------
# 7. 预测
# ----------------------------
y_pred = best_model.predict(X_test_scaled)
y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]  # 良性概率

# ----------------------------
# 8. 评估
# ----------------------------
print("🎯 XGBoost 模型性能")
print("-" * 40)
print(f"测试集准确率: {accuracy_score(y_test, y_pred):.4f}")
print(f"测试集 AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")

print("\n📋 分类报告:")
print(classification_report(y_test, y_pred, target_names=target_names))

# ----------------------------
# 9. 可视化
# ----------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names, ax=axes[0])
axes[0].set_title('混淆矩阵')
axes[0].set_ylabel('真实标签')
axes[0].set_xlabel('预测标签')

# ROC 曲线
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.4f})')
axes[1].plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
axes[1].set_xlim([0.0, 1.0])
axes[1].set_ylim([0.0, 1.05])
axes[1].set_xlabel('假正率')
axes[1].set_ylabel('真正率')
axes[1].set_title('ROC 曲线')
axes[1].legend(loc="lower right")

plt.tight_layout()
plt.show()

# ----------------------------
# 10. 特征重要性（XGBoost 原生支持）
# ----------------------------
importances = best_model.feature_importances_
feat_imp_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print("\n🔍 前10个最重要特征:")
print(feat_imp_df.head(10).to_string(index=False))

# 可视化
top_n = 10
top_feat = feat_imp_df.head(top_n)
plt.figure(figsize=(10, 6))
plt.barh(range(top_n), top_feat['Importance'], color='steelblue')
plt.yticks(range(top_n), top_feat['Feature'])
plt.xlabel('特征重要性')
plt.title(f'XGBoost：前 {top_n} 个最重要特征')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()