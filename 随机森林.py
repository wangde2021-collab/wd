# ------------------------------------------------------------
# 项目：乳腺癌良恶性二分类（随机森林）
# 功能：数据加载 → 标准化 → 随机森林建模 → 多维度评估 → 特征重要性分析
# 作者：AI助手
# 环境要求：Python + sklearn + matplotlib + seaborn + pandas
# ------------------------------------------------------------

# ----------------------------
# 1. 导入所需库
# ----------------------------
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# ----------------------------
# 2. 解决 Matplotlib 中文显示问题
# ----------------------------
plt.rcParams['font.sans-serif'] = [
    'SimHei', 'Microsoft YaHei', 'PingFang SC', 'Arial Unicode MS', 'DejaVu Sans'
]
plt.rcParams['axes.unicode_minus'] = False

# ----------------------------
# 3. 加载数据集
# ----------------------------
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names
target_names = data.target_names

print(f"📊 数据集信息：{data.DESCR.split('..')[0].strip()}")
print(f"样本数: {X.shape[0]}, 特征数: {X.shape[1]}")
print(f"类别分布: 恶性={sum(y==0)}, 良性={sum(y==1)}\n")

# ----------------------------
# 4. 划分训练集与测试集
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ----------------------------
# 5. 特征标准化（对树模型非必需，但保留以保持流程一致）
# 注意：随机森林对量纲不敏感，但标准化不影响结果
# ----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------
# 6. 训练随机森林模型
# ----------------------------
model = RandomForestClassifier(
    n_estimators=100,      # 树的数量
    max_depth=5,           # 控制过拟合（可调）
    random_state=42,
    n_jobs=-1              # 并行加速
)
model.fit(X_train_scaled, y_train)

# ----------------------------
# 7. 预测
# ----------------------------
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)
y_pred_proba_positive = y_pred_proba[:, 1]  # 良性（正类）概率

# ----------------------------
# 8. 模型评估
# ----------------------------
print("🎯 模型性能评估（随机森林）")
print("-" * 50)
print(f"准确率 (Accuracy): {accuracy_score(y_test, y_pred):.4f}")
print(f"AUC (ROC曲线下面积): {roc_auc_score(y_test, y_pred_proba_positive):.4f}")

# 交叉验证（更稳健的评估）
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
print(f"5折交叉验证 AUC 均值: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

print("\n📋 分类报告:")
print(classification_report(y_test, y_pred, target_names=target_names))

# ----------------------------
# 9. 可视化：混淆矩阵 + ROC 曲线
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
fpr, tpr, _ = roc_curve(y_test, y_pred_proba_positive)
roc_auc = auc(fpr, tpr)
axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.4f})')
axes[1].plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', label='随机分类器')
axes[1].set_xlim([0.0, 1.0])
axes[1].set_ylim([0.0, 1.05])
axes[1].set_xlabel('假正率 (False Positive Rate)')
axes[1].set_ylabel('真正率 (True Positive Rate)')
axes[1].set_title('ROC 曲线')
axes[1].legend(loc="lower right")

plt.tight_layout()
plt.show()

# ----------------------------
# 10. 特征重要性分析（随机森林原生支持！）
# ----------------------------
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\n🔍 前10个最重要的特征（随机森林）:")
print(feature_importance_df.head(10).to_string(index=False))

# 可视化前10重要特征
top_n = 10
top_features = feature_importance_df.head(top_n)

plt.figure(figsize=(10, 6))
plt.barh(range(top_n), top_features['Importance'], color='steelblue')
plt.yticks(range(top_n), top_features['Feature'])
plt.xlabel('特征重要性')
plt.title(f'随机森林：前 {top_n} 个最重要特征')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()