# ------------------------------------------------------------
# 项目：乳腺癌良恶性二分类（逻辑回归）
# 功能：数据加载 → 标准化 → 建模 → 多维度评估 → 特征重要性分析
# 环境要求：Python + sklearn + matplotlib + seaborn + pandas
# ------------------------------------------------------------

# ----------------------------
# 1. 导入所需库
# ----------------------------
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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
# 设置支持中文的字体（按优先级尝试）
plt.rcParams['font.sans-serif'] = [
    'SimHei',           # Windows 黑体
    'Microsoft YaHei',  # Windows 微软雅黑
    'PingFang SC',      # Mac 苹方
    'Arial Unicode MS', # 跨平台 Unicode 字体
    'DejaVu Sans'       # 开源备用字体
]
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号（如 -0.5）

# ----------------------------
# 3. 加载数据集
# ----------------------------
data = load_breast_cancer()
X, y = data.data, data.target          # X: 特征矩阵 (569×30), y: 标签 (0=恶性, 1=良性)
feature_names = data.feature_names     # 特征名称列表
target_names = data.target_names       # ['malignant', 'benign']

print(f"📊 数据集信息：{data.DESCR.split('..')[0].strip()}")  # 打印简要描述
print(f"样本数: {X.shape[0]}, 特征数: {X.shape[1]}")
print(f"类别分布: 恶性={sum(y==0)}, 良性={sum(y==1)}\n")

# ----------------------------
# 4. 划分训练集与测试集（8:2，分层抽样保持类别比例）
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # 确保训练/测试集中恶性:良性比例一致
)

# ----------------------------
# 5. 特征标准化（防止量纲影响模型）
# 注意：只用训练集拟合 scaler，避免数据泄露！
# ----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # 拟合并转换训练集
X_test_scaled = scaler.transform(X_test)        # 仅转换测试集（使用训练集的均值和标准差）

# ----------------------------
# 6. 训练逻辑回归模型
# ----------------------------
model = LogisticRegression(
    max_iter=10000,   # 增加迭代次数确保收敛
    random_state=42   # 保证结果可复现
)
model.fit(X_train_scaled, y_train)

# ----------------------------
# 7. 预测
# ----------------------------
y_pred = model.predict(X_test_scaled)               # 预测类别
y_pred_proba = model.predict_proba(X_test_scaled)   # 预测概率，形状 (n, 2)
y_pred_proba_positive = y_pred_proba[:, 1]          # 取“良性”（正类）的概率用于 ROC

# ----------------------------
# 8. 模型评估
# ----------------------------
print("🎯 模型性能评估")
print("-" * 50)
print(f"准确率 (Accuracy): {accuracy_score(y_test, y_pred):.4f}")
print(f"AUC (ROC曲线下面积): {roc_auc_score(y_test, y_pred_proba_positive):.4f}")
print("\n📋 分类报告:")
print(classification_report(y_test, y_pred, target_names=target_names))

# ----------------------------
# 9. 可视化：混淆矩阵 + ROC 曲线
# ----------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- 混淆矩阵 ---
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=target_names,
    yticklabels=target_names,
    ax=axes[0]
)
axes[0].set_title('混淆矩阵')
axes[0].set_ylabel('真实标签')
axes[0].set_xlabel('预测标签')

# --- ROC 曲线 ---
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
# 10. 特征重要性分析（逻辑回归系数）
# ----------------------------
coef = model.coef_[0]  # 逻辑回归对每个特征的权重（长度=30）
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coef,
    'Abs_Coefficient': np.abs(coef)
}).sort_values(by='Abs_Coefficient', ascending=False)

print("\n🔍 前10个最重要的特征（按权重绝对值排序）:")
print(feature_importance_df[['Feature', 'Coefficient']].head(10).to_string(index=False))

# --- 可视化前10重要特征 ---
top_n = 10
top_features = feature_importance_df.head(top_n)

plt.figure(figsize=(10, 6))
colors = ['red' if c < 0 else 'green' for c in top_features['Coefficient']]
plt.barh(range(top_n), top_features['Coefficient'], color=colors)
plt.yticks(range(top_n), top_features['Feature'])
plt.xlabel('逻辑回归系数')
plt.title(f'前 {top_n} 个最重要特征的权重\n（红色：增大该特征 → 更可能是恶性；绿色：更可能是良性）')
plt.gca().invert_yaxis()  # 最重要的特征在顶部
plt.tight_layout()
plt.show()