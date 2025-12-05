# ==============================================
# 1. 导入扩展库（新增LightGBM、网格搜索、正则化等）
# ==============================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix, roc_curve, auc
import re  # 用于提取姓名中的头衔

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==============================================
# 2. 增强版特征工程（核心优化点）
# ==============================================
def load_and_engineer_data(train_path, test_path=None):
    """加载数据并执行特征工程，返回处理后的训练/测试数据"""
    # 加载训练集
    df_train = pd.read_csv(train_path)
    df = df_train.copy()
    test_flag = False
    if test_path:
        df_test = pd.read_csv(test_path)
        df_test['Survived'] = -1  # 标记测试集标签
        df = pd.concat([df, df_test], ignore_index=True)
        test_flag = True

    # -------- 缺失值填充（精细化） --------
    # 修复：头衔提取加异常处理，避免姓名格式错误导致的空值
    def extract_title(name):
        try:
            return re.findall(r'([A-Za-z]+)\.', name)[0]
        except IndexError:
            return 'Mr'  # 异常姓名默认归为Mr

    df['Title'] = df['Name'].apply(extract_title)

    # 年龄：按头衔+舱位分组填充（比仅Pclass+Sex更精准）
    df['Age'] = df.groupby(['Pclass', 'Title'])['Age'].transform(lambda x: x.fillna(x.median())).fillna(28)  # 兜底填充
    # 登船港口：众数填充
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    # 票价：按舱位分组填充
    df['Fare'] = df.groupby('Pclass')['Fare'].transform(lambda x: x.fillna(x.median())).fillna(14.45)  # 兜底填充
    # 舱位：填充U并提取首字母，合并稀有舱位
    df['Cabin'] = df['Cabin'].fillna('U').apply(lambda x: x[0] if pd.notna(x) else 'U')
    rare_cabins = df['Cabin'].value_counts()[df['Cabin'].value_counts() < 10].index
    df['Cabin'] = df['Cabin'].replace(rare_cabins, 'R')  # 稀有舱位合并为R

    # -------- 衍生特征（核心！） --------
    # 1. 家庭规模：兄弟姐妹+父母子女+自己
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    # 2. 是否单身
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    # 3. 头衔合并（减少类别数）
    title_mapping = {
        'Mr': 'Mr', 'Mrs': 'Mrs', 'Miss': 'Miss', 'Master': 'Master',
        'Don': 'Noble', 'Sir': 'Noble', 'Lady': 'Noble', 'Countess': 'Noble', 'Dona': 'Noble',
        'Dr': 'Professional', 'Rev': 'Professional', 'Col': 'Military', 'Major': 'Military', 'Capt': 'Military',
        'Ms': 'Miss', 'Mlle': 'Miss', 'Mme': 'Mrs', 'Unknown': 'Mr'  # 新增异常头衔映射
    }
    df['Title'] = df['Title'].map(title_mapping).fillna('Mr')  # 兜底填充
    # 4. 票价分箱（捕捉非线性关系，修复NaN）
    df['FareBin'] = pd.cut(df['Fare'], bins=[0, 10, 30, 100, 600], labels=['Low', 'Mid', 'High', 'Luxury']).fillna(
        'Luxury')
    # 5. 年龄分箱（修复NaN）
    df['AgeBin'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100],
                          labels=['Child', 'Teen', 'Adult', 'Middle', 'Elder']).fillna('Adult')

    # -------- 筛选特征 --------
    # 最终特征列表（含衍生特征）
    core_features = [
        'Pclass', 'Sex', 'Embarked', 'Cabin', 'Title',
        'FamilySize', 'IsAlone', 'FareBin', 'AgeBin'
    ]
    target = 'Survived'

    # 拆分训练/测试集
    if test_flag:
        df_train_processed = df[df[target] != -1].copy()
        df_test_processed = df[df[target] == -1].copy()
        X_train = df_train_processed[core_features]
        y_train = df_train_processed[target]
        X_test = df_test_processed[core_features]
        passenger_id = df_test_processed['PassengerId']
        return X_train, y_train, X_test, passenger_id
    else:
        X = df[core_features]
        y = df[target]
        return X, y


# 加载数据并执行特征工程（修改为你的路径）
train_path = r'C:\Users\wangd\Desktop\kaggle\1_泰坦尼克号\train.csv'
test_path = r'C:\Users\wangd\Desktop\kaggle\1_泰坦尼克号\test.csv'
X, y, X_test, passenger_id = load_and_engineer_data(train_path, test_path)

# 划分训练集和验证集（比原代码的train_test_split更合理）
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ==============================================
# 3. 预处理管道（全局统一）
# ==============================================
# 定义类别特征（所有非数值特征）
categorical_features = X_train.columns.tolist()  # 经特征工程后均为类别特征
# 预处理：独热编码（忽略未知类别）
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ]
)


# ==============================================
# 4. 重构调参函数（核心：用Pipeline封装预处理+模型，消除特征名警告）
# ==============================================
def tune_model(preprocessor, model, param_grid, X, y):
    """
    网格搜索调参，封装预处理+模型的Pipeline
    :param preprocessor: 全局预处理管道
    :param model: 待调参的基模型
    :param param_grid: 调参网格（注意参数名要加模型别名__）
    :param X: 原始特征（DataFrame，带特征名）
    :param y: 标签
    :return: 最优Pipeline模型
    """
    # 封装预处理+模型的Pipeline
    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    # 网格搜索
    grid_search = GridSearchCV(
        estimator=model_pipeline,
        param_grid=param_grid,
        cv=5,  # 5折交叉验证
        scoring='accuracy',
        n_jobs=-1,  # 并行计算
        verbose=0
    )
    grid_search.fit(X, y)  # 输入是带特征名的DataFrame，由Pipeline内部处理
    print(f"✅ {model.__class__.__name__} 最优参数：{grid_search.best_params_}")
    print(f"✅ 交叉验证最优准确率：{grid_search.best_score_:.4f}")
    return grid_search.best_estimator_


# -------- 定义基模型及调参网格（参数名要加model__前缀！） --------
# XGBoost调参
xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_param = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [3, 5],
    'model__learning_rate': [0.05, 0.1],
    'model__subsample': [0.8, 1.0]
}
best_xgb_pipeline = tune_model(preprocessor, xgb, xgb_param, X_train, y_train)
# 提取调优后的XGB模型（用于堆叠）
best_xgb = best_xgb_pipeline.named_steps['model']

# LightGBM调参（新增高效模型）
lgb = LGBMClassifier(random_state=42, verbose=-1)
lgb_param = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [3, 5],
    'model__learning_rate': [0.05, 0.1],
    'model__num_leaves': [31, 63]
}
best_lgb_pipeline = tune_model(preprocessor, lgb, lgb_param, X_train, y_train)
# 提取调优后的LGB模型（用于堆叠）
best_lgb = best_lgb_pipeline.named_steps['model']

# 随机森林调参
rf = RandomForestClassifier(random_state=42)
rf_param = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [5, 8],
    'model__min_samples_split': [2, 5]
}
best_rf_pipeline = tune_model(preprocessor, rf, rf_param, X_train, y_train)
# 提取调优后的RF模型（用于堆叠）
best_rf = best_rf_pipeline.named_steps['model']

# ==============================================
# 5. 堆叠集成模型（Stacking）- 无警告版本
# ==============================================
# 定义基模型列表（所有模型均为调优后的实例）
base_models = [
    ('xgb', best_xgb),
    ('lgb', best_lgb),
    ('rf', best_rf),
    ('svc', SVC(probability=True, random_state=42)),  # SVM（带概率）
    ('lr', LogisticRegression(random_state=42, max_iter=500))  # 逻辑回归
]

# 堆叠集成：二级模型用逻辑回归，封装全局预处理管道
stacking_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('stacking', StackingClassifier(
        estimators=base_models,
        final_estimator=LogisticRegression(random_state=42, max_iter=500),
        cv=5,
        stack_method='predict_proba'
    ))
])

# 训练堆叠模型
stacking_pipeline.fit(X_train, y_train)

# ==============================================
# 6. 模型评估（准确率大幅提升，无警告）
# ==============================================
# 验证集预测
y_pred = stacking_pipeline.predict(X_val)
y_pred_proba = stacking_pipeline.predict_proba(X_val)[:, 1]

# 输出评估指标
print("\n🎯 优化后模型性能评估")
print("-" * 50)
print(f"验证集准确率: {accuracy_score(y_val, y_pred):.4f}")
print(f"验证集AUC: {roc_auc_score(y_val, y_pred_proba):.4f}")
print("\n📋 分类报告:")
print(classification_report(y_val, y_pred, target_names=['死亡', '生存']))

# 混淆矩阵+ROC曲线可视化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
cm = confusion_matrix(y_val, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['死亡', '生存'], yticklabels=['死亡', '生存'],
            ax=axes[0])
axes[0].set_title('混淆矩阵', fontsize=12)
axes[0].set_ylabel('真实标签')
axes[0].set_xlabel('预测标签')

fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
roc_auc = auc(fpr, tpr)
axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.4f})')
axes[1].plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
axes[1].set_xlabel('假正率')
axes[1].set_ylabel('真正率')
axes[1].set_title('ROC曲线')
axes[1].legend(loc="lower right")
plt.tight_layout()
plt.show()

# ==============================================
# 7. 保存模型+预测测试集（修复路径问题+Kaggle提交格式）
# ==============================================
# 修复：模型保存路径改为和数据集相同的路径
model_save_path = r'C:\Users\wangd\Desktop\kaggle\1_泰坦尼克号\titanic_stacking_model.pkl'
joblib.dump(stacking_pipeline, model_save_path)
print(f"\n✅ 模型已保存至：{model_save_path}")

# 测试集预测
y_test_pred = stacking_pipeline.predict(X_test)
y_test_proba = stacking_pipeline.predict_proba(X_test)[:, 1]

# 修复：生成Kaggle要求的提交文件（仅保留PassengerId和Survived）
submission = pd.DataFrame({
    'PassengerId': passenger_id,
    'Survived': y_test_pred.astype(int)
})
# 预测结果保存路径（不变）
result_save_path = r'C:\Users\wangd\Desktop\kaggle\1_泰坦尼克号\optimized_predict_result.csv'
submission.to_csv(result_save_path, index=False)
print(f"✅ Kaggle提交文件已保存至：{result_save_path}")
print("\n📌 测试集前5条预测结果：")
print(submission.head())

# 可选：生成含概率的结果文件（自己分析用）
analysis_result = pd.DataFrame({
    'PassengerId': passenger_id,
    'Survived': y_test_pred.astype(int),
    'Survived_Probability': y_test_proba
})
analysis_save_path = r'C:\Users\wangd\Desktop\kaggle\1_泰坦尼克号\analysis_result.csv'
analysis_result.to_csv(analysis_save_path, index=False)
print(f"✅ 含概率的分析文件已保存至：{analysis_save_path}")