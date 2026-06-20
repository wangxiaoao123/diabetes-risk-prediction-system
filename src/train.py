# 必要なライブラリをインポートする
from ucimlrepo import fetch_ucirepo

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. データセットの読み込み
# ============================================================

# UCI Machine Learning Repository から
# CDC Diabetes Health Indicators データセットを取得する
cdc_diabetes_health_indicators = fetch_ucirepo(id=891)

# 説明変数（特徴量）と目的変数（糖尿病の有無）を取得する
X = cdc_diabetes_health_indicators.data.features
y = cdc_diabetes_health_indicators.data.targets

# y が DataFrame 形式の場合、1次元の Series に変換する
y = y.iloc[:, 0]


# ============================================================
# 2. データの基本情報を確認する
# ============================================================

print("データ数と特徴量数:")
print(X.shape)

print("\n特徴量の一覧:")
print(X.columns)

print("\n目的変数の分布:")
print(y.value_counts())

print("\n欠損値の確認:")
print(X.isnull().sum())


# ============================================================
# 3. 訓練データとテストデータに分割する
# ============================================================

# stratify=y を指定することで、
# 訓練データとテストデータの糖尿病あり・なしの割合をなるべく同じにする
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================================================
# 4. 特徴量の標準化
# ============================================================

# ロジスティック回帰では、特徴量のスケールをそろえるために標準化を行う
scaler = StandardScaler()

# 訓練データで平均と標準偏差を計算し、変換する
X_train_scaled = scaler.fit_transform(X_train)

# テストデータは、訓練データで計算した基準を使って変換する
X_test_scaled = scaler.transform(X_test)


# ============================================================
# 5. ロジスティック回帰モデルの学習
# ============================================================

# class_weight="balanced" を指定することで、
# 糖尿病あり・なしのデータ数の偏りを考慮する
model = LogisticRegression(
    max_iter=2000,
    random_state=42,
    class_weight="balanced"
)

# モデルを訓練データで学習させる
model.fit(X_train_scaled, y_train)


# ============================================================
# 6. テストデータで予測する
# ============================================================

# テストデータに対して予測を行う
y_pred = model.predict(X_test_scaled)


# ============================================================
# 7. モデルの評価
# ============================================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n===== Model Evaluation =====")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

print("\n混同行列:")
print(confusion_matrix(y_test, y_pred))

print("\n分類レポート:")
print(classification_report(y_test, y_pred))


# ============================================================
# 8. 特徴量の係数を確認する
# ============================================================

# ロジスティック回帰の係数を DataFrame にまとめる
coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
})

# 係数の絶対値が大きい順に並べる
coef_df = coef_df.sort_values(
    by="Coefficient",
    key=abs,
    ascending=False
)

print("\n===== Feature Coefficients =====")
print(coef_df)


# ============================================================
# 9. 特徴量の係数をグラフで表示する
# ============================================================

plt.figure(figsize=(10, 8))

plt.barh(
    coef_df["Feature"],
    coef_df["Coefficient"]
)

plt.xlabel("Coefficient")
plt.ylabel("Feature")
plt.title("Feature Coefficients of Logistic Regression")

# 影響が大きい特徴量を上に表示する
plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()


import joblib

joblib.dump({
    "model": model,
    "scaler": scaler,
    "features": X.columns.tolist()
}, "src/model.pkl")

print("Model and scaler saved to src/model.pkl")