import streamlit as st
import pandas as pd
import joblib

# ============================================================
# モデルの読み込み
# ============================================================

saved = joblib.load("src/model.pkl")

model = saved["model"]
scaler = saved["scaler"]
features = saved["features"]


# ============================================================
# 年齢をCDCデータセットのAgeコードに変換する関数
# ============================================================

def convert_age_to_group(age):
    if age <= 24:
        return 1
    elif age <= 29:
        return 2
    elif age <= 34:
        return 3
    elif age <= 39:
        return 4
    elif age <= 44:
        return 5
    elif age <= 49:
        return 6
    elif age <= 54:
        return 7
    elif age <= 59:
        return 8
    elif age <= 64:
        return 9
    elif age <= 69:
        return 10
    elif age <= 74:
        return 11
    elif age <= 79:
        return 12
    else:
        return 13


# ============================================================
# UI
# ============================================================

st.title("糖尿病リスク予測システム")

st.write(
    "健康指標を入力すると、糖尿病リスクを予測します。"
)

st.caption(
    "本システムは研究・教育目的であり、医療診断を行うものではありません。"
)


# ============================================================
# ユーザー入力
# ============================================================

age_real = st.number_input(
    "年齢",
    min_value=18,
    max_value=100,
    value=26
)

bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=70.0,
    value=24.0
)

highbp_label = st.selectbox(
    "高血圧と診断されたことがありますか？",
    ["いいえ", "はい"]
)

highchol_label = st.selectbox(
    "高コレステロールと診断されたことがありますか？",
    ["いいえ", "はい"]
)

smoker_label = st.selectbox(
    "これまでに一定期間喫煙したことがありますか？",
    ["いいえ", "はい"]
)

phys_activity_label = st.selectbox(
    "日常的に運動習慣がありますか？",
    ["はい", "いいえ"]
)

fruits_label = st.selectbox(
    "普段から果物を摂取していますか？",
    ["はい", "いいえ"]
)

veggies_label = st.selectbox(
    "普段から野菜を摂取していますか？",
    ["はい", "いいえ"]
)

genhlth_label = st.selectbox(
    "現在の健康状態を選択してください",
    ["非常に良い", "良い", "普通", "悪い", "非常に悪い"]
)

sex_label = st.selectbox(
    "性別",
    ["女性", "男性"]
)


# ============================================================
# 入力値の変換
# ============================================================

age = convert_age_to_group(age_real)

highbp = 1 if highbp_label == "はい" else 0
highchol = 1 if highchol_label == "はい" else 0
smoker = 1 if smoker_label == "はい" else 0

phys_activity = 1 if phys_activity_label == "はい" else 0
fruits = 1 if fruits_label == "はい" else 0
veggies = 1 if veggies_label == "はい" else 0

genhlth_map = {
    "非常に良い": 1,
    "良い": 2,
    "普通": 3,
    "悪い": 4,
    "非常に悪い": 5
}
genhlth = genhlth_map[genhlth_label]

sex = 1 if sex_label == "男性" else 0


# ============================================================
# 予測処理
# ============================================================

if st.button("Predict"):

    input_data = pd.DataFrame([{
        "HighBP": highbp,
        "HighChol": highchol,
        "CholCheck": 1,
        "BMI": bmi,
        "Smoker": smoker,
        "Stroke": 0,
        "HeartDiseaseorAttack": 0,
        "PhysActivity": phys_activity,
        "Fruits": fruits,
        "Veggies": veggies,
        "HvyAlcoholConsump": 0,
        "AnyHealthcare": 1,
        "NoDocbcCost": 0,
        "GenHlth": genhlth,
        "MentHlth": 0,
        "PhysHlth": 0,
        "DiffWalk": 0,
        "Sex": sex,
        "Age": age,
        "Education": 4,
        "Income": 6
    }])

    input_data = input_data[features]

    input_scaled = scaler.transform(input_data)

    prob = model.predict_proba(input_scaled)[0][1]

    st.subheader("予測結果")

    st.write(f"糖尿病リスク: {prob * 100:.1f}%")

    if prob < 0.3:
        st.success("低リスク")
    elif prob < 0.7:
        st.warning("中リスク")
    else:
        st.error("高リスク")

    st.write("### 入力内容の確認")
    st.write(input_data)