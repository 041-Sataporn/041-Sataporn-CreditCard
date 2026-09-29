import os
import re
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ------------------------------------------------------------------
# ตั้งค่า
# ------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "credit_card_fraud_mlp.keras"
SCALER_PATH = BASE_DIR / "scaler.pkl"      # ไม่บังคับ: ถ้าตอนเทรนมีการ scale ข้อมูล
TEST_PATH = BASE_DIR / "test_data.csv"     # ไม่บังคับ: ชุดทดสอบ (ข้อมูลดิบ + คอลัมน์ Class)

SYSTEM_NAME = "ระบบตรวจจับการทุจริตบัตรเครดิต"
SYSTEM_NAME_EN = "Credit Card Fraud Detection"

DEVELOPERS = [
    "039 ศักดิโชติ แตงโสภา",
    "041 สถาพร ขวาธิจักร",
    "044 อติชาต พันธุ์ขะวงษ์",
]

# ลำดับฟีเจอร์ที่โมเดลรับเข้า (30 ค่า) ตรงกับ creditcard.csv (ไม่รวมคอลัมน์ Class)
FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

# ใช้เป็นค่าสำรอง เมื่อไม่มีไฟล์ test_data.csv ในโฟลเดอร์
# ใส่ค่าจริงจากชุดทดสอบตอนเทรน เช่น 0.9994 (ถ้าเป็น None จะแสดง "—")
MODEL_METRICS = {
    "Accuracy": None,
    "Precision": None,
    "Recall": None,
    "F1-score": None,
    "ROC-AUC": None,
    "AUPRC": None,
}

# ------------------------------------------------------------------
# หน้าเพจ + สไตล์มินิมอล
# ------------------------------------------------------------------
st.set_page_config(page_title=SYSTEM_NAME, page_icon="🛡️", layout="centered")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Thai:wght@300;400;500;600&display=swap');
    :root {
        --ink: #1f2937; --muted: #6b7280; --line: #e5e7eb;
        --accent: #0f766e; --bad: #b91c1c; --bad-bg: #fef2f2;
        --ok: #0f766e; --ok-bg: #f0fdfa; --soft: #f9fafb;
    }
    html, body, [class*="css"], .stApp { font-family: 'Noto Sans Thai', sans-serif; color: var(--ink); }
    #MainMenu, footer { visibility: hidden; }
    .block-container { max-width: 820px; padding-top: 3rem; padding-bottom: 3rem; }

    .app-header { text-align: center; padding: 1rem 0 1.6rem; border-bottom: 1px solid var(--line); margin-bottom: 1.6rem; }
    .app-header h1 { font-size: 1.85rem; font-weight: 600; margin: 0; letter-spacing: -0.01em; }
    .app-header p { color: var(--muted); margin: .35rem 0 0; font-size: .95rem; font-weight: 300; }

    .section-title { font-size: 1.05rem; font-weight: 600; margin: 2rem 0 .8rem; }

    .result { border: 1px solid var(--line); border-radius: 14px; padding: 1.4rem; text-align: center; margin-top: 1rem; }
    .result.safe { background: var(--ok-bg); border-color: #99f6e4; }
    .result.fraud { background: var(--bad-bg); border-color: #fecaca; }
    .result .label { font-size: 1.15rem; font-weight: 600; }
    .result.safe .label, .result.safe .prob { color: var(--ok); }
    .result.fraud .label, .result.fraud .prob { color: var(--bad); }
    .result .prob { font-size: 2.6rem; font-weight: 600; line-height: 1.2; }
    .result .sub { color: var(--muted); font-size: .85rem; font-weight: 300; }

    div[data-testid="stMetric"] { background: var(--soft); border: 1px solid var(--line); border-radius: 12px; padding: .9rem 1rem; }
    div[data-testid="stMetricLabel"] { color: var(--muted); }
    .stButton > button, .stDownloadButton > button {
        background: var(--accent); color: white; border: 0; border-radius: 10px;
        padding: .55rem 1.4rem; font-weight: 500;
    }
    .stButton > button:hover, .stDownloadButton > button:hover { background: #115e59; color: white; }
    button[data-baseweb="tab"] { font-weight: 500; }

    .app-footer { margin-top: 3rem; padding-top: 1.2rem; border-top: 1px solid var(--line); text-align: center; color: var(--muted); font-size: .88rem; font-weight: 300; }
    .app-footer .names { color: var(--ink); font-weight: 400; margin-top: .3rem; line-height: 1.8; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------
# โหลดโมเดลและข้อมูล
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="กำลังโหลดโมเดล…")
def load_assets():
    import keras

    model = keras.models.load_model(MODEL_PATH, compile=False)
    scaler = None
    if SCALER_PATH.exists():
        import joblib

        scaler = joblib.load(SCALER_PATH)
    return model, scaler


def predict_proba(model, scaler, df: pd.DataFrame) -> np.ndarray:
    x = df[FEATURES].astype("float32").values
    if scaler is not None:
        x = scaler.transform(x)
    return model.predict(x, batch_size=1024, verbose=0).ravel()


@st.cache_data(show_spinner="กำลังประเมินโมเดลจากชุดทดสอบ…")
def load_test_predictions(_model, _scaler, file_mtime: float):
    """อ่าน test_data.csv แล้วทำนายทั้งไฟล์ (แคชไว้ ไม่คำนวณซ้ำทุกครั้งที่กดปุ่ม)"""
    df = pd.read_csv(TEST_PATH)
    if any(c not in df.columns for c in FEATURES) or "Class" not in df.columns:
        return None, None
    return df, predict_proba(_model, _scaler, df)


def compute_metrics(y_true: np.ndarray, prob: np.ndarray, thr: float):
    y_pred = (prob >= thr).astype(int)
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-score": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, prob) if len(np.unique(y_true)) > 1 else None,
        "AUPRC": average_precision_score(y_true, prob) if len(np.unique(y_true)) > 1 else None,
    }
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return metrics, cm


def fmt_pct(v):
    return "—" if v is None else f"{v * 100:.2f}%"


def render_result(p: float, thr: float):
    is_fraud = p >= thr
    cls = "fraud" if is_fraud else "safe"
    label = "น่าสงสัยว่าเป็นการทุจริต" if is_fraud else "รายการปกติ"
    st.markdown(
        f"""
        <div class="result {cls}">
            <div class="label">{label}</div>
            <div class="prob">{p * 100:.2f}%</div>
            <div class="sub">ความน่าจะเป็นที่เป็นการทุจริต (เกณฑ์ตัดสิน {thr:.2f})</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_cards(metrics: dict):
    items = list(metrics.items())
    for start in range(0, len(items), 3):
        cols = st.columns(3)
        for col, (name, val) in zip(cols, items[start:start + 3]):
            col.metric(name, fmt_pct(val))


def render_confusion(cm):
    cm_df = pd.DataFrame(
        cm,
        index=["จริง: ปกติ", "จริง: ทุจริต"],
        columns=["ทำนาย: ปกติ", "ทำนาย: ทุจริต"],
    )
    st.dataframe(cm_df, width="stretch")


# ------------------------------------------------------------------
# ส่วนหัว
# ------------------------------------------------------------------
st.markdown(
    f"""
    <div class="app-header">
        <h1>{SYSTEM_NAME}</h1>
        <p>{SYSTEM_NAME_EN} · โครงข่ายประสาทเทียมแบบ MLP</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not MODEL_PATH.exists():
    st.error(f"ไม่พบไฟล์โมเดล: {MODEL_PATH.name} กรุณาวางไว้โฟลเดอร์เดียวกับ app.py")
    st.stop()

try:
    model, scaler = load_assets()
except Exception as e:  # noqa: BLE001
    st.error(f"โหลดโมเดลไม่สำเร็จ: {e}")
    st.stop()

test_df, test_prob = None, None
if TEST_PATH.exists():
    try:
        test_df, test_prob = load_test_predictions(model, scaler, TEST_PATH.stat().st_mtime)
    except Exception as e:  # noqa: BLE001
        st.warning(f"อ่านไฟล์ test_data.csv ไม่สำเร็จ: {e}")

with st.expander("ตั้งค่าเกณฑ์ตัดสิน (ขั้นสูง)"):
    threshold = st.slider(
        "เกณฑ์ตัดสินว่าเป็นการทุจริต",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.05,
        help="ถ้าความน่าจะเป็นสูงกว่าหรือเท่ากับค่านี้ จะจัดว่าเป็นรายการทุจริต "
        "(ลดค่าลง = จับทุจริตได้มากขึ้น แต่แจ้งเตือนผิดมากขึ้น)",
    )

# ------------------------------------------------------------------
# ส่วนใช้งานหลัก
# ------------------------------------------------------------------
tab_single, tab_batch = st.tabs(["ตรวจสอบรายการเดียว", "ตรวจสอบหลายรายการ (CSV)"])

with tab_single:
    modes = ["กรอกทีละช่อง", "วางค่า 30 ตัว"]
    if test_df is not None:
        modes.append("เลือกจากชุดทดสอบ")
    mode = st.radio("วิธีกรอก", modes, horizontal=True, label_visibility="collapsed")

    values = None
    actual = None

    if mode == "กรอกทีละช่อง":
        st.caption("กรอกข้อมูลธุรกรรม ค่าที่ไม่ระบุจะเป็น 0")
        c1, c2 = st.columns(2)
        time_v = c1.number_input("Time (วินาที)", value=0.0, format="%.4f")
        amount_v = c2.number_input("Amount (จำนวนเงิน)", value=0.0, min_value=0.0, format="%.2f")
        with st.expander("ค่าคุณลักษณะ V1–V28"):
            cols = st.columns(4)
            vs = [
                cols[i % 4].number_input(f"V{i + 1}", value=0.0, format="%.4f", key=f"v{i + 1}")
                for i in range(28)
            ]
        values = [time_v] + vs + [amount_v]

    elif mode == "วางค่า 30 ตัว":
        st.caption("ลำดับ: Time, V1–V28, Amount (คั่นด้วยจุลภาคหรือเว้นวรรค)")
        raw = st.text_area("ค่า 30 ตัว", height=120, label_visibility="collapsed",
                           placeholder="0, -1.359, -0.072, 2.536, ... , 149.62")
        parts = [p for p in re.split(r"[,\s;]+", raw.strip()) if p]
        if parts:
            try:
                nums = [float(p) for p in parts]
                if len(nums) != len(FEATURES):
                    st.warning(f"ต้องมี {len(FEATURES)} ค่า (ตอนนี้มี {len(nums)} ค่า)")
                else:
                    values = nums
            except ValueError:
                st.warning("พบค่าที่ไม่ใช่ตัวเลข กรุณาตรวจสอบอีกครั้ง")

    else:  # เลือกจากชุดทดสอบ
        st.caption("เลือกรายการจากชุดทดสอบ แล้วดูว่าโมเดลทำนายตรงกับค่าจริงหรือไม่")
        kind = st.radio("ประเภท", ["ทั้งหมด", "ปกติ", "ทุจริต"], horizontal=True)
        pool = test_df if kind == "ทั้งหมด" else test_df[test_df["Class"] == (1 if kind == "ทุจริต" else 0)]
        if len(pool) == 0:
            st.warning("ไม่มีรายการประเภทนี้ในชุดทดสอบ")
        else:
            idx = st.number_input(f"ลำดับรายการ (1–{len(pool):,})", min_value=1, max_value=len(pool), value=1, step=1)
            row = pool.iloc[int(idx) - 1]
            values = row[FEATURES].astype(float).tolist()
            actual = int(row["Class"])
            st.caption(f"Time = {row['Time']:.0f} · Amount = {row['Amount']:.2f}")

    if st.button("ตรวจสอบ", key="predict_one", disabled=values is None):
        one = pd.DataFrame([values], columns=FEATURES)
        prob = float(predict_proba(model, scaler, one)[0])
        render_result(prob, threshold)
        if actual is not None:
            pred = int(prob >= threshold)
            truth = "ทุจริต" if actual == 1 else "ปกติ"
            if pred == actual:
                st.success(f"ค่าจริงในไฟล์: {truth} — โมเดลทำนายถูกต้อง")
            else:
                st.error(f"ค่าจริงในไฟล์: {truth} — โมเดลทำนายคลาดเคลื่อน")

with tab_batch:
    st.caption(
        "อัปโหลดไฟล์ CSV ที่มีคอลัมน์ Time, V1–V28, Amount "
        "(ถ้ามีคอลัมน์ Class ระบบจะคำนวณความแม่นยำจากไฟล์นั้นให้ด้วย)"
    )
    up = st.file_uploader("เลือกไฟล์ CSV", type=["csv"], label_visibility="collapsed")

    if up is not None:
        try:
            df = pd.read_csv(up)
        except Exception as e:  # noqa: BLE001
            st.error(f"อ่านไฟล์ไม่สำเร็จ: {e}")
            df = None

        if df is not None:
            missing = [c for c in FEATURES if c not in df.columns]
            if missing:
                st.error("คอลัมน์ไม่ครบ ขาด: " + ", ".join(missing))
            else:
                with st.spinner("กำลังประมวลผล…"):
                    prob = predict_proba(model, scaler, df)
                out = df.copy()
                out["fraud_probability"] = prob
                out["prediction"] = np.where(prob >= threshold, "Fraud", "Normal")

                n_fraud = int((prob >= threshold).sum())
                c1, c2, c3 = st.columns(3)
                c1.metric("จำนวนรายการ", f"{len(out):,}")
                c2.metric("น่าสงสัยทุจริต", f"{n_fraud:,}")
                c3.metric("สัดส่วนทุจริต", f"{n_fraud / max(len(out), 1) * 100:.2f}%")

                first = ["fraud_probability", "prediction"]
                show_cols = first + [c for c in out.columns if c not in first]
                st.dataframe(out[show_cols].head(500), width="stretch", height=300)
                st.download_button(
                    "ดาวน์โหลดผลลัพธ์ (CSV)",
                    out.to_csv(index=False).encode("utf-8-sig"),
                    file_name="fraud_predictions.csv",
                    mime="text/csv",
                )

                if "Class" in df.columns:
                    y_true = df["Class"].astype(int).values
                    m, cm = compute_metrics(y_true, prob, threshold)
                    st.session_state["live_metrics"] = {"metrics": m, "cm": cm, "n": len(df), "thr": threshold}

# ------------------------------------------------------------------
# ความแม่นยำของระบบ (ด้านล่าง)
# ------------------------------------------------------------------
st.markdown('<div class="section-title">ความแม่นยำของระบบ</div>', unsafe_allow_html=True)

if test_df is not None:
    m, cm = compute_metrics(test_df["Class"].astype(int).values, test_prob, threshold)
    st.caption(f"ประเมินจากชุดทดสอบ {len(test_df):,} รายการ (เกณฑ์ตัดสิน {threshold:.2f})")
    render_metric_cards(m)
    render_confusion(cm)
elif any(v is not None for v in MODEL_METRICS.values()):
    st.caption("ค่าประเมินจากชุดข้อมูลทดสอบตอนพัฒนาโมเดล")
    render_metric_cards(MODEL_METRICS)
else:
    st.info(
        "ยังไม่มีค่าความแม่นยำ — วางไฟล์ test_data.csv (ข้อมูลทดสอบที่มีคอลัมน์ Class) "
        "ไว้ข้าง app.py ระบบจะคำนวณให้อัตโนมัติ หรืออัปโหลดไฟล์ในแท็บ CSV ด้านบน"
    )

live = st.session_state.get("live_metrics")
if live:
    st.caption(f"ผลวัดจากไฟล์ที่อัปโหลด ({live['n']:,} รายการ, เกณฑ์ {live['thr']:.2f})")
    render_metric_cards(live["metrics"])
    render_confusion(live["cm"])

st.caption(
    "หมายเหตุ: ข้อมูลทุจริตมีสัดส่วนน้อยมาก (ราว 0.17%) ค่า Accuracy อย่างเดียวอาจสูงเกินจริง "
    "ผู้จัดทำชุดข้อมูลแนะนำให้ใช้ AUPRC ร่วมกับ Precision, Recall และ F1-score"
)

# ------------------------------------------------------------------
# ท้ายหน้า: ผู้พัฒนา
# ------------------------------------------------------------------
st.markdown(
    f"""
    <div class="app-footer">
        ผู้พัฒนา
        <div class="names">{'<br>'.join(DEVELOPERS)}</div>
    </div>
    """,
    unsafe_allow_html=True,
)
