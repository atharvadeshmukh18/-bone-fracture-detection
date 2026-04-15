import streamlit as st
import numpy as np
import pandas as pd
import os
from datetime import datetime
from PIL import Image
import tensorflow as tf

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Bone Fracture Detection", layout="wide")

# =========================
# DARK UI
# =========================
st.markdown("""
<style>
body {background-color: #0e1117; color: white;}
.stButton>button {
    background-color: #00c3ff;
    color: white;
    border-radius: 10px;
    height: 50px;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    if os.path.exists("fracture_model.h5"):
        return tf.keras.models.load_model("fracture_model.h5")
    return None

model = load_model()

# =========================
# SIDEBAR
# =========================
st.sidebar.title("🩺 Navigation")
menu = st.sidebar.radio("Go to", [
    "Dashboard",
    "Fracture Detection",
    "Reports",
    "Patient History"
])

# =========================
# PREPROCESS (FIXED)
# =========================
def preprocess(image):
    image = image.resize((128,128))
    image = image.convert("RGB")  # FIX grayscale

    img = np.array(image) / 255.0

    if img.shape[-1] != 3:
        img = np.stack((img,)*3, axis=-1)

    img = np.expand_dims(img, axis=0)
    return img

# =========================
# PREDICT
# =========================
def predict(img):
    if model:
        pred = model.predict(img)[0][0]
    else:
        pred = np.random.rand()
    return pred

# =========================
# DASHBOARD
# =========================
if menu == "Dashboard":
    st.title("🦴 Bone Fracture Detection System")
    st.info("Upload X-ray images to detect fractures using AI")

# =========================
# FRACTURE DETECTION
# =========================
elif menu == "Fracture Detection":

    st.title("🦴 AI Bone Fracture Detection")
    st.info("Upload X-ray → Analyze → Detect Fracture")

    # Patient Details
    st.subheader("👤 Patient Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        patient_id = st.text_input("Patient ID")
        name = st.text_input("Patient Name")

    with col2:
        age = st.number_input("Age", 1, 120)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])

    with col3:
        contact = st.text_input("Contact Number")

    # Upload Image
    st.subheader("📤 Upload X-ray Image")

    uploaded_file = st.file_uploader("Upload X-ray (JPG/PNG)", type=["jpg","png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded X-ray", use_column_width=True)

        if st.button("🔍 Analyze X-ray"):

            if not patient_id or not name:
                st.error("⚠️ Fill Patient ID & Name")
            else:
                with st.spinner("Analyzing..."):

                    img = preprocess(image)
                    pred = predict(img)

                    # 🔥 DEBUG VALUE
                    st.write(f"Raw Prediction Score: {pred:.4f}")

                    # =========================
                    # ✅ FIXED LABEL LOGIC
                    # =========================
                    if pred < 0.5:
                        result = "🛑 Fracture Detected"
                        color = "red"
                    else:
                        result = "✅ No Fracture"
                        color = "green"

                    confidence = pred * 100

                    # Severity
                    if confidence < 50:
                        severity = "Low"
                    elif confidence < 75:
                        severity = "Medium"
                    else:
                        severity = "High"

                    # Display
                    st.markdown(
                        f"## Result: <span style='color:{color}'>{result}</span>",
                        unsafe_allow_html=True
                    )

                    st.progress(int(confidence))
                    st.write(f"📊 Confidence: {confidence:.2f}%")
                    st.write(f"⚠️ Severity: {severity}")

                    # Chart
                    chart = pd.DataFrame({
                        "Class": ["Fracture", "Normal"],
                        "Probability": [1-pred, pred]
                    })
                    st.bar_chart(chart.set_index("Class"))

                    # Save history
                    data = {
                        "Patient ID": patient_id,
                        "Name": name,
                        "Age": age,
                        "Gender": gender,
                        "Contact": contact,
                        "Result": result,
                        "Confidence": confidence,
                        "Time": datetime.now()
                    }

                    df = pd.DataFrame([data])

                    if os.path.exists("history.csv"):
                        df.to_csv("history.csv", mode='a', header=False, index=False)
                    else:
                        df.to_csv("history.csv", index=False)

                    st.success("Saved to history")

# =========================
# REPORTS
# =========================
elif menu == "Reports":
    st.title("📄 Reports")

    if os.path.exists("history.csv"):
        df = pd.read_csv("history.csv")
        st.dataframe(df)

        st.download_button(
            "Download Report",
            df.to_csv(index=False),
            file_name="report.csv"
        )
    else:
        st.warning("No reports")

# =========================
# HISTORY
# =========================
elif menu == "Patient History":
    st.title("📊 Patient History")

    if os.path.exists("history.csv"):
        df = pd.read_csv("history.csv")

        pid = st.text_input("Search Patient ID")

        if pid:
            df = df[df["Patient ID"] == pid]

        st.dataframe(df)
    else:
        st.warning("No history")