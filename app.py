import io
import os
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.inference import (
    get_model,
    predict_image,
    generate_gradcam,
)
from src.storage import append_prediction, load_history


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        BASE_DIR / "fracture_model.h5"
    )
)


# ============================================================
# STREAMLIT PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="MedRay AI | Bone Fracture Detection",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background: #0b1020;
    }

    .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }

    .hero {
        padding: 1.5rem;
        border-radius: 18px;
        background: linear-gradient(
            135deg,
            #111827,
            #172554
        );
        margin-bottom: 1rem;
    }

    .metric-card {
        padding: 1rem;
        border: 1px solid #26324a;
        border-radius: 14px;
        background: #111827;
    }

    .disclaimer {
        padding: .9rem 1rem;
        border-left: 4px solid #f59e0b;
        background: #1f2937;
        border-radius: 8px;
    }

    .heatmap-info {
        padding: 1rem;
        border-radius: 12px;
        background: #111827;
        border: 1px solid #26324a;
        margin-top: 1rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

def render_header():

    st.markdown(
        """
        <div class="hero">

          <h1>🦴 MedRay AI</h1>

          <p style="font-size:1.05rem">
            AI-assisted bone fracture screening
            from X-ray images.
          </p>

          <p style="opacity:.8">
            TensorFlow • Streamlit • Docker • CI/CD • Explainable AI
          </p>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# DASHBOARD
# ============================================================

def dashboard():

    render_header()

    history = load_history()

    total = len(history)

    fractures = (
        int(
            (
                history["result"]
                == "Fracture Detected"
            ).sum()
        )
        if total
        else 0
    )

    normal = total - fractures

    c1, c2, c3 = st.columns(3)

    for col, label, value in [
        (c1, "Analyses", total),
        (c2, "Fracture detections", fractures),
        (c3, "No fracture", normal),
    ]:

        with col:

            st.markdown(
                f"""
                <div class="metric-card">
                    <b>{label}</b>
                    <h2>{value}</h2>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()

    st.subheader("How it works")

    st.write(
        """
        1. Upload a JPG/PNG X-ray.
        
        2. The image is resized to the model input size.
        
        3. The trained binary CNN classifier returns a probability.
        
        4. Grad-CAM generates an AI attention map.
        
        5. The prediction is recorded locally for this demo instance.
        """
    )

    st.markdown(
        """
        <div class="disclaimer">

        ⚠️ <b>Research/demo only:</b>
        This application is not a medical device and must not
        be used to diagnose or treat patients.

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# FRACTURE DETECTION
# ============================================================

def detection():

    render_header()

    st.subheader("Patient / Case Details")

    # --------------------------------------------------------
    # Patient information
    # --------------------------------------------------------

    c1, c2, c3 = st.columns(3)

    with c1:

        patient_id = st.text_input(
            "Case ID",
            placeholder="CASE-001",
        )

    with c2:

        age = st.number_input(
            "Age",
            min_value=0,
            max_value=120,
            value=25,
        )

    with c3:

        gender = st.selectbox(
            "Sex",
            [
                "Female",
                "Male",
                "Other",
                "Prefer not to say",
            ],
        )

    # --------------------------------------------------------
    # Upload X-ray
    # --------------------------------------------------------

    uploaded = st.file_uploader(
        "Upload X-ray",
        type=[
            "jpg",
            "jpeg",
            "png",
        ],
    )

    if uploaded:

        image = Image.open(
            uploaded
        ).convert("RGB")

        st.divider()

        st.subheader(
            "🩻 Uploaded X-ray"
        )

        st.image(
            image,
            caption="Original X-ray",
            width="stretch",
        )

        # ----------------------------------------------------
        # Analyze button
        # ----------------------------------------------------

        if st.button(
            "🔍 Analyze X-ray",
            type="primary",
            use_container_width=True,
        ):

            # ------------------------------------------------
            # Validate Case ID
            # ------------------------------------------------

            if not patient_id.strip():

                st.error(
                    "Enter a Case ID before analysis."
                )

                return

            # ------------------------------------------------
            # Load model and perform prediction
            # ------------------------------------------------

            try:

                with st.spinner(
                    "Loading AI model..."
                ):

                    model = get_model(
                        MODEL_PATH
                    )

                with st.spinner(
                    "Analyzing X-ray..."
                ):

                    result, confidence, raw_score = (
                        predict_image(
                            model,
                            image,
                        )
                    )

            except FileNotFoundError:

                st.error(
                    f"""
                    Model file not found:

                    `{MODEL_PATH}`

                    Copy your existing
                    `fracture_model.h5`
                    into the project root or set
                    the MODEL_PATH environment variable.
                    """
                )

                return

            except Exception as exc:

                st.error(
                    f"Model inference failed: {exc}"
                )

                return

            # ------------------------------------------------
            # Generate Grad-CAM
            # ------------------------------------------------

            try:

                with st.spinner(
                    "Generating AI attention heatmap..."
                ):

                    heatmap = generate_gradcam(
                        model,
                        image,
                    )

            except Exception as exc:

                st.warning(
                    f"""
                    Prediction completed, but the
                    Grad-CAM heatmap could not be generated.

                    Error:
                    {exc}
                    """
                )

                heatmap = None

            # ------------------------------------------------
            # Save prediction
            # ------------------------------------------------

            prediction_record = {

                "case_id": patient_id.strip(),

                "age": age,

                "gender": gender,

                "result": result,

                "confidence": confidence,

                "raw_score": raw_score,

                "timestamp": datetime.now(
                    timezone.utc
                ).isoformat(),
            }

            st.session_state[
                "last_prediction"
            ] = prediction_record

            append_prediction(
                prediction_record
            )

            # ------------------------------------------------
            # Prediction result
            # ------------------------------------------------

            st.divider()

            st.subheader(
                "📊 Prediction Result"
            )

            if result == "Fracture Detected":

                st.error(
                    f"🛑 {result}"
                )

            else:

                st.success(
                    f"✅ {result}"
                )

            # ------------------------------------------------
            # Confidence
            # ------------------------------------------------

            c1, c2 = st.columns(2)

            with c1:

                st.metric(
                    "Model Confidence",
                    f"{confidence:.2f}%",
                )

            with c2:

                st.metric(
                    "Raw Model Score",
                    f"{raw_score:.4f}",
                )

            # ------------------------------------------------
            # Probability chart
            # ------------------------------------------------

            st.subheader(
                "Prediction Probability"
            )

            chart = pd.DataFrame(
                {
                    "Class": [
                        "Fracture",
                        "No fracture",
                    ],

                    "Probability": [
                        1 - raw_score,
                        raw_score,
                    ],
                }
            )

            st.bar_chart(
                chart.set_index(
                    "Class"
                )
            )

            # ------------------------------------------------
            # Grad-CAM
            # ------------------------------------------------

            if heatmap is not None:

                st.divider()

                st.subheader(
                    "🔥 AI Attention / Grad-CAM"
                )

                st.markdown(
                    """
                    <div class="heatmap-info">

                    <b>What does this heatmap show?</b>

                    <br><br>

                    The highlighted regions represent areas
                    that contributed more strongly to the CNN's
                    prediction.

                    <br><br>

                    🔴 <b>Red / Yellow</b> → stronger model attention

                    <br>

                    🔵 <b>Blue</b> → lower model attention

                    <br><br>

                    ⚠️ The heatmap represents model attention
                    and does <b>not</b> provide an exact fracture
                    boundary or clinical diagnosis.

                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.write("")

                # --------------------------------------------
                # Original vs Heatmap
                # --------------------------------------------

                col1, col2 = st.columns(2)

                with col1:

                    st.image(
                        image,
                        caption="Original X-ray",
                        width="stretch",
                    )

                with col2:

                    st.image(
                        heatmap,
                        caption="Grad-CAM AI Attention",
                        width="stretch",
                    )

                # --------------------------------------------
                # Download heatmap
                # --------------------------------------------

                heatmap_buffer = (
                    io.BytesIO()
                )

                heatmap.save(
                    heatmap_buffer,
                    format="PNG",
                )

                st.download_button(
                    label="⬇️ Download Grad-CAM Heatmap",

                    data=(
                        heatmap_buffer.getvalue()
                    ),

                    file_name=(
                        f"gradcam_"
                        f"{patient_id.strip()}.png"
                    ),

                    mime="image/png",

                    use_container_width=True,
                )

            # ------------------------------------------------
            # Final disclaimer
            # ------------------------------------------------

            st.info(
                """
                ⚠️ This result and Grad-CAM visualization
                are AI research/demo outputs and are not
                a clinical diagnosis. Consult a qualified
                medical professional for interpretation.
                """
            )


# ============================================================
# REPORTS
# ============================================================

def reports():

    render_header()

    st.subheader(
        "Prediction Reports"
    )

    history = load_history()

    if history.empty:

        st.info(
            "No analyses have been recorded yet."
        )

        return

    st.dataframe(
        history,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "⬇️ Download CSV report",

        history.to_csv(
            index=False
        ),

        "bone_fracture_reports.csv",

        "text/csv",
    )


# ============================================================
# CASE HISTORY
# ============================================================

def history_page():

    render_header()

    st.subheader(
        "Case History"
    )

    history = load_history()

    if history.empty:

        st.info(
            "No case history available."
        )

        return

    case_id = st.text_input(
        "Search Case ID"
    )

    if case_id.strip():

        history = history[
            history["case_id"]
            .astype(str)
            .str.contains(
                case_id.strip(),
                case=False,
                na=False,
            )
        ]

    st.dataframe(
        history,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:

    st.title(
        "🩺 Navigation"
    )

    menu = st.radio(
        "Go to",
        [
            "Dashboard",
            "Fracture Detection",
            "Reports",
            "Case History",
        ],
    )

    st.divider()

    st.caption(
        "MedRay AI • Bone Fracture Detection"
    )


# ============================================================
# PAGE ROUTING
# ============================================================

if menu == "Dashboard":

    dashboard()

elif menu == "Fracture Detection":

    detection()

elif menu == "Reports":

    reports()

else:

    history_page()