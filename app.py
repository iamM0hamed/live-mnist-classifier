import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np
import streamlit as st
from keras.saving import load_model
from streamlit_drawable_canvas import st_canvas

from src.drawing_preprocessing import preprocess_drawing

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="🔢",
    layout="centered",
)


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "mnist_cnn.keras"


# --------------------------------------------------
# Custom CSS
# --------------------------------------------------

st.markdown(
    """
    <style>

        .main-title {
            text-align: center;
            font-size: 2.8rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            text-align: center;
            color: #888888;
            font-size: 1.05rem;
            margin-bottom: 2rem;
        }

        .section-title {
            font-size: 1.35rem;
            font-weight: 600;
            margin-bottom: 0.7rem;
        }

        .prediction-box {
            padding: 1.5rem;
            border-radius: 1rem;
            border: 1px solid rgba(128, 128, 128, 0.25);
            text-align: center;
            margin-top: 1rem;
            margin-bottom: 1.5rem;
        }

        .prediction-label {
            font-size: 1rem;
            color: #888888;
            margin-bottom: 0.3rem;
        }

        .prediction-digit {
            font-size: 5rem;
            font-weight: 800;
            line-height: 1;
        }

        .confidence {
            font-size: 1.15rem;
            margin-top: 0.7rem;
        }

        .footer {
            text-align: center;
            color: #888888;
            font-size: 0.85rem;
            margin-top: 3rem;
            padding-top: 1rem;
            border-top: 1px solid rgba(128, 128, 128, 0.2);
        }

    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Model
# --------------------------------------------------


@st.cache_resource
def load_mnist_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file was not found at: {MODEL_PATH}")

    return load_model(MODEL_PATH)


try:
    model = load_mnist_model()

except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

except Exception as error:
    st.error("Failed to load the model.")
    st.exception(error)
    st.stop()


# --------------------------------------------------
# Session state
# --------------------------------------------------

if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0


# --------------------------------------------------
# Header
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🔢 MNIST Digit Classifier</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Draw a handwritten digit and let a CNN classify it."
    "</div>",
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.header("About")

    st.write("""
        This application uses a Convolutional Neural Network
        trained on the MNIST handwritten digit dataset.
        """)

    st.divider()

    st.subheader("Model")

    st.write("**Input:** `28 × 28 × 1`")
    st.write("**Classes:** `0 – 9`")
    st.write("**Architecture:** CNN")
    st.write("**Framework:** Keras")

    st.divider()

    st.caption(
        "The drawing is converted into the same format used "
        "during MNIST training before prediction."
    )


# --------------------------------------------------
# Main layout
# --------------------------------------------------

draw_col, result_col = st.columns([1.1, 1])


# ==================================================
# Drawing section
# ==================================================

with draw_col:

    st.markdown(
        '<div class="section-title">Draw a digit</div>',
        unsafe_allow_html=True,
    )

    st.write("Draw one digit in the canvas below.")

    if st.button("🗑️ Clear", use_container_width=True):

        st.session_state.canvas_key += 1
        st.rerun()

    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=19,
        stroke_color="white",
        background_color="black",
        width=450,
        height=450,
        drawing_mode="freedraw",
        key=f"canvas_{st.session_state.canvas_key}",
        update_streamlit=True,
        return_image_data=True,
    )


# ==================================================
# Result section
# ==================================================

with result_col:

    st.markdown(
        '<div class="section-title">Prediction</div>',
        unsafe_allow_html=True,
    )

    if canvas_result.image_data is not None:

        processed_image = preprocess_drawing(canvas_result.image_data)

        # ------------------------------------------
        # Empty canvas
        # ------------------------------------------

        if np.max(processed_image) == 0:

            st.info("✏️ Draw a digit first to get a prediction.")

        # ------------------------------------------
        # Prediction
        # ------------------------------------------

        else:

            predictions = model.predict(
                processed_image,
                verbose=0,
            )

            predicted_digit = int(np.argmax(predictions[0]))

            confidence = float(predictions[0][predicted_digit])

            # --------------------------------------
            # Main prediction
            # --------------------------------------

            st.markdown(
                f"""
                <div class="prediction-box">

                    <div class="prediction-label">
                        Predicted Digit
                    </div>

                    <div class="prediction-digit">
                        {predicted_digit}
                    </div>

                    <div class="confidence">
                        Confidence:
                        <strong>{confidence:.2%}</strong>
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

            # --------------------------------------
            # Processed image
            # --------------------------------------

            st.subheader("Processed Image")

            st.image(
                processed_image[0, :, :, 0],
                width=180,
                clamp=True,
            )

            st.caption("28 × 28 image sent to the CNN.")

            # --------------------------------------
            # Probabilities
            # --------------------------------------

            st.subheader("Class Probabilities")

            for digit, probability in enumerate(predictions[0]):

                probability = float(probability)

                st.progress(
                    probability,
                    text=f"{digit}: {probability:.2%}",
                )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.markdown(
    """
    <div class="footer">
        MNIST Digit Classifier · Built with Keras & Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
