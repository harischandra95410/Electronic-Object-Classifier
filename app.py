from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATHS = (
    PROJECT_DIR / "models" / "seed_42" / "dual_stream_model.keras",
    PROJECT_DIR / "models" / "seed_100" / "dual_stream_model.keras",
    PROJECT_DIR / "dual_stream_model.keras",
)

st.set_page_config(
    page_title="Electronic Object Classifier",
    page_icon="📷",
)
st.title("Electronic Object Classifier")
st.write("Upload JPG or PNG images to view the original, Canny edge map, and Conv1 activation.")

def edge_map(rgb_uint8):
    import cv2

    gray = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    magnitude_uint8 = np.clip(magnitude, 0, 255).astype(np.uint8)
    threshold, _ = cv2.threshold(
        magnitude_uint8,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    high_threshold = float(threshold)
    low_threshold = 0.5 * high_threshold
    if high_threshold < 1:
        low_threshold, high_threshold = 30, 120
    return cv2.Canny(blurred, low_threshold, high_threshold)


def conv1_activation(model, rgb_model_input, original_rgb):
    import cv2

    pre_stem_normalization = model.get_layer(
        "convnext_tiny_prestem_normalization"
    )
    conv1 = model.get_layer("convnext_tiny_stem").get_layer(
        "convnext_tiny_stem_conv"
    )
    activations = conv1(pre_stem_normalization(rgb_model_input)).numpy()[0]
    heatmap = np.sqrt(np.mean(np.square(activations), axis=-1))
    low, high = np.percentile(heatmap, (5, 99))
    if high <= low:
        raise ValueError("Conv1 activations do not contain a visible range.")
    heatmap = np.clip((heatmap - low) / (high - low), 0, 1) ** 0.65
    height, width = original_rgb.shape[:2]
    heatmap = cv2.resize(
        heatmap.astype(np.float32),
        (width, height),
        interpolation=cv2.INTER_CUBIC,
    )
    heatmap = np.clip(heatmap, 0, 1)
    colored = cv2.applyColorMap(
        np.round(heatmap * 255).astype(np.uint8),
        cv2.COLORMAP_TURBO,
    )
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(original_rgb, 0.4, colored, 0.6, 0)


@st.cache_resource
def load_models(model_paths: tuple[str, ...]):
    import tensorflow as tf

    tf.keras.layers.Dense.from_config = classmethod(
        lambda cls, config: cls(
            **{
                key: value
                for key, value in config.items()
                if key != "quantization_config"
            }
        )
    )
    return tuple(
        tf.keras.models.load_model(
            model_path,
            compile=False,
            safe_mode=False,
        )
        for model_path in model_paths
    )


uploaded_images = st.file_uploader(
    "Choose one or more images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

if uploaded_images:
    missing_models = [path for path in MODEL_PATHS if not path.is_file()]
    if missing_models:
        st.error(
            "Trained ensemble checkpoint(s) are missing: "
            + ", ".join(str(path) for path in missing_models)
        )
    else:
        import cv2
        import tensorflow as tf

        try:
            models = load_models(tuple(str(path) for path in MODEL_PATHS))
            if any(len(model.inputs) != 2 for model in models):
                raise ValueError(
                    "This app expects three notebook-trained, two-input "
                    "Dual-Stream models."
                )
        except (OSError, ValueError) as error:
            st.error(f"Could not load the trained ensemble: {error}")
        else:
            for uploaded_image in uploaded_images:
                try:
                    image_bytes = uploaded_image.getvalue()
                    original_bgr = cv2.imdecode(
                        np.frombuffer(image_bytes, dtype=np.uint8),
                        cv2.IMREAD_COLOR,
                    )
                    if original_bgr is None:
                        raise ValueError("The uploaded file is not a valid image.")
                    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
                    resized_rgb = cv2.resize(
                        original_rgb,
                        (224, 224),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    display_edges = edge_map(original_rgb)
                    model_edges = edge_map(resized_rgb)
                    edge_input = np.repeat(
                        model_edges[..., np.newaxis],
                        3,
                        axis=-1,
                    ).astype(np.float32) / 255.0

                    image = tf.convert_to_tensor(resized_rgb, dtype=tf.float32)
                    rgb_input = tf.keras.applications.convnext.preprocess_input(
                        image
                    )
                    rgb_batch = tf.expand_dims(rgb_input, 0)
                    edge_batch = edge_input[np.newaxis, ...]
                    probabilities = np.mean(
                        [
                            model.predict(
                                [rgb_batch, edge_batch],
                                verbose=0,
                            )[0]
                            for model in models
                        ],
                        axis=0,
                    )
                    confidence = float(np.max(probabilities))
                    activation = conv1_activation(
                        models[-1],
                        rgb_batch,
                        original_rgb,
                    )

                    original = st.columns(3)
                    with original[0]:
                        st.markdown(
                            "<div style='text-align:center'><strong>Original</strong></div>",
                            unsafe_allow_html=True,
                        )
                        st.image(original_rgb, channels="RGB", width="stretch")
                    with original[1]:
                        st.markdown(
                            "<div style='text-align:center'><strong>Canny Edge Map (original size)</strong></div>",
                            unsafe_allow_html=True,
                        )
                        st.image(display_edges, channels="GRAY", width="stretch")
                    with original[2]:
                        st.markdown(
                            "<div style='text-align:center'><strong>Conv1 Activations</strong></div>",
                            unsafe_allow_html=True,
                        )
                        st.image(activation, channels="RGB", width="stretch")

                    metrics = st.columns(2)
                    metrics[0].metric("Ensemble confidence", f"{confidence:.1%}")
                    metrics[1].metric("Mean test accuracy (3 seeds)", "94.40%")
                    st.caption(
                        "Confidence averages the three models' predicted probabilities. "
                        "Mean test accuracy averages their individual results on 143 "
                        "held-out images per seed."
                    )
                    st.divider()
                except (OSError, ValueError, tf.errors.InvalidArgumentError) as error:
                    st.error(f"Could not classify {uploaded_image.name}: {error}")
