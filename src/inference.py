from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

INPUT_SIZE = (128, 128)


def get_model(model_path: Path):
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(model_path)

    import tensorflow as tf

    return tf.keras.models.load_model(model_path, compile=False)


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(INPUT_SIZE)

    array = np.asarray(image, dtype=np.float32) / 255.0

    return np.expand_dims(array, axis=0)


def predict_image(model, image: Image.Image) -> Tuple[str, float, float]:
    batch = preprocess_image(image)

    raw = float(
        np.asarray(
            model.predict(batch, verbose=0)
        ).reshape(-1)[0]
    )

    raw = float(np.clip(raw, 0.0, 1.0))

    # Existing project logic:
    # raw < 0.5  -> Fracture
    # raw >= 0.5 -> No Fracture
    result = "Fracture Detected" if raw < 0.5 else "No Fracture"

    confidence = (
        (1.0 - raw) * 100
        if result == "Fracture Detected"
        else raw * 100
    )

    return result, confidence, raw


def _find_last_conv_layer(model):
    """
    Find the last convolutional layer in the model.
    """

    import tensorflow as tf

    # Prefer an actual Conv2D layer.
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer

    raise ValueError(
        "No Conv2D layer was found in the loaded model. "
        "Grad-CAM cannot be generated."
    )


def generate_gradcam(
    model,
    image: Image.Image,
    target_class: str | None = None,
) -> Image.Image:
    """
    Generate a Grad-CAM heatmap and overlay it on the original X-ray.

    Parameters
    ----------
    model:
        Loaded TensorFlow/Keras model.

    image:
        Original PIL image.

    target_class:
        "fracture" or "no_fracture".
        If omitted, the model's predicted class is used.

    Returns
    -------
    PIL.Image.Image
        Original X-ray with Grad-CAM overlay.
    """

    import tensorflow as tf
    import matplotlib.cm as cm

    # -----------------------------
    # Prepare image
    # -----------------------------

    batch = preprocess_image(image)

    # -----------------------------
    # Determine prediction
    # -----------------------------

    prediction = float(
        np.asarray(
            model.predict(batch, verbose=0)
        ).reshape(-1)[0]
    )

    prediction = float(np.clip(prediction, 0.0, 1.0))

    predicted_class = (
        "fracture"
        if prediction < 0.5
        else "no_fracture"
    )

    if target_class is None:
        target_class = predicted_class

    target_class = target_class.lower()

    if target_class not in {"fracture", "no_fracture"}:
        raise ValueError(
            "target_class must be 'fracture' or 'no_fracture'."
        )

    # -----------------------------
    # Find last convolutional layer
    # -----------------------------

    last_conv_layer = _find_last_conv_layer(model)

    # Model that returns:
    # 1. feature maps from last Conv2D
    # 2. final model prediction
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            last_conv_layer.output,
            model.output,
        ],
    )

    # -----------------------------
    # Calculate gradients
    # -----------------------------

    input_tensor = tf.convert_to_tensor(batch)

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(
            input_tensor,
            training=False,
        )

        raw_prediction = predictions[:, 0]

        # IMPORTANT:
        #
        # Your model uses:
        # raw < 0.5 -> Fracture
        #
        # Therefore:
        #
        # fracture score = -raw
        # no-fracture score = raw
        #
        # This lets Grad-CAM highlight regions
        # that contribute toward the selected class.

        if target_class == "fracture":
            target_score = -raw_prediction
        else:
            target_score = raw_prediction

    # Gradient of target score with respect to feature maps
    gradients = tape.gradient(
        target_score,
        conv_outputs,
    )

    # -----------------------------
    # Global average pooling
    # -----------------------------

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(1, 2),
    )

    conv_outputs = conv_outputs[0]
    pooled_gradients = pooled_gradients[0]

    # Weight each feature map
    heatmap = tf.reduce_sum(
        conv_outputs * pooled_gradients,
        axis=-1,
    )

    # ReLU
    heatmap = tf.maximum(heatmap, 0)

    # Normalize
    max_value = tf.reduce_max(heatmap)

    if float(max_value) > 0:
        heatmap = heatmap / max_value

    heatmap = heatmap.numpy()

    # -----------------------------
    # Convert heatmap to image
    # -----------------------------

    heatmap_uint8 = np.uint8(
        heatmap * 255
    )

    heatmap_image = Image.fromarray(
        heatmap_uint8
    ).resize(
        image.convert("RGB").size,
        Image.Resampling.BILINEAR,
    )

    # -----------------------------
    # Apply color map
    # -----------------------------

    heatmap_array = np.asarray(
        heatmap_image,
        dtype=np.float32,
    ) / 255.0

    colored_heatmap = cm.jet(
        heatmap_array
    )[:, :, :3]

    colored_heatmap = np.uint8(
        colored_heatmap * 255
    )

    colored_heatmap = Image.fromarray(
        colored_heatmap
    ).convert("RGB")

    # -----------------------------
    # Overlay on original X-ray
    # -----------------------------

    original = image.convert("RGB")

    overlay = Image.blend(
        original,
        colored_heatmap,
        alpha=0.40,
    )

    return overlay