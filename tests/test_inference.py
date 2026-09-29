import numpy as np
from PIL import Image

from src.inference import preprocess_image, predict_image


class DummyModel:
    def __init__(self, value):
        self.value = value

    def predict(self, batch, verbose=0):
        assert batch.shape == (1, 128, 128, 3)
        assert batch.dtype == np.float32
        return np.array([[self.value]], dtype=np.float32)


def test_preprocess_shape_and_range():
    image = Image.new("L", (300, 200), color=128)
    result = preprocess_image(image)
    assert result.shape == (1, 128, 128, 3)
    assert 0 <= result.min() <= result.max() <= 1


def test_fracture_label_uses_existing_threshold():
    result, confidence, raw = predict_image(DummyModel(0.2), Image.new("RGB", (20, 20)))
    assert result == "Fracture Detected"
    assert abs(raw - 0.2) < 1e-4
    assert abs(confidence - 80.0) < 1e-4


def test_normal_label():
    result, confidence, raw = predict_image(DummyModel(0.8), Image.new("RGB", (20, 20)))
    assert result == "No Fracture"
    assert abs(raw - 0.8) < 1e-4
    assert abs(confidence - 80.0) < 1e-4
