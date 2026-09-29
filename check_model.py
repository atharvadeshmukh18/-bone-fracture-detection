import tensorflow as tf
import numpy as np

MODEL_PATH = "fracture_model.h5"
TEST_DIR = "dataset/test"

IMG_SIZE = (128, 128)
BATCH_SIZE = 32

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

print("Loading test dataset...")

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False,
)

print("\nClass names:")
print(test_ds.class_names)

# Normalize images exactly like your inference.py
test_ds = test_ds.map(
    lambda images, labels: (tf.cast(images, tf.float32) / 255.0, labels)
)

print("\nRunning predictions...")

predictions = model.predict(test_ds, verbose=1).reshape(-1)

# Get true labels
true_labels = np.concatenate(
    [labels.numpy().reshape(-1) for _, labels in test_ds],
    axis=0
)

predicted_labels = (predictions >= 0.5).astype(int)

accuracy = np.mean(predicted_labels == true_labels) * 100

print("\n" + "=" * 50)
print("MODEL VALIDATION RESULTS")
print("=" * 50)

print(f"Test images: {len(true_labels)}")
print(f"Accuracy: {accuracy:.2f}%")

print("\nClass mapping:")
print("0 = fractured")
print("1 = not fractured")

# Confusion counts
fractured_total = np.sum(true_labels == 0)
not_fractured_total = np.sum(true_labels == 1)

fractured_correct = np.sum(
    (true_labels == 0) & (predicted_labels == 0)
)

fractured_wrong = np.sum(
    (true_labels == 0) & (predicted_labels == 1)
)

not_fractured_correct = np.sum(
    (true_labels == 1) & (predicted_labels == 1)
)

not_fractured_wrong = np.sum(
    (true_labels == 1) & (predicted_labels == 0)
)

print("\nFractured images:")
print(f"  Total: {fractured_total}")
print(f"  Correctly detected: {fractured_correct}")
print(f"  Incorrectly detected as no fracture: {fractured_wrong}")

print("\nNot fractured images:")
print(f"  Total: {not_fractured_total}")
print(f"  Correctly detected: {not_fractured_correct}")
print(f"  Incorrectly detected as fracture: {not_fractured_wrong}")

print("\n" + "=" * 50)