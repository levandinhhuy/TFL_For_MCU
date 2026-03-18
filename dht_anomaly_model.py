import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt

PREFIX = "dht_anomaly_model"
ALERT_LEVELS = {
    0: "normal",
    1: "warning",
    2: "critical",
}

data = pd.read_csv("dataset.csv")
data.columns = [col.strip().lower() for col in data.columns]

X = data[["temperature", "humidity"]].values.astype("float32")
y = data["label"].values.astype("int32")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

normalizer = tf.keras.layers.Normalization(axis=-1, name="input_norm")
normalizer.adapt(X_train)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    normalizer,
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(3, activation="softmax")
])

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"],
)

# Thêm EarlyStopping để dừng khi val_loss không cải thiện sau 5 epochs
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Lưu kết quả vào biến history
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    verbose=1
)

# Plot training curves and save as image
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="val")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig(PREFIX + "_training_plot.png", dpi=150)
plt.close()

model.save(PREFIX + ".h5")

def predict_alert_level(temperature, humidity):
    sample = tf.convert_to_tensor([[temperature, humidity]], dtype=tf.float32)
    probs = model(sample, training=False).numpy()[0]
    level = int(tf.argmax(probs).numpy())
    return level, ALERT_LEVELS[level], probs

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {acc:.4f}")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open(PREFIX + ".tflite", "wb") as f:
    f.write(tflite_model)

tflite_path = PREFIX + '.tflite'
output_header_path = PREFIX + '.h'

with open(tflite_path, 'rb') as tflite_file:
    tflite_content = tflite_file.read()

hex_lines = [', '.join([f'0x{byte:02x}' for byte in tflite_content[i:i + 12]]) for i in range(0, len(tflite_content), 12)]
hex_array = ',\n  '.join(hex_lines)

with open(output_header_path, 'w') as header_file:
    header_file.write('const unsigned char dht_anomaly_model_tflite[] = {\n  ')
    header_file.write(f'{hex_array}\n')
    header_file.write('};\n\n')