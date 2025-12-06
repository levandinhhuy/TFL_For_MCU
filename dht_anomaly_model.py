import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import numpy as np

PREFIX = "dht_anomaly_model"
MODE = "TEST"  # "TRAIN", "TEST", or "FULL"

# ============ LOAD & SCALE DATA ============
data = pd.read_csv("dataset.csv", names=["Temperature (C)", "Humidity (%)", "Label"])
X = data[["Temperature (C)", "Humidity (%)"]].values
y = data["Label"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ============ BUILD MODEL ============
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss="binary_crossentropy", optimizer='adam', metrics=["accuracy"])

# ============ TRAIN ============
if MODE in ["TRAIN", "FULL"]:
    class_weights = {0: 4.5, 1: 1.0}
    model.fit(X_train, y_train, epochs=50, batch_size=16, 
              validation_data=(X_test, y_test), class_weight=class_weights, verbose=1)
    model.save(PREFIX + '.h5')
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(PREFIX + ".tflite", "wb") as f:
        f.write(tflite_model)
    
    # Convert to C header
    hex_lines = [', '.join([f'0x{byte:02x}' for byte in tflite_model[i:i + 12]]) 
                 for i in range(0, len(tflite_model), 12)]
    hex_array = ',\n  '.join(hex_lines)
    
    with open(PREFIX + ".h", 'w') as header_file:
        header_file.write(f'const unsigned char {PREFIX}_tflite[] = {{\n  ')
        header_file.write(f'{hex_array}\n')
        header_file.write('};\n')
    
else:
    model = tf.keras.models.load_model(PREFIX + '.h5')

# ============ TEST ============
if MODE in ["TEST", "FULL"]:
    interpreter = tf.lite.Interpreter(model_path=PREFIX + ".tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    test_cases = [
        (20.0, 0.40, "Cool, 40%"), (22.0, 0.45, "Cool, 45%"), 
        (24.0, 0.50, "Comfortable"), (26.0, 0.60, "Room, 60%"),
        (30.0, 0.50, "Warm, 50%"), (35.0, 0.40, "Very hot"),
        (5.0, 0.05, "Freezing"), (45.0, 0.95, "Very hot, 95%"),
        (18.0, 0.35, "Cool, 35%"), (28.0, 0.55, "Warm, 55%"),
        (32.0, 0.45, "Hot, 45%"), (38.0, 0.40, "Very hot, 40%"),
        (10.0, 0.10, "Cold, 10%"), (15.0, 0.90, "Cool, 90%"),
        (42.0, 0.92, "Extreme hot, 92%"), (3.0, 0.03, "Freezing, 3%"),
        (25.0, 0.50, "Room, 50%"), (27.0, 0.65, "Warm, 65%"),
        (33.0, 0.30, "Hot, 30%"), (40.0, 0.85, "Very hot, 85%"),
    ]

    print(f"{'Temp°C':<8} {'Humid':<8} {'Prediction':<12} {'Label':<15}")
    print("-" * 50)
    
    for temp, humid, desc in test_cases:
        test_scaled = scaler.transform([[temp, humid]]).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], test_scaled.reshape(1, 2))
        interpreter.invoke()
        pred = interpreter.get_tensor(output_details[0]['index'])[0][0]
        label = "🔴 ANOMALY" if pred > 0.5 else "🟢 NORMAL"
        print(f"{temp:<8.1f} {humid:<8.2f} {pred:<12.4f} {label}")
    
    print(f"\n📝 Scaling constants: Temp mean={scaler.mean_[0]:.2f}, Humid mean={scaler.mean_[1]:.4f}")
    print(f"📝 Scaling constants: Temp std={np.sqrt(scaler.var_[0]):.2f}, Humid std={np.sqrt(scaler.var_[1]):.4f}")