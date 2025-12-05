import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

PREFIX = "dht_anomaly_model"

data = pd.read_csv("dataset.csv", names=["Temperature (C)", "Humidity (%)", "Label"])
X = data[["Temperature (C)", "Humidity (%)"]].values
y = data["Label"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Enhanced classifier model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    
    # First hidden layer với batch normalization
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    
    # Second hidden layer
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.BatchNormalization(), 
    tf.keras.layers.Dropout(0.2),
    
    # Third hidden layer
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    
    # Output layer
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Advanced optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)

model.compile(
    loss="binary_crossentropy", 
    optimizer=optimizer, 
    metrics=["accuracy", "precision", "recall"]
)

print("\n=== Model Architecture ===")
model.summary()

# Enhanced training với callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-7, verbose=1)
]

history = model.fit(
    X_train, y_train, 
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    verbose=1
)

# Evaluate model
print("\n=== Model Evaluation ===")
train_loss, train_acc, train_prec, train_rec = model.evaluate(X_train, y_train, verbose=0)
test_loss, test_acc, test_prec, test_rec = model.evaluate(X_test, y_test, verbose=0)

print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Prec: {train_prec:.4f}, Rec: {train_rec:.4f}")
print(f"Test  - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, Prec: {test_prec:.4f}, Rec: {test_rec:.4f}")
model.save(PREFIX + '.h5')


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

hex_lines = [', '.join([f'0x{byte:02x}' for byte in tflite_content[i:i + 12]]) for i in
         range(0, len(tflite_content), 12)]

hex_array = ',\n  '.join(hex_lines)

with open(output_header_path, 'w') as header_file:
    header_file.write('const unsigned char model[] = {\n  ')
    header_file.write(f'{hex_array}\n')
    header_file.write('};\n\n')