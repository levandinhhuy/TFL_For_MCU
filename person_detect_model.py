import tensorflow as tf
import numpy as np
import os
from PIL import Image

PREFIX = "person_detect_model"
DATA_DIR = 'dataset'
IMG_HEIGHT = 240
IMG_WIDTH = 240

BATCH_SIZE = 16   
EPOCHS = 40     

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),      
    tf.keras.layers.RandomZoom(0.2),          
    tf.keras.layers.RandomContrast(0.2),      
    tf.keras.layers.RandomBrightness(0.2),    
])

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="training", seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE,
    color_mode='rgb'
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="validation", seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE,
    color_mode='rgb'
)

class_counts = {}
for class_name in train_ds.class_names:
    class_dir = os.path.join(DATA_DIR, class_name)
    class_counts[class_name] = len([f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

total_samples = sum(class_counts.values())
class_weights = {}
for i, (class_name, count) in enumerate(class_counts.items()):
    class_weights[i] = total_samples / (len(class_counts) * count)

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    
    data_augmentation,
    tf.keras.layers.Rescaling(1./255),
    
    tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(2),
    
    tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(2),
    
    tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(2),
    
    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dense(64, activation='relu'),

    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(train_ds.class_names), activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True), # Tăng patience lên
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=4, verbose=1)
    ]
)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open(PREFIX + ".tflite", "wb") as f:
    f.write(tflite_model)

print(f"Kích thước Model: {len(tflite_model)/1024:.2f} KB")
print(f"Final Train Acc: {history.history['accuracy'][-1]:.4f}")
print(f"Final Val Acc: {history.history['val_accuracy'][-1]:.4f}")

tflite_path = PREFIX + '.tflite'
output_header_path = PREFIX + '.h'

with open(tflite_path, 'rb') as tflite_file:
    tflite_content = tflite_file.read()

hex_lines = [', '.join([f'0x{byte:02x}' for byte in tflite_content[i:i + 12]]) 
             for i in range(0, len(tflite_content), 12)]
hex_array = ',\n  '.join(hex_lines)

with open(output_header_path, 'w') as header_file:
    header_file.write(f'const unsigned char {PREFIX.lower()}[] = {{\n  ')
    header_file.write(f'{hex_array}\n')
    header_file.write('};\n')

