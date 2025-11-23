import tensorflow as tf
import numpy as np
import os
from PIL import Image

PREFIX = "PERSON_DETECT_MODEL"
DATA_DIR = 'dataset'
IMG_HEIGHT = 240
IMG_WIDTH = 240

BATCH_SIZE = 8   
EPOCHS = 30

if os.path.exists(DATA_DIR):
    for root, _, files in os.walk(DATA_DIR):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(root, f)
                try:
                    with Image.open(path) as img:
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                            img.save(path, quality=95)
                except:
                    os.remove(path)
else:
    exit()

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

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    include_top=False,
    weights='imagenet',
    alpha=0.35
)

base_model.trainable = False

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    
    data_augmentation,
    tf.keras.layers.Rescaling(1./127.5, offset=-1),  
    
    base_model,
    
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3), 
    tf.keras.layers.Dense(len(train_ds.class_names), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("Phase 1: Training with frozen MobileNetV2...")
history = model.fit(train_ds, validation_data=val_ds, epochs=10)

base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)

print("\nPhase 2: Fine-tuning top layers...")
history_fine = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS-10)

def representative_data_gen():
    for input_value, _ in val_ds.take(100):
        yield [tf.cast(input_value, tf.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
converter.representative_dataset = representative_data_gen

try:
    tflite_model = converter.convert()
except Exception as e:
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32
    tflite_model = converter.convert()

with open(PREFIX + ".tflite", "wb") as f:
    f.write(tflite_model)

print(f"\n{'='*50}")
print(f"Kích thước Model: {len(tflite_model)/1024:.2f} KB")
print(f"Training Accuracy: {history_fine.history['accuracy'][-1]*100:.2f}%")
print(f"Validation Accuracy: {history_fine.history['val_accuracy'][-1]*100:.2f}%")
print(f"{'='*50}")

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
