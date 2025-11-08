import tensorflow as tf
import numpy as np
import os

# ================= CẤU HÌNH =================
PREFIX = "PERSON_DETECT_MODEL"  # Tên tiền tố cho các file đầu ra
DATA_DIR = 'dataset'            # Tên thư mục chứa dữ liệu ảnh
IMG_HEIGHT = 240                # Chiều cao ảnh đầu vào cho mô hình
IMG_WIDTH = 240                 # Chiều rộng ảnh đầu vào cho mô hình
BATCH_SIZE = 16                 # Số lượng ảnh xử lý trong một lần (batch)
EPOCHS = 10                     # Số lần duyệt qua toàn bộ tập dữ liệu

data = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

class_names = data.class_names

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    tf.keras.layers.Conv2D(16, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(data, epochs=EPOCHS)
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