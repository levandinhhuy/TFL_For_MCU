# TFL_For_MCU

Một bộ ví dụ và mô hình phát hiện người (person detection) nhằm dùng với TensorFlow Lite cho các vi điều khiển / MCU (TFLite Micro) và thiết bị nhúng.

## Tổng quan
Repo này chứa các file mô hình (HDF5, TFLite) và các header C được sinh sẵn để nhúng vào firmware MCU, cùng một vài script Python phục vụ kiểm thử và chuyển đổi.

## Nội dung chính của repository
- PERSON_DETECT_MODEL.exe
- PERSON_DETECT_MODEL.h
- PERSON_DETECT_MODEL.h5
- PERSON_DETECT_MODEL.tflite
- TEST_AI_ESP.h
- TEST_AI_ESP.h5
- TEST_AI_ESP.tflite
- TFL_For_MCU.py
- person_detect_model.py
- sensor_data.csv
- dataset/ (thư mục dữ liệu)
- README.md (file này)

## Yêu cầu
- Python 3.8+ (để chạy script kiểm thử / chuyển đổi)
- TensorFlow hoặc tflite-runtime (nếu cần chạy/interprete .tflite trên máy phát triển)
- Công cụ build cho MCU (toolchain, IDE tương ứng với board của bạn)
- xxd hoặc công cụ tương đương để chuyển file nhị phân sang C array (nếu bạn cần sinh header thủ công)

## Hướng dẫn nhanh

1. Chạy kiểm thử / inference trên máy:
- Cài dependencies (ví dụ):
  pip install -r requirements.txt
  (Nếu không có file requirements.txt, cài tensorflow hoặc tflite-runtime theo nhu cầu.)

- Ví dụ chạy script:
  python person_detect_model.py
  hoặc
  python TFL_For_MCU.py

2. Sử dụng model trên MCU:
- Nếu bạn sử dụng header đã có sẵn (PERSON_DETECT_MODEL.h hoặc TEST_AI_ESP.h), chỉ cần include file .h vào project C/C++ của bạn.
- Nếu bạn bắt đầu từ file .tflite và cần sinh header C:
  xxd -i PERSON_DETECT_MODEL.tflite > PERSON_DETECT_MODEL.cc
  (Chỉnh tên và định dạng theo toolchain/IDE của bạn.)
- Tích hợp model array vào dự án TFLite Micro: đưa mảng model vào phần định nghĩa `const unsigned char g_model[] = { ... };` và tham chiếu tới nó khi khởi tạo Interpreter tùy theo hướng dẫn TFLite Micro.

3. Nếu bạn muốn tái huấn luyện hoặc chuyển đổi:
- Mô hình huấn luyện thô có thể nằm ở PERSON_DETECT_MODEL.h5 hoặc TEST_AI_ESP.h5 — mở và xử lý bằng TensorFlow/Keras.
- Để chuyển sang TFLite:
  ```python
  # ví dụ (sử dụng TensorFlow)
  import tensorflow as tf
  model = tf.keras.models.load_model('PERSON_DETECT_MODEL.h5')
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  tflite_model = converter.convert()
  open('PERSON_DETECT_MODEL.tflite', 'wb').write(tflite_model)
  ```

## Ví dụ tích hợp cho ESP / MCU
- Các file TEST_AI_ESP.* gợi ý về mẫu dành cho ESP hoặc board nhúng tương tự — kiểm tra file header để biết tên mảng và cách dùng.
- Trên MCU: thêm runtime TFLite Micro, tạo Interpreter, cấp bộ nhớ arena, và gọi Invoke để thực hiện inference.

## Dữ liệu và kiểm thử
- dataset/: nơi chứa dữ liệu ảnh/dữ liệu huấn luyện (nếu có).
- sensor_data.csv: ví dụ dữ liệu cảm biến (mẫu).
