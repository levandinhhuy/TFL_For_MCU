import argparse
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

PREFIX = "gas_leak_model"
FEATURE_COLS = ["temperature", "humidity", "gas"]
NUM_CLASSES = 3
ALERT_LEVELS = {
    0: "normal",
    1: "warning",
    2: "critical",
}
COLUMN_ALIASES = {
    "temperature": ["temperature", "temp", "temp_c", "t"],
    "humidity": ["humidity", "hum", "rh"],
    "gas": ["gas", "mq", "gas_value", "sensor"],
    "label": ["label", "class", "target", "y"],
}


def resolve_column(df_columns, canonical_name: str) -> str:
    available = {col.strip().lower(): col for col in df_columns}
    for alias in COLUMN_ALIASES[canonical_name]:
        if alias in available:
            return available[alias]
    raise ValueError(
        f"Dataset is missing required column for '{canonical_name}'. "
        f"Expected one of: {COLUMN_ALIASES[canonical_name]}"
    )


def load_dataset(csv_path: str = "dataset.csv", feature_cols=None, label_col: str = "label", test_size: float = 0.2, random_state: int = 42):
    if feature_cols is None:
        feature_cols = FEATURE_COLS
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    resolved_feature_cols = [resolve_column(df.columns, col) for col in feature_cols]
    resolved_label_col = resolve_column(df.columns, label_col)
    X = df[resolved_feature_cols].values.astype("float32")
    y = df[resolved_label_col].values.astype("int32")
    # debug: print label distribution
    try:
        vc = pd.Series(y).value_counts().to_dict()
        print(f"Label distribution (entire dataset): {vc}")
    except Exception:
        pass
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def build_model(input_shape=(len(FEATURE_COLS),)) -> Tuple[tf.keras.Model, tf.keras.layers.Layer]:
    normalizer = tf.keras.layers.Normalization(axis=-1, name="input_norm")
    inputs = tf.keras.Input(shape=input_shape)
    x = normalizer(inputs)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    x = tf.keras.layers.Dense(8, activation="relu")(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model, normalizer


def train_model(model: tf.keras.Model, normalizer: tf.keras.layers.Layer, X_train, y_train, X_val, y_val,
                epochs: int = 100, batch_size: int = 32, patience: int = 5, prefix: str = PREFIX):
    # adapt normalizer
    normalizer.adapt(X_train)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(X_val, y_val), callbacks=[early_stop], verbose=1)
    plot_history(history, prefix)
    model.save(prefix + ".h5")
    return history


def plot_history(history, prefix: str = PREFIX):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history.get("accuracy", []), label="train")
    plt.plot(history.history.get("val_accuracy", []), label="val")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history.get("loss", []), label="train")
    plt.plot(history.history.get("val_loss", []), label="val")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(prefix + "_training_plot.png", dpi=150)
    plt.close()


def evaluate_model(model: tf.keras.Model, X_test, y_test):
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {acc:.4f}")
    # detailed diagnostics
    preds = model.predict(X_test)
    y_pred = np.argmax(preds, axis=1)
    try:
        print("Classification report:\n", classification_report(y_test, y_pred, digits=4))
        print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    except Exception:
        pass
    return loss, acc


def export_tflite(model: tf.keras.Model, tflite_path: str, optimize: bool = True):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if optimize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    return tflite_path


def write_c_header(tflite_path: str, header_path: str, array_name: str = "gas_leak_model_tflite"):
    with open(tflite_path, 'rb') as tflite_file:
        tflite_content = tflite_file.read()
    hex_lines = [', '.join([f'0x{byte:02x}' for byte in tflite_content[i:i + 12]]) for i in range(0, len(tflite_content), 12)]
    hex_array = ',\n  '.join(hex_lines)
    with open(header_path, 'w') as header_file:
        header_file.write(f'const unsigned char {array_name}[] = {{\n  ')
        header_file.write(f'{hex_array}\n')
        header_file.write('};\n\n')


def predict_alert_level_from_model(model: tf.keras.Model, temperature: float, humidity: float, gas: float):
    sample = tf.convert_to_tensor([[temperature, humidity, gas]], dtype=tf.float32)
    probs = model(sample, training=False).numpy()[0]
    level = int(tf.argmax(probs).numpy())
    return level, ALERT_LEVELS[level], probs


def cli():
    parser = argparse.ArgumentParser(description='Gas leak anomaly model pipeline')
    sub = parser.add_subparsers(dest='command')

    train_p = sub.add_parser('train')
    train_p.add_argument('--data', default='dataset.csv')
    train_p.add_argument('--epochs', type=int, default=100)
    train_p.add_argument('--batch', type=int, default=32)

    eval_p = sub.add_parser('evaluate')
    eval_p.add_argument('--model', default=PREFIX + '.h5')
    eval_p.add_argument('--data', default='dataset.csv')

    export_p = sub.add_parser('export')
    export_p.add_argument('--model', default=PREFIX + '.h5')
    export_p.add_argument('--tflite', default=PREFIX + '.tflite')
    export_p.add_argument('--header', default=PREFIX + '.h')

    predict_p = sub.add_parser('predict')
    predict_p.add_argument('--model', default=PREFIX + '.h5')
    predict_p.add_argument('--temperature', type=float, required=True)
    predict_p.add_argument('--humidity', type=float, required=True)
    predict_p.add_argument('--gas', type=float, required=True)

    args = parser.parse_args()
    if args.command == 'train':
        X_train, X_test, y_train, y_test = load_dataset(args.data)
        # print train/test label distributions for debugging
        try:
            print("Train label distribution:", pd.Series(y_train).value_counts().to_dict())
            print("Test label distribution:", pd.Series(y_test).value_counts().to_dict())
        except Exception:
            pass
        model, normalizer = build_model(input_shape=(X_train.shape[1],))
        train_model(model, normalizer, X_train, y_train, X_test, y_test, epochs=args.epochs, batch_size=args.batch)
        evaluate_model(model, X_test, y_test)
        tflite_path = export_tflite(model, PREFIX + '.tflite')
        write_c_header(tflite_path, PREFIX + '.h')
    elif args.command == 'evaluate':
        X_train, X_test, y_train, y_test = load_dataset(args.data)
        model = tf.keras.models.load_model(args.model)
        evaluate_model(model, X_test, y_test)
    elif args.command == 'export':
        model = tf.keras.models.load_model(args.model)
        tflite_path = export_tflite(model, args.tflite)
        write_c_header(tflite_path, args.header)
    elif args.command == 'predict':
        model = tf.keras.models.load_model(args.model)
        level, label, probs = predict_alert_level_from_model(model, args.temperature, args.humidity, args.gas)
        print({'level': level, 'label': label, 'probs': probs.tolist()})
    else:
        parser.print_help()


if __name__ == '__main__':
    cli()