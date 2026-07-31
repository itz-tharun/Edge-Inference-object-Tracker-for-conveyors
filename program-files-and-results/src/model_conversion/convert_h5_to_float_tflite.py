"""
convert_h5_to_float_tflite.py
-------------------------------
Converts pen_detector_model.h5 to a PLAIN FLOAT32 .tflite file - no
int8 quantization, no calibration/representative dataset involved.
This avoids the quantization bug entirely, since float32 conversion
just carries the same float weights straight across - nothing to
miscalibrate.

USAGE:
    python convert_h5_to_float_tflite.py --h5 pen_detector_model.h5 --out pen_detector_model_float32.tflite

Dependencies:
    pip install tensorflow
"""

import argparse

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet import preprocess_input


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", default=r"C:\Users\dhars\Downloads\pen_detector_model.h5")
    parser.add_argument("--out", default=r"C:\Users\dhars\Desktop\pen_detector_model_float32.tflite")
    args = parser.parse_args()

    print(f"[INFO] Loading {args.h5} ...")
    model = load_model(args.h5)

    print("[INFO] Converting to float32 TFLite (no quantization, no calibration dataset)...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Deliberately NOT setting converter.optimizations or a representative_dataset -
    # that's what triggers int8 calibration. Leaving both out keeps this a
    # straight float32 conversion.
    tflite_model = converter.convert()

    with open(args.out, "wb") as f:
        f.write(tflite_model)
    print(f"[INFO] Saved: {args.out}")

    # --- Sanity check: run the same input through both the h5 model and
    # the new float tflite model, confirm they agree. Not a real pen image,
    # just a numeric check that the conversion didn't silently break anything.
    print("\n[INFO] Running a quick numeric sanity check...")
    test_input = np.random.rand(1, 224, 224, 3).astype("float32") * 255.0
    test_input_preprocessed = preprocess_input(test_input.copy())

    h5_out = float(model.predict(test_input_preprocessed, verbose=0)[0][0])

    interpreter = tf.lite.Interpreter(model_path=args.out)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    interpreter.set_tensor(input_detail["index"], test_input_preprocessed)
    interpreter.invoke()
    tflite_out = float(interpreter.get_tensor(output_detail["index"])[0][0])

    diff = abs(h5_out - tflite_out)
    print(f"  h5 output     : {h5_out:.6f}")
    print(f"  tflite output : {tflite_out:.6f}")
    print(f"  difference    : {diff:.6f}")
    if diff < 1e-4:
        print("  [OK] Outputs match closely - conversion looks correct.")
    else:
        print("  [WARN] Outputs differ more than expected - worth double-checking.")


if __name__ == "__main__":
    main()