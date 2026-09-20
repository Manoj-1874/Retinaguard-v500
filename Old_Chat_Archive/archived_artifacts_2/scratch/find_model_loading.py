with open(r"e:\V500\app.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

for idx, line in enumerate(lines):
    if "load_model" in line or "tf.keras" in line or "keras.models" in line or ".h5" in line or "MODEL_PATH" in line or "model =" in line:
        print(f"Line {idx + 1}: {line.strip()}")
