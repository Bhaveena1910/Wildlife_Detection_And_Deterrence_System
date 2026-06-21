import requests
import cv2
import numpy as np
import tensorflow as tf
import time
import os
import subprocess

# =========================
# CONFIG
# =========================
ESP32_URL = "http://192.168.4.1/capture"
CONF_THRESHOLD = 0.7
CAPTURE_DELAY = 15

# =========================
# LOAD MODEL
# =========================
print("Loading model...")
model = tf.keras.models.load_model("animal_model.h5")

with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

print("Model loaded successfully")

# =========================
# STORAGE
# =========================
os.makedirs("captured_images", exist_ok=True)

last_state = "OFF"


# =========================
# MAIN LOOP
# =========================
while True:
    try:
        response = requests.get(ESP32_URL, timeout=5)

        img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            continue

        # preprocess
        img_resized = cv2.resize(img, (224, 224))
        img_norm = img_resized / 255.0
        img_input = np.expand_dims(img_norm, axis=0)

        # predict
        predictions = model.predict(img_input, verbose=0)
        class_index = np.argmax(predictions)
        confidence = np.max(predictions)

        predicted_label = labels[class_index]

        print(f"Detected: {predicted_label} ({confidence:.2f})")

        # =========================
        # ACTION LOGIC
        # =========================
        if confidence > CONF_THRESHOLD:

            # save image
            folder = f"captured_images/{predicted_label}"
            os.makedirs(folder, exist_ok=True)

            filename = f"{folder}/{int(time.time())}.jpg"
            cv2.imwrite(filename, img)

            print(f"📸 Saved: {filename}")

        # display
        cv2.imshow("ESP32 LIVE", img)

        if cv2.waitKey(1) == 27:
            print("Stopping system...")
            break

        time.sleep(CAPTURE_DELAY)

    except Exception as e:
        print("❌ Error:", e)
        time.sleep(5)

# cleanup
signal_process.terminate()
cv2.destroyAllWindows()