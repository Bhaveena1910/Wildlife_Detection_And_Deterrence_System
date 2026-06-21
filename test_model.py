import tensorflow as tf
import cv2
import numpy as np
import pygame
import os

# -------------------------------
# 🔹 Initialize pygame for sound
# -------------------------------
pygame.mixer.init()

def play_sound(label):
    try:
        file_path = f"sounds/{label}.mp3"

        if not os.path.exists(file_path):
            print(f"⚠️ Sound file not found: {file_path}")
            return

        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()

        print(f"🔊 Playing sound for: {label}")

        # Wait until sound finishes
        while pygame.mixer.music.get_busy():
            continue

    except Exception as e:
        print("❌ Sound error:", e)

# -------------------------------
# 🔹 Load model
# -------------------------------
try:
    model = tf.keras.models.load_model("animalmodel.h5")
    print("✅ Model loaded")
except:
    print("❌ Model not found")
    exit()

# -------------------------------
# 🔹 Load labels
# -------------------------------
try:
    with open("labels.txt", "r") as f:
        labels = [line.strip() for line in f.readlines()]
    print("✅ Labels loaded:", labels)
except:
    print("❌ labels.txt not found")
    exit()

# -------------------------------
# 🔹 Load test image
# -------------------------------
img_path = r"D:\sem6\dl\wildlife_project\test.jpg"   # 🔁 change if needed

img = cv2.imread(img_path)

if img is None:
    print("❌ Image not found:", img_path)
    exit()

print("✅ Image loaded")

# -------------------------------
# 🔹 Preprocess image
# -------------------------------
img_resized = cv2.resize(img, (224, 224))
img_norm = img_resized / 255.0
img_input = np.expand_dims(img_norm, axis=0)

# -------------------------------
# 🔹 Predict
# -------------------------------
predictions = model.predict(img_input)

class_index = np.argmax(predictions)
confidence = np.max(predictions)

predicted_label = labels[class_index]

print(f"\n🎯 Prediction: {predicted_label}")
print(f"📊 Confidence: {confidence:.2f}")

# -------------------------------
# 🔹 Play sound if confident
# -------------------------------
if confidence > 0.7:
    play_sound(predicted_label)
else:
    print("⚠️ Low confidence — no sound played")

# -------------------------------
# 🔹 Show image (optional)
# -------------------------------
cv2.imshow("Test Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()