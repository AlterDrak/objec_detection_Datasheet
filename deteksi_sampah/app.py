from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import os
import cv2
import time
import base64
import numpy as np

app = Flask(__name__)

# Folder
UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load YOLO model
model = YOLO("best.pt")


@app.route("/", methods=["GET", "POST"])
def index():

    result_image = None
    uploaded_image = None

    detected_class = None
    confidence = None
    detection_time = None

    total_objects = 0
    full_count = 0
    half_count = 0
    empty_count = 0
    overflow_count = 0

    if request.method == "POST":

        file = request.files.get("image")

        if file:

            # Save uploaded image
            filepath = os.path.join(
                UPLOAD_FOLDER,
                file.filename
            )

            file.save(filepath)

            uploaded_image = filepath

            # Start detection timer
            start_time = time.time()

            # Predict
            results = model(filepath)

            # End timer
            end_time = time.time()

            detection_time = round(
                end_time - start_time,
                3
            )

            # Annotated image
            annotated_frame = results[0].plot()

            # Save result image
            result_path = os.path.join(
                RESULT_FOLDER,
                "result_" + file.filename
            )

            cv2.imwrite(
                result_path,
                annotated_frame
            )

            result_image = result_path

            # Get detection boxes
            boxes = results[0].boxes

            total_objects = len(boxes)

            classes_detected = []

            highest_conf = 0

            # Loop all detected objects
            for box in boxes:

                class_id = int(box.cls[0])

                class_name = model.names[class_id]

                conf = float(box.conf[0]) * 100

                classes_detected.append(class_name)

                # Count objects
                if class_name == "full":
                    full_count += 1

                elif class_name == "half-full":
                    half_count += 1

                elif class_name == "empty":
                    empty_count += 1

                elif class_name == "overflowing":
                    overflow_count += 1

                # Highest confidence
                if conf > highest_conf:
                    highest_conf = conf

            # Final status logic (prioritas: overflowing > full > half-full > empty)
            if "overflowing" in classes_detected:
                detected_class = "overflowing"

            elif "full" in classes_detected:
                detected_class = "full"

            elif "half-full" in classes_detected:
                detected_class = "half-full"

            elif "empty" in classes_detected:
                detected_class = "empty"

            else:
                detected_class = "Tidak Terdeteksi"

            confidence = round(highest_conf, 2)

    return render_template(

        "index.html",

        result_image=result_image,
        uploaded_image=uploaded_image,

        detected_class=detected_class,
        confidence=confidence,
        detection_time=detection_time,

        total_objects=total_objects,
        full_count=full_count,
        half_count=half_count,
        empty_count=empty_count,
        overflow_count=overflow_count
    )


@app.route("/detect_frame", methods=["POST"])
def detect_frame():
    """Endpoint untuk deteksi live camera frame."""
    try:
        data = request.get_json()

        if not data or "image" not in data:
            return jsonify({"error": "No image data"}), 400

        # Decode base64 image
        image_data = data["image"]

        # Remove data URL prefix if present
        if "," in image_data:
            image_data = image_data.split(",")[1]

        img_bytes = base64.b64decode(image_data)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Failed to decode image"}), 400

        # Start timer
        start_time = time.time()

        # Predict
        results = model(img, conf=0.25)

        # End timer
        detection_time = round(time.time() - start_time, 3)

        # Get detection data
        boxes = results[0].boxes
        detections = []

        full_count = 0
        half_count = 0
        empty_count = 0
        overflow_count = 0

        highest_conf = 0
        detected_class = "Tidak Terdeteksi"
        classes_detected = []

        img_h, img_w = img.shape[:2]

        for box in boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            conf = float(box.conf[0])

            # Get bounding box coordinates (xyxy format)
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            classes_detected.append(class_name)

            # Count objects
            if class_name == "full":
                full_count += 1
            elif class_name == "half-full":
                half_count += 1
            elif class_name == "empty":
                empty_count += 1
            elif class_name == "overflowing":
                overflow_count += 1

            if conf * 100 > highest_conf:
                highest_conf = conf * 100

            detections.append({
                "class": class_name,
                "confidence": round(conf * 100, 1),
                "bbox": {
                    "x1": round(x1 / img_w, 4),
                    "y1": round(y1 / img_h, 4),
                    "x2": round(x2 / img_w, 4),
                    "y2": round(y2 / img_h, 4)
                }
            })

        # Final status
        if "overflowing" in classes_detected:
            detected_class = "overflowing"
        elif "full" in classes_detected:
            detected_class = "full"
        elif "half-full" in classes_detected:
            detected_class = "half-full"
        elif "empty" in classes_detected:
            detected_class = "empty"

        return jsonify({
            "detections": detections,
            "status": detected_class,
            "confidence": round(highest_conf, 2),
            "total_objects": len(boxes),
            "full_count": full_count,
            "half_count": half_count,
            "empty_count": empty_count,
            "overflow_count": overflow_count,
            "detection_time": detection_time
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Coba jalankan dengan HTTPS (SSL) agar kamera bisa diakses di perangkat mobile
    try:
        import OpenSSL
        print("\n" + "="*80)
        print("Menjalankan server dengan HTTPS (Self-signed)...")
        print("Silakan akses melalui: https://[IP_KAMU]:5000")
        print("="*80 + "\n")
        app.run(host="0.0.0.0", port=5000, debug=True, ssl_context="adhoc")
    except ImportError:
        print("\n" + "="*80)
        print("PERINGATAN: Modul 'pyopenssl' tidak ditemukan.")
        print("Untuk mengaktifkan kamera di HP / perangkat mobile lain, browser memerlukan koneksi aman (HTTPS).")
        print("Silakan jalankan perintah berikut untuk menginstal modul pendukung:")
        print("    pip install pyopenssl cryptography")
        print("Setelah diinstal, jalankan kembali server ini agar otomatis menggunakan HTTPS.")
        print("="*80 + "\n")
        print("Menjalankan server dengan HTTP biasa...")
        app.run(host="0.0.0.0", port=5000, debug=True)