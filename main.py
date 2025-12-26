import os
import base64
import numpy as np
import face_recognition
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

KNOWN_FACES_DIR = "known_faces"
if not os.path.exists(KNOWN_FACES_DIR):
    os.makedirs(KNOWN_FACES_DIR)

def decode_base64_image(base64_string):
    try:
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        if image.mode != "RGB":
            image = image.convert("RGB")
        return np.array(image)
    except Exception as e:
        return None

@app.route("/register", methods=["POST"])
def register():
    try:
        data = request.json
        user_id = data.get("user_id")
        image_base64 = data.get("image")

        if not user_id or not image_base64:
            return jsonify({"success": False, "message": "Missing user_id or image"}), 400

        image_np = decode_base64_image(image_base64)
        if image_np is None:
            return jsonify({"success": False, "message": "Invalid image data"}), 400
        
        # Detect face to ensure image has a face
        face_encodings = face_recognition.face_encodings(image_np)
        if not face_encodings:
            return jsonify({"success": False, "message": "No face detected in image"}), 400
        
        # Save image
        file_path = os.path.join(KNOWN_FACES_DIR, f"{user_id}.jpg")
        img = Image.fromarray(image_np)
        img.save(file_path)
        
        return jsonify({"success": True, "message": "Face registered successfully"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route("/verify", methods=["POST"])
def verify():
    try:
        data = request.json
        user_id = data.get("user_id")
        image_base64 = data.get("image")

        if not user_id or not image_base64:
            return jsonify({"success": False, "message": "Missing user_id or image"}), 400

        # Load known face
        known_face_path = os.path.join(KNOWN_FACES_DIR, f"{user_id}.jpg")
        if not os.path.exists(known_face_path):
            return jsonify({"success": False, "message": "Face not registered for this user"}), 404
        
        known_image = face_recognition.load_image_file(known_face_path)
        known_encodings = face_recognition.face_encodings(known_image)
        
        if not known_encodings:
            return jsonify({"success": False, "message": "Stored face image is invalid"}), 500
        
        known_encoding = known_encodings[0]
        
        # Decode current image
        current_image_np = decode_base64_image(image_base64)
        if current_image_np is None:
            return jsonify({"success": False, "message": "Invalid current image data"}), 400

        current_encodings = face_recognition.face_encodings(current_image_np)
        
        if not current_encodings:
            return jsonify({"success": False, "message": "No face detected in current image"}), 400
        
        current_encoding = current_encodings[0]
        
        # Compare faces
        results = face_recognition.compare_faces([known_encoding], current_encoding, tolerance=0.6)
        face_distances = face_recognition.face_distance([known_encoding], current_encoding)
        
        match = bool(results[0])
        distance = float(face_distances[0])
        confidence = 1.0 - distance
        
        return jsonify({
            "success": True,
            "match": match,
            "distance": distance,
            "confidence": confidence
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
