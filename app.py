from flask import Flask, request, jsonify, render_template
from keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import io
import base64
from PIL import Image

app = Flask(__name__)

# Load pre-trained model
MODEL_PATH = "model_weedcrops.h5"
model = load_model(MODEL_PATH)

# Define classes
classes = ["Crop", "Weed"]

def preprocess_image(file_stream):
    """
    Preprocess an image file stream into a format suitable for prediction.
    """
    try:
        img = load_img(file_stream, target_size=(150, 150))  # Resize to model input size
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img / 255.0  # Normalize to [0, 1]
        return img
    except Exception as e:
        raise ValueError(f"Error during image preprocessing: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' in request.files:
            # Handle file upload
            file = request.files['file']
            if file.filename == '':
                return jsonify({"error": "No file selected."}), 400
            # Convert SpooledTemporaryFile to BytesIO
            file_stream = io.BytesIO(file.read())
            img = preprocess_image(file_stream)

        elif request.is_json:
            # Handle Base64 image from webcam
            data = request.get_json()
            image_data = data.get('image')
            if not image_data:
                return jsonify({"error": "No image data provided."}), 400
            img_bytes = io.BytesIO(base64.b64decode(image_data.split(",")[1]))
            img = preprocess_image(img_bytes)

        else:
            return jsonify({"error": "Invalid request format."}), 400

        # Perform prediction
        predictions = model.predict(img)
        predicted_class = classes[np.argmax(predictions)]
        confidence = np.max(predictions) * 100

        return jsonify({
            "result": predicted_class,
            "confidence": f"{confidence:.2f}%"
        })

    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5001)
