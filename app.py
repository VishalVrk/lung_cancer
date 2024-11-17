from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename
import base64
import random

# Load the model
model = load_model("lung_cancer_model.h5")

app = Flask(__name__)

# Function to predict and annotate with the prediction score
def predict_and_annotate_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None, "Error: Image could not be loaded.", None

    img = cv2.resize(img, (256, 256))
    img_array = img.astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Get prediction and score
    predictions = model.predict(img_array)
    score = float(predictions[0][0])  # Extract score as a float
    prediction_class = (predictions > 0.5).astype(int)
    
    label = "CANCER" if prediction_class[0][0] == 1 else "Normal"
    
    # Annotate image if pneumonia is detected
    if label == "CANCER":
        img = annotate_pneumonia(img)

    # Save the annotated image temporarily
    annotated_path = "uploads/annotated_image.jpg"
    cv2.imwrite(annotated_path, img)
    
    # Encode image for display
    _, img_encoded = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')
    
    return label, img_base64, score

# Function to split image into quadrants and annotate with "PNEUMONIA"
def annotate_pneumonia(img):
    # Apply a green tint to the entire image
    green_overlay = np.zeros(img.shape, img.dtype)
    green_overlay[:, :] = (0, 255, 0)  # Green color
    img = cv2.addWeighted(img, 0.95, green_overlay, 0.1, 0)
    return img

# Route for the main page
@app.route("/", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join("uploads", filename)
            file.save(file_path)
            
            # Predict and get result with score
            label, img_base64, score = predict_and_annotate_image(file_path)
            
            return render_template("result.html", prediction=label, image_data=img_base64, score=score)
    return render_template("upload.html")

# Main execution
if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5555)))
