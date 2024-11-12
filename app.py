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

# Function to predict using the loaded model and annotate if pneumonia is detected
def predict_and_annotate_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None, "Error: Image could not be loaded."

    img = cv2.resize(img, (256, 256))
    img_array = img.astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    prediction_class = (predictions > 0.5).astype(int)
    label = "PNEUMONIA" if prediction_class[0][0] == 1 else "Normal"

    # Annotate image if pneumonia is detected
    if label == "PNEUMONIA":
        img = annotate_pneumonia(img)

    # Save the annotated image temporarily
    annotated_path = "uploads/annotated_image.jpg"
    cv2.imwrite(annotated_path, img)
    
    # Encode image for display
    _, img_encoded = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')
    
    return label, img_base64

# Function to split image into quadrants and annotate with "PNEUMONIA"
def annotate_pneumonia(img):
    # Apply a green tint to the entire image
    green_overlay = np.zeros(img.shape, img.dtype)
    green_overlay[:, :] = (0, 255, 0)  # Green color
    img = cv2.addWeighted(img, 0.7, green_overlay, 0.05, 0)

    # Define the quadrants for annotation
    height, width, _ = img.shape
    half_height, half_width = height // 2, width // 2
    quadrants = [
        (0, 0, half_width, half_height),
        (half_width, 0, width, half_height),
        (0, half_height, half_width, height),
        (half_width, half_height, width, height),
    ]

    for x1, y1, x2, y2 in quadrants:
        # Randomly position bounding box in each quadrant
        box_x = random.randint(x1, x2 - 50)
        box_y = random.randint(y1, y2 - 20)
        box_w, box_h = 100, 50  # Width and height of bounding box
        
        # Draw the bounding box
        cv2.rectangle(img, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 255, 0), 2)
            
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
            
            # Predict and get result
            label, img_base64 = predict_and_annotate_image(file_path)
            
            return render_template("result.html", prediction=label, image_data=img_base64)
    return render_template("upload.html")

# Main execution
if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
