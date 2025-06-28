from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np
import base64
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load your trained model
model = tf.keras.models.load_model("Blood Cell.h5")
class_labels = ['eosinophil', 'lymphocyte', 'monocyte', 'neutrophil']

# Predict the class of image
def predict_image_class(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_preprocessed = preprocess_input(img_resized.reshape((1, 224, 224, 3)))
    prediction = model.predict(img_preprocessed)
    predicted_class = class_labels[np.argmax(prediction)]
    return predicted_class, img_rgb

# Home page route
@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        if file:
            file_path = os.path.join("static/uploads", file.filename)
            file.save(file_path)

            class_label, img_rgb = predict_image_class(file_path)

            _, buffer = cv2.imencode('.png', cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
            img_str = base64.b64encode(buffer).decode('utf-8')

            return render_template("result.html", class_label=class_label, img_data=img_str)

    return render_template("home.html")

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
