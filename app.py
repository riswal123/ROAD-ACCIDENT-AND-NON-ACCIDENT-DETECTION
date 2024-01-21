from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import cv2

app = Flask(__name__)
model = tf.keras.models.load_model("D:\Project\Mini-Project-II\model.h5")
class_names = ['Accident', 'Non Accident']

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    image = request.files["image"]
    image = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
    image = cv2.resize(image, (240, 240))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    predictions = model.predict(image)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_class_index]

    # Check if the predicted class is the expected class
    expected_class = request.form.get("expected_class")

    if predicted_class == expected_class:
        confidence = predictions[0][predicted_class_index]
    else:
        confidence = 1 - predictions[0][predicted_class_index]

    return jsonify({"prediction": predicted_class, "confidence": float(confidence)})

if __name__ == "__main__":
    app.run()
