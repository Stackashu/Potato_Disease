from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np

app = Flask(__name__)

# Load model
model = load_model("Potato_model_v2.h5")
class_names = ["Early Blight", "Late Blight", "Healthy"]

def prepare_image(img, target_size=(256,256)):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(target_size)
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array,0)
    return img_array

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    img = Image.open(file.stream)
    img_array = prepare_image(img)
    pred = model.predict(img_array)
    pred_class = class_names[np.argmax(pred[0])]
    confidence = round(100*np.max(pred[0]),2)
    return jsonify({"prediction": pred_class, "confidence": confidence})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
