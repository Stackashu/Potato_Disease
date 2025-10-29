from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import uvicorn

app = FastAPI()

# Load model
model = load_model("Potato_model_v2.h5")
class_names = ["Early Blight", "Late Blight", "Healthy"]

def prepare_image(img, target_size=(256,256)):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, 0)
    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img = Image.open(file.file)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")
    img_array = prepare_image(img)
    pred = model.predict(img_array)
    pred_class = class_names[np.argmax(pred[0])]
    confidence = round(100*np.max(pred[0]), 2)
    return JSONResponse(content={"prediction": pred_class, "confidence": confidence})

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=5000)
