# 🥔 Potato Disease Classification API

This project provides a **FastAPI-based machine learning API** for detecting potato leaf diseases using a trained TensorFlow model.  
You can deploy it easily on Render, and use the `/predict` endpoint to upload an image and receive the predicted disease along with confidence.

---

## 🚀 Live API Endpoint

**POST**  
`https://potato-disease-1-ypvi.onrender.com/predict`

**Form Data:**  
- `file`: (Required) — Upload an image file of a potato leaf.

**Response Example:**
```json
{
  "prediction": "Early Blight",
  "confidence": 95.42
}
