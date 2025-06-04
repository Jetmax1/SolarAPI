from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

# Load model and class labels
model = load_model("m32.h5")
class_labels = ['Cell', 'Cell-Multi', 'Cracking', 'Diode', 'Diode-Multi',
                'Hot-Spot', 'Hot-Spot-Multi', 'No-Anomaly', 'Offline-Module',
                'Shadowing', 'Soiling', 'Vegetation']

# FastAPI app
app = FastAPI(title="Solar Fault Detection API")

def preprocess(img: Image.Image):
    img = img.convert("RGB")
    img = img.resize((24, 40))  # width, height
    img_array = np.array(img)
    return img_array.reshape(1, 40, 24, 3)

@app.get("/")
def read_root():
    return {"message": "Solar Fault Detection API is live!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        img_input = preprocess(image)
        preds = model.predict(img_input)
        pred_index = np.argmax(preds[0])
        confidence = float(np.max(preds[0]))
        predicted_class = class_labels[pred_index]

        return {
            "predicted_class": predicted_class,
            "confidence": confidence
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
