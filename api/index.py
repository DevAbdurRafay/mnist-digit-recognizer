import base64
import io
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = Path(__file__).parent.parent / "model" / "mnist_weights.npz"
_weights = None


def get_weights():
    global _weights
    if _weights is None:
        data = np.load(MODEL_PATH)
        _weights = {
            "W1": data["arr_0"], "b1": data["arr_1"],
            "W2": data["arr_2"], "b2": data["arr_3"],
            "W3": data["arr_4"], "b3": data["arr_5"],
        }
    return _weights


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def forward(x, w):
    h1 = relu(x @ w["W1"] + w["b1"])
    h2 = relu(h1 @ w["W2"] + w["b2"])
    out = h2 @ w["W3"] + w["b3"]
    return softmax(out.flatten())


def count_boundary_pixels(binary: np.ndarray) -> int:
    padded = np.pad(binary, 1, mode="constant", constant_values=0)
    neighbors = (
        padded[:-2, 1:-1] + padded[2:, 1:-1] +
        padded[1:-1, :-2] + padded[1:-1, 2:]
    )
    boundary = binary & (neighbors < 4)
    return int(boundary.sum())


def preprocess(image_data_url: str) -> np.ndarray:
    _, encoded = image_data_url.split(",", 1)
    raw = base64.b64decode(encoded)
    img = Image.open(io.BytesIO(raw)).convert("L")
    arr = np.array(img, dtype=np.float32)

    arr[arr < 0.20 * 255] = 0

    if arr.max() == 0:
        raise ValueError("empty")

    rows = np.any(arr > 0, axis=1)
    cols = np.any(arr > 0, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    crop = arr[rmin:rmax + 1, cmin:cmax + 1]

    h, w = crop.shape
    binary = (crop > 0).astype(np.int32)
    ink_pixels = int(binary.sum())
    boundary = count_boundary_pixels(binary)
    stroke_width = (2 * ink_pixels / boundary) if boundary > 0 else 1.0
    target_width = 0.11 * max(h, w)

    if stroke_width < target_width:
        pad = max(1, int((target_width - stroke_width) / 2) + 1)
        filter_size = 2 * pad + 1
        crop_img = Image.fromarray(np.clip(crop, 0, 255).astype(np.uint8))
        crop_img = crop_img.filter(ImageFilter.MaxFilter(filter_size))
        crop = np.array(crop_img, dtype=np.float32)

    h, w = crop.shape
    scale = 20.0 / max(h, w)
    new_h = max(1, round(h * scale))
    new_w = max(1, round(w * scale))
    crop_img = Image.fromarray(np.clip(crop, 0, 255).astype(np.uint8))
    crop_img = crop_img.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("L", (28, 28), 0)
    paste_x = (28 - new_w) // 2
    paste_y = (28 - new_h) // 2
    canvas.paste(crop_img, (paste_x, paste_y))
    canvas_arr = np.array(canvas, dtype=np.float32)

    total = canvas_arr.sum()
    if total > 0:
        cy = (np.arange(28)[:, None] * canvas_arr).sum() / total
        cx = (np.arange(28)[None, :] * canvas_arr).sum() / total
        shift_y = int(round(14 - cy))
        shift_x = int(round(14 - cx))
        canvas_arr = np.roll(canvas_arr, shift_y, axis=0)
        canvas_arr = np.roll(canvas_arr, shift_x, axis=1)

    return (canvas_arr / 255.0).flatten().reshape(1, 784)


class PredictRequest(BaseModel):
    image: str


def _predict_handler(req: PredictRequest):
    try:
        x = preprocess(req.image)
    except ValueError as e:
        if "empty" in str(e):
            raise HTTPException(status_code=400, detail={"error": "Canvas is empty"})
        raise HTTPException(status_code=400, detail={"error": str(e)})
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": f"Preprocessing failed: {str(e)}"})

    try:
        w = get_weights()
        probs = forward(x, w)
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": f"Inference failed: {str(e)}"})

    pred = int(np.argmax(probs))
    conf = round(float(probs[pred]) * 100, 2)
    prob_list = [round(float(p) * 100, 2) for p in probs]

    return {"predicted_digit": pred, "confidence": conf, "probabilities": prob_list}


# /api/index   → Vercel file-based routing (api/index.py → /api/index)
# /api/predict → local uvicorn dev
@app.post("/api/index")
@app.post("/api/predict")
def predict(req: PredictRequest):
    return _predict_handler(req)


# Local dev only: serve public/ as static files
PUBLIC_DIR = Path(__file__).parent.parent / "public"
if PUBLIC_DIR.exists():
    app.mount("/", StaticFiles(directory=PUBLIC_DIR, html=True), name="static")
