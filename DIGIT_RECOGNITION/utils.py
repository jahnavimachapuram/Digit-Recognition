import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

MODEL_PATH = "saved_model"            # folder; load latest model file

def load_last_model():
    import glob
    all_saves = sorted(glob.glob(f"{MODEL_PATH}/mnist_cnn_*.keras"))
    if not all_saves:
        raise FileNotFoundError("No trained model found — run mnist_cnn.py first.")
    return tf.keras.models.load_model(all_saves[-1])

def preprocess_pil_image(img: Image.Image) -> np.ndarray:
    """
    Takes a PIL Image (any size, white background, black strokes) and
    returns a (1,28,28,1) float32 array ready for prediction.
    """
    img = img.convert("L")                 # to greyscale
    img = ImageOps.invert(img)             # white digits on black -> invert
    img = img.resize((28,28), Image.LANCZOS)
    arr = np.asarray(img).astype("float32")/255.0
    arr = arr.reshape(1,28,28,1)
    return arr

def predict_digit(img: Image.Image) -> int:
    model = load_last_model()
    arr   = preprocess_pil_image(img)
    pred  = model(arr, training=False)
    return int(np.argmax(pred, axis=1)[0])
