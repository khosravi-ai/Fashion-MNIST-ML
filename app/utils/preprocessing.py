from PIL import Image
import io
import  numpy as np
from numpy import ndarray

def preprocess_image(image_bytes : bytes ) -> ndarray:
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert("L")
    img = img.resize((28, 28))
    arr = np.array(img, dtype=np.float32)
    arr = 255 - arr
    arr /= 255.0
    return arr.flatten()



