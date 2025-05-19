import numpy as np
from PIL import Image
import io

def preprocess_image(file):
    image = Image.open(io.BytesIO(file.read()))
    image = image.convert('L')  # Convert to grayscale
    image = image.resize((128, 128))  # Resize to match model input shape
    image_array = np.array(image) / 255.0  # Normalize
    image_array = image_array.reshape(1, 128, 128, 1)  # Reshape for CNN

    return image_array
