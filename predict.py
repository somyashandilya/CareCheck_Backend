import tensorflow as tf
import cv2
import os
import numpy as np
import joblib

def predict_with_cnn(image_path):
    cnn_model = tf.keras.models.load_model('models/cnn_model.h5')
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = img.reshape(1, 128, 128, 1)

    prediction = cnn_model.predict(img)
    class_idx = np.argmax(prediction)
    classes = ['Benign', 'Malignant', 'Normal']
    return classes[class_idx]

def predict_with_sklearn_model(model_path, image_path):
    model = joblib.load(model_path)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = img.reshape(1, -1)

    prediction = model.predict(img)
    classes = ['Benign', 'Malignant', 'Normal']
    return classes[prediction[0]]

def predict_image(image_path, model_type='cnn'):
    if model_type == 'cnn':
        result = predict_with_cnn(image_path)
    else:
        model_map = {
            'random_forest': 'models/random_forest_model.pkl',
            'knn': 'models/knn_model.pkl',
            'svm': 'models/svm_model.pkl',
            'logistic_regression': 'models/logistic_regression_model.pkl',
        }
        if model_type in model_map:
            result = predict_with_sklearn_model(model_map[model_type], image_path)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    print(f"Prediction result: {result}")

def predict_from_folder(test_folder, model_type='cnn'):
    for filename in os.listdir(test_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(test_folder, filename)
            print(f"Predicting image: {image_path}")
            predict_image(image_path, model_type)

if __name__ == "__main__":
    test_folder = '/Users/somyashandilya/Desktop/CareCheck_Backend/Test'  # Test folder path
    model_type = 'cnn'  # Change this to the model you want to use, e.g., 'random_forest', 'knn', 'svm', 'logistic_regression'
    predict_from_folder(test_folder, model_type)
