import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical # type: ignore
import matplotlib.pyplot as plt

def load_images_from_dir(directory, label):
    images = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(directory, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (128, 128))
                images.append(img)
                labels.append(label)
    return images, labels

def preprocess_data():
    # Define directories
    benign_dir = '/Users/somyashandilya/Desktop/CareCheck_Backend/dataset/Benign'
    malignant_dir = '/Users/somyashandilya/Desktop/CareCheck_Backend/dataset/Malignant'
    normal_dir = '/Users/somyashandilya/Desktop/CareCheck_Backend/dataset/Normal'

    # Load the data
    benign_images, benign_labels = load_images_from_dir(benign_dir, label=0)  # Benign: 0
    malignant_images, malignant_labels = load_images_from_dir(malignant_dir, label=1)  # Malignant: 1
    normal_images, normal_labels = load_images_from_dir(normal_dir, label=2)  # Normal: 2

    # Combine all data
    images = np.array(benign_images + malignant_images + normal_images)
    labels = np.array(benign_labels + malignant_labels + normal_labels)

    # Normalize pixel values to [0, 1]
    images = images / 255.0

    # One-hot encode the labels
    labels = to_categorical(labels, num_classes=3)

    # Split data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Reshape data for CNN
    X_train = X_train.reshape(-1, 128, 128, 1)
    X_val = X_val.reshape(-1, 128, 128, 1)
    X_test = X_test.reshape(-1, 128, 128, 1)
    
    # Verify data shapes
    print("Training Data Shape:", X_train.shape, y_train.shape)
    print("Validation Data Shape:", X_val.shape, y_val.shape)
    print("Test Data Shape:", X_test.shape, y_test.shape)

    # Visualize class distribution
    labels_count = [np.sum(labels[:, i]) for i in range(3)]  # Count for each class
    class_names = ['Benign', 'Malignant', 'Normal']

    plt.figure(figsize=(8, 6))
    plt.bar(class_names, labels_count, color='royalblue')
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.title('Class Distribution in Dataset')
    plt.show()
    
        # Define the categories and their corresponding image paths
    categories = {
        "Benign": benign_dir,
        "Malignant": malignant_dir,
        "Normal": normal_dir
    }

    # Iterate over categories
    for category, image_dir in categories.items():
        # Load images from the directory
        image_paths = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir) if filename.endswith(('.jpg', '.png'))]

        # Create subplots for each category
        fig, ax = plt.subplots(1, 3, figsize=(8, 8))
        ax = ax.ravel()

        # Randomly sample 3 images from each category
        for i, img_path in enumerate(np.random.choice(image_paths, size=3, replace=False)):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for correct display
            ax[i].imshow(img)
            ax[i].axis("off")
            ax[i].set_title(category)

        plt.show()

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    preprocess_data()
