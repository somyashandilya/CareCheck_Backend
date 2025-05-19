import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from preprocess import preprocess_data
import joblib

def build_cnn_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_models():
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data()

    # CNN Model
    cnn_model = build_cnn_model()
    history = cnn_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=17, batch_size=32)
    cnn_model.save('models/cnn_model.h5')
    np.save('cnn_training_history.npy', history.history)  # Save training history


    # Flatten data for other models
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    # Random Forest Model
    rf_model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    rf_model.fit(X_train_flat, y_train.argmax(axis=1))  # Convert one-hot to class indices
    joblib.dump(rf_model, 'models/random_forest_model.pkl')
    
    # KNN Model
    knn_model = KNeighborsClassifier(n_neighbors=10)
    knn_model.fit(X_train_flat, y_train.argmax(axis=1))
    joblib.dump(knn_model, 'models/knn_model.pkl')

    # SVM Model with low regularization
    svm_model = SVC(kernel='linear', C=0.001)  
    svm_model.fit(X_train_flat, y_train.argmax(axis=1))
    joblib.dump(svm_model, 'models/svm_model.pkl')

    # Logistic Regression Model
    lr_model = LogisticRegression(C=0.001, max_iter=500)
    lr_model.fit(X_train_flat, y_train.argmax(axis=1))
    joblib.dump(lr_model, 'models/logistic_regression_model.pkl')

    print("Models trained and saved successfully.")

if __name__ == "__main__":
    train_models()
