import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import plot_tree
from sklearn.svm import SVC
from preprocess import preprocess_data

def evaluate_model(y_true, y_pred, model_name):
    print(f"Evaluating {model_name}...")
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("Classification Report:\n", classification_report(y_true, y_pred))

def evaluate():
    # Load the preprocessed dataset
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data()
    
    # Save test data for evaluation
    np.save('X_test.npy', X_test)
    np.save('y_test.npy', y_test)
    
    y_test_labels = np.argmax(y_test, axis=1)  # Convert one-hot to class labels
    
    # CNN Model Evaluation
    cnn_model = load_model('models/cnn_model.h5')
    cnn_preds = np.argmax(cnn_model.predict(X_test), axis=1)
    evaluate_model(y_test_labels, cnn_preds, 'CNN')
    
    # Load and evaluate other models
    model_paths = {
        'Random Forest': 'models/random_forest_model.pkl',
        'KNN': 'models/knn_model.pkl',
        'SVM': 'models/svm_model.pkl',
        'Logistic Regression': 'models/logistic_regression_model.pkl'
    }
    
    for model_name, path in model_paths.items():
        model = joblib.load(path)
        preds = model.predict(X_test.reshape(X_test.shape[0], -1))
        evaluate_model(y_test_labels, preds, model_name)
    
    print("Evaluation complete.")
    
    # Plot CNN Training Accuracy & Loss
    history = np.load('cnn_training_history.npy', allow_pickle=True).item()
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('CNN Accuracy Over Epochs')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('CNN Loss Over Epochs')
    plt.show()
    
    # Visualizing a Decision Tree from Random Forest
    rf_model = joblib.load('models/random_forest_model.pkl')
    plt.figure(figsize=(20, 10))
    plot_tree(rf_model.estimators_[0], filled=True, rounded=True)
    plt.title('Decision Tree from Random Forest')
    plt.show()
    
    # Decision Boundary for SVM
    pca = PCA(n_components=2)
    X_vis = pca.fit_transform(X_test.reshape(X_test.shape[0], -1))
    scaler = StandardScaler()
    X_vis_scaled = scaler.fit_transform(X_vis)
    
    svm_vis_model = SVC(kernel='linear')
    svm_vis_model.fit(X_vis_scaled, y_test_labels)
    
    plot_decision_boundary(svm_vis_model, X_vis_scaled, y_test_labels, 'SVM (2D)')
    
    # KNN Accuracy vs. K Value Graph
    k_values = list(range(1, 21))
    k_accuracies = []
    knn_model = joblib.load('models/knn_model.pkl')
    
    for k in k_values:
        knn_model.n_neighbors = k
        preds = knn_model.predict(X_test.reshape(X_test.shape[0], -1))
        k_accuracies.append(accuracy_score(y_test_labels, preds))
    
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, k_accuracies, marker='o', linestyle='--', color='b')
    plt.xlabel('Number of Neighbors (K)')
    plt.ylabel('Accuracy')
    plt.title('KNN Accuracy vs. K Value')
    plt.xticks(k_values)
    plt.show()
    
    # ROC Curve for Logistic Regression
    lr_model = joblib.load('models/logistic_regression_model.pkl')
    y_scores = lr_model.predict_proba(X_test.reshape(X_test.shape[0], -1))[:, 1]
    fpr, tpr, _ = roc_curve(y_test_labels, y_scores, pos_label=1)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {roc_auc:.2f})', color='darkorange')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Logistic Regression')
    plt.legend()
    plt.show()
    
    print("Graphical Evaluation Completed.")

def plot_decision_boundary(model, X, y, model_name):
    h = .02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', s=20)
    plt.title(f'Decision Boundary - {model_name}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

if __name__ == "__main__":
    evaluate()
