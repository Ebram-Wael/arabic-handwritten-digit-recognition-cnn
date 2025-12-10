import os
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json

from itertools import cycle
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import learning_curve
from itertools import cycle

from data import get_datasets

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model.h5")
REPORT_FOLDER = os.path.join(os.path.dirname(__file__), "reports")

data_dict = get_datasets()
X_test, y_test = data_dict["test"]
X_val, y_val = data_dict["val"]

try:
    best_model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Successfully loaded model from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading the model: {e}")

y_scores = best_model.predict(X_test)
y_preds = y_scores.argmax(axis=1)

print(f"shape of y_score: {y_scores.shape}")
print(f"shape of y_pred: {y_preds.shape}")

def plot_learning_curve(history):
    acc = history['accuracy']
    val_acc = history['val_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo-', label='Training acc')
    plt.plot(epochs, val_acc, 'ro-', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Training loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    save_png = os.path.join(REPORT_FOLDER, "learning_curve.png")
    plt.savefig(save_png)
    plt.show()
    plt.close()

def multi_class_evaluation(y_test, y_preds, y_scores):
    class_names = [str(i) for i in range(10)]

    accuracy = accuracy_score(y_test, y_preds)
    print(f"\nEvaluation Metrics ")
    print(f"Model Accuracy: {accuracy:.4f}\n")

    print("Classification Report\n")
    classification_report(y_test, y_preds, target_names=class_names)

    conf_metrix = confusion_matrix(y_test, y_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        conf_metrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.title("Confusion Matrix (Arabic Numbers Multiclass Classification)")
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')

    save_png = os.path.join(REPORT_FOLDER, "confusion_matrix_for_arabic_number_classification_model.png")
    plt.savefig(save_png)
    plt.show()
    plt.close()

if __name__ == "__main__":
    HISTORY_PATH = os.path.join(os.path.dirname(__file__), "training_history.json")
    try:
        with open(HISTORY_PATH, 'r') as f:
            history = json.load(f)
        plot_learning_curve(history)
        print(f"Successfully loaded and plotted learning curve from {HISTORY_PATH}")
    except FileNotFoundError:
        print(f"History file not found at {HISTORY_PATH}. Skipping learning curve plot.")
    except Exception as e:
        print(f"Error loading or plotting the learning curve: {e}")

    multi_class_evaluation(y_test, y_preds, y_scores)