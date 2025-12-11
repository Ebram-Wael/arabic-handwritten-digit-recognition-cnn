import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from collections import defaultdict
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


model = load_model("best_model.h5")
image_size = (28,28)


def preprocess_image(img_path):
    img = Image.open(img_path).convert("L")
    img = img.resize(image_size)
    img_array = np.array(img)/255.0
    img_array = img_array.reshape(1,28,28,1)
    return img_array


dataset_folder = "../Data/dataset_split/test"

y_true = []
y_pred = []

class_correct = defaultdict(int)
class_total = defaultdict(int)


for class_name in os.listdir(dataset_folder):
    class_folder = os.path.join(dataset_folder, class_name)
    if not os.path.isdir(class_folder):
        continue

    for img_name in os.listdir(class_folder):
        img_path = os.path.join(class_folder, img_name)
        img_array = preprocess_image(img_path)
        preds = model.predict(img_array)
        cls = np.argmax(preds)
        
        # Append results
        y_true.append(int(class_name))
        y_pred.append(cls)
        
        # Update class-level counts
        class_total[int(class_name)] += 1
        if cls == int(class_name):
            class_correct[int(class_name)] += 1

print("\n--- Accuracy per class ---")
for cls in sorted(class_total.keys()):
    correct = class_correct[cls]
    total = class_total[cls]
    acc = (correct / total * 100) if total > 0 else 0
    print(f"Class {cls}: {correct}/{total} images correct → Accuracy: {acc:.2f}%")


y_true = np.array(y_true)
y_pred = np.array(y_pred)
overall_acc = np.sum(y_true == y_pred) / len(y_true) * 100
print(f"\nOverall Accuracy: {overall_acc:.2f}%")


cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()
