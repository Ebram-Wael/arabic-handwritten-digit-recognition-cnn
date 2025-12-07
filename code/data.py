import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


dataset_dir = os.path.join(os.path.dirname(__file__), "..", "Data", "dataset_split")
image_size = (28, 28)
num_classes = 10



def degrade_image(img_array, scale=0.5):
    img = Image.fromarray((img_array*255).astype(np.uint8))
    new_size = (int(img.width*scale), int(img.height*scale))
    img_small = img.resize(new_size, Image.BILINEAR)
    img_bad = img_small.resize((img.width, img.height), Image.BILINEAR)
    return np.array(img_bad)/255.0


def add_noise(img_array, noise_level=0.1):
    noise = np.random.normal(0, noise_level, img_array.shape)
    img_noisy = img_array + noise
    return np.clip(img_noisy, 0, 1)



def load_images(folder_path):
    images = []
    labels = []
    for class_idx in range(num_classes):
        class_folder = os.path.join(folder_path, str(class_idx))
        for img_name in os.listdir(class_folder):
            img_path = os.path.join(class_folder, img_name)
            img = Image.open(img_path).convert('L')  # Grayscale
            img_resized = img.resize(image_size)
            img_array = np.array(img_resized) / 255.0  # Scale
            images.append(img_array)
            labels.append(class_idx)
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)
    images = images.reshape(-1, image_size[0], image_size[1], 1)  # for CNN
    return images, labels



# clean dataset
X_train, y_train = load_images(os.path.join(dataset_dir, "train"))
X_val, y_val     = load_images(os.path.join(dataset_dir, "valid_train"))
X_test, y_test   = load_images(os.path.join(dataset_dir, "test"))


# low-quality version

X_train_bad = np.array([add_noise(degrade_image(x.squeeze()), noise_level=0.05) for x in X_train]).reshape(-1,28,28,1)
X_val_bad   = np.array([add_noise(degrade_image(x.squeeze()), noise_level=0.05) for x in X_val]).reshape(-1,28,28,1)
X_test_bad  = np.array([add_noise(degrade_image(x.squeeze()), noise_level=0.05) for x in X_test]).reshape(-1,28,28,1)


def get_datasets():
    return {
        "train": (X_train, y_train),
        "val":   (X_val, y_val),
        "test":  (X_test, y_test),

        "train_bad": X_train_bad,
        "val_bad":   X_val_bad,
        "test_bad":  X_test_bad
    }
