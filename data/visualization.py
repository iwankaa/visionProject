import matplotlib.pyplot as plt
import numpy as np
import os
from config import *
from .dataset import denormalize

def show_images(dataloader, class_names):
    images, labels = next(iter(dataloader))
    images = denormalize(images)
    fig, axes = plt.subplots(3, 5, figsize=(12, 8))
    for i, ax in enumerate(axes.flat):
        img = images[i].permute(1, 2, 0).numpy()
        label = class_names[labels[i]]
        ax.imshow(img)
        ax.set_title(label)
        ax.axis("off")
    plt.show()

def show_data_distribution(class_names):
    train_counts = [len(os.listdir(os.path.join(TRAIN_DIR, cls))) for cls in class_names]
    val_counts = [len(os.listdir(os.path.join(VAL_DIR, cls))) for cls in class_names]
    test_counts = [len(os.listdir(os.path.join(TEST_DIR, cls))) for cls in class_names]

    print("Number of images in each class:")
    for i, cls in enumerate(class_names):
        print(f"Class {cls}:")
        print(f"  Train: {train_counts[i]}")
        print(f"  Validation: {val_counts[i]}")
        print(f"  Test: {test_counts[i]}")

    total_train = sum(train_counts)
    total_val = sum(val_counts)
    total_test = sum(test_counts)
    total_images = total_train + total_val + total_test

    print("\nTotal number of images:")
    print(f"Train: {total_train} ({total_train / total_images * 100:.2f}%)")
    print(f"Validation: {total_val} ({total_val / total_images * 100:.2f}%)")
    print(f"Test: {total_test} ({total_test / total_images * 100:.2f}%)")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(class_names, train_counts, color='blue', label="Train")
    ax.bar(class_names, test_counts, color='green', label="Test")
    ax.bar(class_names, val_counts, color='orange', label="Validation")
    plt.xticks(rotation=45)
    plt.ylabel("Number of images")
    plt.title("Image distribution across classes")
    plt.legend()
    plt.show()