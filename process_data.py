import os
import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

data_dir = "Animal-10-split"
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")
test_dir = os.path.join(data_dir, "test")

train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

val_test_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
val_dataset = datasets.ImageFolder(val_dir, transform=val_test_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform=val_test_transforms)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def denormalize(tensor):
    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    return tensor * std + mean


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


class_names = train_dataset.classes
show_images(train_loader, class_names)

train_counts = [len(os.listdir(os.path.join(train_dir, cls))) for cls in class_names]
val_counts = [len(os.listdir(os.path.join(val_dir, cls))) for cls in class_names]
test_counts = [len(os.listdir(os.path.join(test_dir, cls))) for cls in class_names]

print("Кількість зображень у кожному класі:")
for i, cls in enumerate(class_names):
    print(f"Клас {cls}:")
    print(f"  Train: {train_counts[i]}")
    print(f"  Validation: {val_counts[i]}")
    print(f"  Test: {test_counts[i]}")

total_train = sum(train_counts)
total_val = sum(val_counts)
total_test = sum(test_counts)
total_images = total_train + total_val + total_test

print("\nЗагальна кількість зображень:")
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