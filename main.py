from data.dataset import get_dataloaders
from data.visualization import show_images, show_data_distribution
from training.train import train_model, plot_training_history


def main():
    train_loader, val_loader, test_loader = get_dataloaders()
    class_names = train_loader.dataset.classes

    show_images(train_loader, class_names)
    show_data_distribution(class_names)

    train_loss, train_acc, val_loss, val_acc = train_model(train_loader, val_loader, class_names)

    plot_training_history(train_loss, train_acc, val_loss, val_acc)


if __name__ == "__main__":
    main()