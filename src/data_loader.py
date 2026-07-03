import os
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split


# Definimos transformaciones (Se mantiene intacto)
def get_transforms(augment=False):
    if augment:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            # Sin normalización ImageNet — no aplica para redes desde cero
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])


batch_size = 16  # requerimiento del proyecto


def get_loaders(dataset_path, transformation=False, batch_size=batch_size):
    # Comprobamos si el dataset ya viene pre-dividido físicamente
    has_splits = (
            os.path.isdir(os.path.join(dataset_path, 'train')) and
            os.path.isdir(os.path.join(dataset_path, 'val')) and
            os.path.isdir(os.path.join(dataset_path, 'test'))
    )

    if has_splits:
        print(f"\n[INFO] Detectada estructura pre-dividida (train/val/test) en: {dataset_path}")

        # Cargamos directamente desde las subcarpetas
        train_dataset = datasets.ImageFolder(root=os.path.join(dataset_path, 'train'),
                                             transform=get_transforms(augment=transformation))
        val_dataset = datasets.ImageFolder(root=os.path.join(dataset_path, 'val'),
                                           transform=get_transforms(augment=False))
        test_dataset = datasets.ImageFolder(root=os.path.join(dataset_path, 'test'),
                                            transform=get_transforms(augment=False))

        # Obtenemos la información de las clases usando el dataset de train
        num_classes = len(train_dataset.classes)
        class_names = train_dataset.classes
        class_to_idx = train_dataset.class_to_idx

        total_size = len(train_dataset) + len(val_dataset) + len(test_dataset)
        train_size = len(train_dataset)
        val_size = len(val_dataset)
        test_size = len(test_dataset)

    else:
        print(f"\n[INFO] Detectada estructura plana. Dividiendo al vuelo: {dataset_path}")
        data = datasets.ImageFolder(root=dataset_path)
        all_indices = list(range(len(data))) # generamos una lista con numeros del 0 al len(data) - 1
        all_labels = data.targets # lista con la etiqueta de cada imagen

        # Primer split
        train_val_indices, test_indices, train_val_labels, _ = train_test_split(
            all_indices, all_labels, test_size=0.15, random_state=42, stratify=all_labels
        )
        # Segundo split
        train_indices, val_indices = train_test_split(
            train_val_indices, test_size=0.15/0.85, random_state=42, stratify=train_val_labels
        )

        train_full = datasets.ImageFolder(root=dataset_path, transform=get_transforms(augment=transformation))
        val_test_full = datasets.ImageFolder(root=dataset_path, transform=get_transforms(augment=False))

        train_dataset = Subset(train_full, train_indices)
        val_dataset = Subset(val_test_full, val_indices)
        test_dataset = Subset(val_test_full, test_indices)

        num_classes = len(data.classes)
        class_names = data.classes
        class_to_idx = data.class_to_idx

        total_size = len(data)
        train_size = len(train_dataset)
        val_size = len(val_dataset)
        test_size = len(test_dataset)

    # 5. DataLoaders (Comunes para ambas opciones)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available()
    )

    loaders = {"train": train_loader, "val": val_loader, "test": test_loader}

    info = {
        "num_classes": num_classes,
        "class_names": class_names,
        "class_to_idx": class_to_idx,
        "sizes": {
            "total": total_size,
            "train": train_size,
            "val": val_size,
            "test": test_size,
        }
    }

    print(f"   Clases ({info['num_classes']}): {info['class_names']}")
    print(f"   Total: {info['sizes']['total']} imágenes")
    print(f"   Train: {info['sizes']['train']} | Val: {info['sizes']['val']} | Test: {info['sizes']['test']}")

    return loaders, info