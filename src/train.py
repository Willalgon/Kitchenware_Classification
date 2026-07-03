import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from sklearn.metrics import f1_score
import time

from data_loader import get_loaders
from models.baseline import BaselineCNN
from models.advanced_model import build_transfer_model
from models.improved_cnn import build_improved_model, FocalLoss
from models.free_model import build_free_model, unfreeze_layer4

import random
import numpy as np


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo actual: {device}")

    print("\n---- CONFIGURACION DEL EXPERIMENTO ----")
    print(
        "Escoge el modelo:\n(baseline / improved_guille / improved_samuel / improved_alvaro / resnet50 / efficientnet_v2_s / vit_b_16 / free_model)")
    model_type = input().strip()
    print(
        "\nEscoge el Dataset:\n(group_dataset / odd_pooled / even_pooled / combined / public_kaggle / G00_dataset_split_Reference)")
    dataset_name = input().strip()

    dataset_path = f"../datasets/{dataset_name}"

    num_epochs = 50
    batch_size = 16
    set_seed(42)

    os.makedirs("results", exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)

    transform = False if model_type == "baseline" else True
    loaders, info = get_loaders(dataset_path, transformation=transform, batch_size=batch_size)
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    if model_type == "baseline":
        model = BaselineCNN(num_classes=info["num_classes"]).to(device)
    elif model_type in ["improved_guille", "improved_samuel", "improved_alvaro"]:
        model = build_improved_model(model_type, num_classes=info["num_classes"]).to(device)
    elif model_type in ["resnet50", "efficientnet_v2_s", "vit_b_16"]:
        model = build_transfer_model(model_type, num_classes=info["num_classes"]).to(device)
    elif model_type == "free_model":
        model = build_free_model(num_classes=info["num_classes"]).to(device)
    else:
        print(f"Error: modelo '{model_type}' no existe.")
        return

    #  Criterion / Optimizer / Scheduler por modelo
    scheduler = None
    FASE2_EPOCH = None
    WARMUP_EPOCHS = None

    if model_type == "baseline":
        # Sin cambios — configuración original exacta
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    elif model_type == "improved_guille":
        # EJE: REGULARIZACIÓN
        # - Data augmentation: activado via transformation=True en get_loaders
        # - Dropout reducido (0.20 vs 0.40 del baseline) — definido en ImprovedGuille
        # - weight_decay mayor (1e-3 vs 1e-4) — penaliza pesos grandes
        # - AdaptiveAvgPool elimina el overfitting de la capa fc gigante
        print("\n[INFO] Guille — Regularización: dropout=0.20, weight_decay=1e-3, data augmentation ON")
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5)

    elif model_type == "improved_samuel":
        # EJE: OPTIMIZACIÓN
        # - AdamW con lr=3e-4 (más estable que Adam con lr=0.001)
        # - CosineAnnealingLR: reduce lr suavemente hasta eta_min
        # - Kaiming init: arranque más rápido (definido en ImprovedSamuel)
        print("\n[INFO] Samuel — Optimización: AdamW + CosineAnnealingLR + Kaiming init")
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=1e-6
        )

    elif model_type == "improved_alvaro":

        # EJE: FUNCIÓN DE PÉRDIDA con warm-up

        # - Primeras WARMUP_EPOCHS épocas: CrossEntropy pura para estabilizar

        # - Resto: FocalLoss(gamma=1.0) con Label Smoothing integrado

        # - Esto evita que FocalLoss destruya el aprendizaje temprano

        print("\n[INFO] Álvaro — FocalLoss(gamma=1.0) con warm-up de 5 épocas en CrossEntropy")
        WARMUP_EPOCHS = 5
        criterion_warmup = nn.CrossEntropyLoss()
        criterion_focal = FocalLoss(gamma=1.0, label_smoothing=0.05)
        criterion = criterion_warmup  # empieza con CE
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    elif model_type == "vit_b_16":
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    elif model_type == "free_model":
        print("\n[INFO] Free Model — ResNet-50 fine-tuning 2 fases")
        print("       Fase 1 (épocas 1-10): solo cabeza | Fase 2 (épocas 11+): cabeza + layer4")
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        # Fase 1: lr alto solo para la cabeza
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-3, weight_decay=1e-2
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=1e-6
        )
        FASE2_EPOCH = 10  # época en que se activa el fine-tuning

    else:
        # resnet50, efficientnet_v2_s
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    train_losses, val_losses = [], []
    train_accs, val_accs, val_f1s = [], [], []
    mejor_f1 = 0.0
    paciencia = 12
    epocas_sin_mejora = 0

    print(f"\nIniciando entrenamiento de {model_type}...")
    start_time = time.time()

    for epoch in range(num_epochs):

        if model_type == "free_model" and FASE2_EPOCH is not None and epoch == FASE2_EPOCH:
            unfreeze_layer4(model)
            optimizer = torch.optim.AdamW([
                {'params': model.layer4.parameters(), 'lr': 1e-5},  # backbone: lr mínimo
                {'params': model.fc.parameters(), 'lr': 1e-4},  # cabeza: lr normal
            ], weight_decay=1e-2)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=num_epochs - FASE2_EPOCH, eta_min=1e-7
            )
            print("[INFO] Optimizer y scheduler reiniciados para Fase 2.")

        # Entrenamiento
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        avg_train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train
        train_losses.append(avg_train_loss)
        train_accs.append(train_acc)

        # Validación
        model.eval()
        running_val_loss, correct_val, total_val = 0.0, 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss_v = criterion(outputs, labels)
                running_val_loss += loss_v.item()
                _, preds = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (preds == labels).sum().item()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = running_val_loss / len(val_loader)
        val_acc = 100 * correct_val / total_val
        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        val_f1s.append(macro_f1)

        print(
            f"Epoca [{epoch + 1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val F1-Score: {macro_f1:.4f}")

        # Scheduler step
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(macro_f1)  # necesita la métrica
            else:
                scheduler.step()

        # Early Stopping + guardar mejor modelo
        if macro_f1 > mejor_f1:
            mejor_f1 = macro_f1
            torch.save(model.state_dict(), f"saved_models/best_{model_type}_{dataset_name}.pth")
            epocas_sin_mejora = 0
        else:
            epocas_sin_mejora += 1
            if model_type != "baseline" and epocas_sin_mejora >= paciencia:
                print(f"\nEarly Stopping activado. No hubo mejoras en {paciencia} epocas consecutivas.")
                print("Deteniendo el entrenamiento para prevenir sobreajuste.")
                break

    print(f"\nEntrenamiento finalizado. Mejor Macro F1: {mejor_f1:.4f}")
    end_time = time.time()
    total_time_seconds = end_time - start_time
    minutos = int(total_time_seconds // 60)
    segundos = int(total_time_seconds % 60)
    print(f"Tiempo total de entrenamiento: {minutos}m {segundos}s")

    # Gráficas
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    ax1.plot(train_losses, color='blue', marker='o', label='Train Loss')
    ax1.plot(val_losses, color='red', marker='x', label='Val Loss')
    ax1.set_title('Curvas de Perdida (Loss)')
    ax1.set_xlabel('Epoca');
    ax1.set_ylabel('Loss');
    ax1.legend()

    ax2.plot(train_accs, color='blue', marker='o', label='Train Acc')
    ax2.plot(val_accs, color='red', marker='x', label='Val Acc')
    ax2.set_title('Curvas de Accuracy')
    ax2.set_xlabel('Epoca');
    ax2.set_ylabel('Accuracy (%)');
    ax2.legend()

    ax3.plot(val_f1s, color='green', marker='s', label='Val F1')
    ax3.set_title('Rendimiento en Validacion')
    ax3.set_xlabel('Epoca');
    ax3.set_ylabel('Macro F1-Score');
    ax3.legend()

    plt.suptitle(f'Evolucion: {model_type} en Dataset {dataset_name}')
    fig.tight_layout()
    plt.savefig(f"results/learning_curve_{model_type}_{dataset_name}.png")
    print("Grafica de diagnostico guardada exitosamente en la carpeta 'results'.")


if __name__ == "__main__":
    main()