import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights


def print_model_info(name, model, pretrained, frozen):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n======================================")
    print(f"MODELO CARGADO: {name}")
    print(f"======================================")
    print(f"   Pre-entrenado (ImageNet): {pretrained}")
    print(f"   Backbone Congelado:       {frozen}")
    print(f"   Parámetros Totales:       {total_params:,}")
    print(f"   Parámetros Entrenables:   {trainable_params:,}")
    print(f"======================================\n")


def build_free_model(num_classes=6):
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    # congelar el backbone en la fase 1
    for param in model.parameters():
        param.requires_grad = False

    # Cabeza mejorada respecto al advanced: añadimos capa intermedia 2048→512→num_classes
    in_features = model.fc.in_features  # 2048
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.4),
        nn.Linear(512, num_classes)
    )
    # La cabeza nueva siempre tiene requires_grad=True por defecto

    print_model_info("FreeModel — ResNet-50 Fine-Tuning 2 Fases", model,
                     pretrained=True, frozen=True)
    return model


def unfreeze_layer4(model):
    for name, param in model.named_parameters():
        if "layer4" in name or "fc" in name:
            param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[INFO] Fase 2 activada — layer4 descongelada.")
    print(f"       Parámetros entrenables ahora: {trainable:,}\n")