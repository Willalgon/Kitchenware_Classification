import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    ResNet50_Weights,
    EfficientNet_V2_S_Weights,
    ViT_B_16_Weights
)

def print_model_info(name: str, model: nn.Module, pretrained: bool, frozen: bool):

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


def build_resnet50(num_classes = 6, pretrained = True, freeze_backbone= True):
    if pretrained:
        weights = ResNet50_Weights.IMAGENET1K_V1
    else:
        weights = None

    model = models.resnet50(weights=weights)

    # Congelamos el backbone PRIMERO (antes de cambiar la capa final)
    if freeze_backbone:
        for name, param in model.named_parameters():
            # Congelamos todo lo que no sea la capa final
            if "fc" not in name:
                param.requires_grad = False

    # Reemplazamos la capa final (fc): 2048 → num_classes
    # Como la creamos nueva, por defecto sus parámetros SÍ tendrán requires_grad=True
    in_features = model.fc.in_features  # 2048
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )

    print_model_info("ResNet-50", model, pretrained, freeze_backbone)
    return model


def build_efficientnet_v2(num_classes: int = 6, pretrained: bool = True, freeze_backbone: bool = True):
    weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_v2_s(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # En EfficientNet, la capa final se llama 'classifier', y es un Sequential
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes)
    )

    print_model_info("EfficientNet-V2-S", model, pretrained, freeze_backbone)
    return model


def build_vit_b_16(num_classes: int = 6, pretrained: bool = True, freeze_backbone: bool = True):

    weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.vit_b_16(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.heads.head.in_features
    model.heads = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )

    print_model_info("ViT-B/16", model, pretrained, freeze_backbone)
    return model

def build_transfer_model(architecture, num_classes = 6,
                         pretrained = True, freeze_backbone = True):
    builders = {
        "resnet50":          build_resnet50,
        "efficientnet_v2_s": build_efficientnet_v2,
        "vit_b_16":          build_vit_b_16,
    }

    architecture = architecture.lower()
    if architecture not in builders:
        raise ValueError(
            f"Arquitectura '{architecture}' no soportada. "
            f"Elige entre: {list(builders.keys())}"
        )

    return builders[architecture](
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
    )