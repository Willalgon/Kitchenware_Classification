import os
import time
import random
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score, f1_score
import numpy as np

from data_loader import get_loaders
from models.baseline import BaselineCNN
from models.advanced_model import build_transfer_model
from models.improved_cnn import build_improved_model
from models.free_model import build_free_model


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluar_modelo():
    set_seed(42)

    # 1. Menú de configuración
    print("\n---- CONFIGURACIÓN DE LA EVALUACIÓN ----")
    print("Escoge el modelo:\n(baseline / improved_guille / improved_samuel / improved_alvaro / resnet50 / efficientnet_v2_s / vit_b_16 / free_model)")
    model_type = input().strip()
    print("\nEscoge el Dataset con el que fue entrenado:\n(group_dataset / odd_pooled / even_pooled / combined / public_kaggle)")
    dataset_name = input().strip()

    dataset_path = f"../datasets/{dataset_name}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Ejecutando en dispositivo: {device.type.upper()}")

    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)

    # 2. Cargar el test DataLoader
    loaders, info = get_loaders(dataset_path, transformation=False, batch_size=16)
    test_loader = loaders["test"]
    class_names = info["class_names"]

    # 3. Instanciar el modelo exacto que el usuario ha elegido
    if model_type == "baseline":
        model = BaselineCNN(num_classes=info["num_classes"])
    elif model_type in ["improved_guille", "improved_samuel", "improved_alvaro"]:
        model = build_improved_model(model_type, num_classes=info["num_classes"]).to(device)
    elif model_type in ["resnet50", "efficientnet_v2_s", "vit_b_16"]:
        model = build_transfer_model(model_type, num_classes=info["num_classes"])
    elif model_type == "free_model":  # ← AÑADIDO
        model = build_free_model(num_classes=info["num_classes"])
    else:
        print(f"\nError: La arquitectura '{model_type}' no está soportada.")
        return

    model = model.to(device)
    ruta_modelo = f"saved_models/best_{model_type}_{dataset_name}.pth"

    # 4. Cargar los pesos entrenados
    try:
        model.load_state_dict(torch.load(ruta_modelo, map_location=device))
        print(f"Pesos cargados correctamente desde {ruta_modelo}")
    except FileNotFoundError:
        print(f"\nError Crítico: No se encontró el archivo de pesos en {ruta_modelo}.")
        print("Asegúrate de haber ejecutado 'train.py' con esta misma configuración primero.")
        return

    model.eval()

    y_true = []
    y_pred = []

    print("\nEvaluando el modelo con el conjunto de TEST...")

    # 5. Bucle de inferencia con medición de tiempos
    start_time = time.time()

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    end_time = time.time()
    total_time = end_time - start_time
    time_per_batch = (total_time / len(test_loader)) * 1000

    # 6. Cálculo de métricas y guardado de reporte
    accuracy = accuracy_score(y_true, y_pred) * 100
    macro_f1 = f1_score(y_true, y_pred, average='macro') * 100
    report = classification_report(y_true, y_pred, target_names=class_names)

    ruta_reporte = os.path.join(out_dir, f"report_{model_type}_{dataset_name}.txt")
    with open(ruta_reporte, "w", encoding="utf-8") as f:
        f.write(f"==================================================\n")
        f.write(f"RESULTADOS GLOBALES: {model_type.upper()} | DATASET: {dataset_name}\n")
        f.write(f"==================================================\n")
        f.write(f"ACCURACY GLOBAL:       {accuracy:.2f}%\n")
        f.write(f"MACRO F1-SCORE:        {macro_f1:.2f}%\n")
        f.write(f"Tiempo de inferencia:  {total_time:.2f} segundos\n")
        f.write(f"Tiempo medio/batch:    {time_per_batch:.2f} ms\n")
        f.write(f"==================================================\n\n")
        f.write("REPORTE DE CLASIFICACIÓN DETALLADO:\n")
        f.write(report)

    print("\n" + "=" * 50)
    print(f"ACCURACY GLOBAL: {accuracy:.2f}%")
    print(f"MACRO F1-SCORE:  {macro_f1:.2f}%")
    print("=" * 50)
    print(report)
    print(f"Tiempo de inferencia ({time_per_batch:.2f} ms/batch)")

    # 7. Matriz de confusión visual
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45, values_format='d')

    plt.title(f"Matriz de Confusión - {model_type.upper()} ({dataset_name})", pad=20)
    plt.tight_layout()

    ruta_imagen = os.path.join(out_dir, f"confusion_matrix_{model_type}_{dataset_name}.png")
    plt.savefig(ruta_imagen, dpi=300)

    print(f"\nMatriz de confusión guardada en: {ruta_imagen}")
    print(f"Reporte de métricas guardado en: {ruta_reporte}")
    print("Consejo de ingeniería: Revisa la imagen para ver el sesgo entre clases y coméntalo en la memoria.")


if __name__ == "__main__":
    evaluar_modelo()