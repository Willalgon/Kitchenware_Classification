# Kitchenware Classification

---

## Requerimientos técnicos

### Stack Técnico

| Componente | Herramienta |
|---|---|
| Lenguaje | Python 3.11 / Python 3.12 |
| Entorno | Anaconda / Miniconda |
| Deep Learning | Pytorch |
| Visión | Torchvision |
| ML | ScikitLearn |

### IDE y Git

- Se recomienda PyCharm Professional (gratis con el correo de la universidad).
- Es obligatorio usar GitLab para el control de versiones

---

## Datasets a utilizar

| # | Dataset | Descripción |
|---|---|---|
| 1 | Dataset grupal propio | — |
| 2 | Dataset ODD | Fondo variable |
| 3 | Dataset EVEN | Fondo homogéneo |
| 4 | COMBINADO | ODD + EVEN |
| 5 | Dataset público | Kaggle — sólo las 6 clases mencionadas |

---

## Experimentos

### Experimento 1 — Baseline (Modelo Base)

CNN sencilla con 7 capas para establecer una puntuación de referencia.

| Parámetro | Valor |
|---|---|
| Entrada | 224 x 224 px |
| Optimizador | Adam (LR = 0.001) |
| Función de pérdida | CrossEntropy Loss |
| Épocas | 20 |
| Batch size | 16 |

**Arquitectura:** 3 bloques convolucionales (Conv2 + BatchNorm + ReLU + MaxPool + Dropout) y dos capas densas finales.

> **NOTA:** Propablemente sufriremos de overfitting

---

### Experimento 2 — Mejora del Baseline (Reparto de tareas)

Cada miembro del grupo asumen una responsabilidad:

#### Estudiante A — Regularizaciónn (Guillermo Álvarez)
Probar distintos niveles de Dropout, Weight Decay o Data Augmentation.

#### Estudiante B — Optimización (Samuel Llamas)
Comparar SGD con Momentum, Adam o usar Learning Rate Schedulers.

#### Estudiante C — Funciones de Pérdida (Álvaro Vigueras)
Probar Label Smoothing o Focal Loss.

---

### Experimento 3 — Arquitectura Avanzada

Implementar modelos más profundos o usar Transfer Learning con redes como ResNet, MobileNet o EfficientNet.

---

## Entrega

Se entrega un archivo ZIP nombrado como `AAA_Group02_Nombre_Practical.zip` que incluya:

1. **Presentación PowerPoint:** con al descripción de los DataSets, tabla de resultados, gráficas de entrenamiento (loss/accuracy) y matrices de confusión.
2. **Análisis Crítico:** debemos responder qué experimento fue mejor, que dataset fue más difícil y qué clases se confunden más.
3. **Código fuente:** todo el código comprimido (sin las imágenes para que no pese) e instrucciones para ejecutarlo.
4. **Tablas Obligatorias:** 4 tablas específicas: resultados principales, Mejoras del Experimento 2, Descripción de la arquitectura del Experimento 3 y coste computacional.

---

## Especificaciones de Entorno

El proyecto exige un entorno virtual específico para evitar conflictos. Procedimiento:

```bash
conda env create -f environment.yml
conda activate AAA_Kitchenware_Class
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

Utilizaremos PyCharm Pro de la cual tenemos la versión pro de forma gratuita.

---

## Estructura — GitLab

```
AAA_Kitchenware_Class/
│
├── datasets/
│   ├── combined/
│   ├── even_pooled/
│   ├── group_dataset/
│   ├── odd_pooled/
│   └── public_kaggle/
│
├── results/
│
├── src/
│   ├── models/
│   │   ├── advanced_model.py
│   │   ├── baseline.py
│   │   ├── improved_cnn.py
│   │   └── data_loader.py
│   ├── data_loader.py
│   ├── evaluate.py
│   └── train.py
│
├── .gitignore
└── README.md
```

- `datasets/` — Incluimos 5 directorios correspondientes a los 5 dataset sque utilizaremos: combined, even_pooled, group_dataset, odd_pooled, public_kaggle.
- `results/` — Donde almacenaremos los resultados de los modelos para su correspondiente análisis, estudio, y mejora.
- `src/` — Donde almacenaremos los scripts: data_loader.py, evaluate.py, train.py. Incluiremos también la carpeta `models/` en la cual incluiremos los scripts: advanced_model.py, baseline.py, improved_cnn.py, data_loader.py.
- `.gitignore` — Archivo oculto a la vista en gitlab donde ocultamos lo que no queremos visible, como por ejemplo los datasets.
- `README.md` — Por último desarrollamos este README explicativo de este proyecto a medida que avanzamos y picamos código. Queremos reflejar en este el comprendimiento de los modelos y cómo actúan respecto a nuestros datos. Queremos mediante el README llevar un constante análisis del proyecto con el cual cualquier persona sepa o no de la materia pueda llegar a entender el objetivo del deep learning. A su vez explicaremos todo requerimiento técnico y explicaciones técnicas necesarias para facilitar la comprensión tanto del profesorado que evaluará nuestro proyecto, como nuestro propio aprendizaje.

---

## Orden de implementación

Seguiremos este orden de trabajo:

1. `data_loader.py` — leeremos los datos y los transformaremos.
2. `model.py` — definiremos la estructura de nuestra red neuronal (capas, neuronas, ReLUs...).
3. `train.py` — unimos data_loader y model.py (el cual pertenezca) para crear un bucle de entrenamiento y hacer que la red aprenda.
4. `evaluate.py` — tras entrenar el modelo, evaluaremos el modelo para datos desconocidos.

---

## Instrucciones de ejecución de los experimentos
Como ya se ha comentado, el proyecto se ha dividido en una serie de scripts, para su estructurada elaboración y entendimiento del orden de ejecución de CNNs y redes neuronales.
1. Ejecutaremos el script `train.py`. Durante su ejecución se nos pedirá por pantalla introducir cuál dataset queremos usar, y posteriormente qué modelo queremos utilizar para entrenar este dataset seleccionado previamente.
Tras la finalización de la ejecución de `train.py`, se guardará en la carpeta `results` una imagen png que contiene los siguientes gráficos: `Curva de Loss (Train vs Val)`, `Curva de Accuracy (Train vs Val)`, `Curva de Macro F1-Score en validación`.
Así mismo, se almacenará en `saved_models/` los parámetros que mejor F1-Score obtuvieron. Este archivo se utilizará para, al momento de ejecutar `evaluate.py`, no volver a entrenar y simplemente cargar los parámetros para conseguir la matriz de confusión y métricas con el conjunto de test.
2. Por último, ejecutamos `evaluate.py` para obtener todas las métricas y matriz de confusión. Este carga ese modelo guardado y lo evalúa únicamente sobre el conjunto de test.
- `IMPORTANTE`: todos los experimentos los hemos realizado utilizando google collab ya que no disponemos de GPUs potentes en nuestros dispositivos. La única modificación que hemos hecho en el código para ejecutar los scripts en google collab es la siguiente:
`dataset_path = f"../datasets/{dataset_name}"` -> `dataset_path = "/content/dataset_local"` Si usted posee una tarjeta gráfica potente en su dispositivo, no hace falta que modifique nada del código y puede ejecutar los scripts en local.
- SE HA IMPLEMENTADO UNA SEMILLA PARA QUE EN CADA EJECUCIÓN DEL MISMO TIPO, EL RESULTADO SEA EL MISMO.

---

## Data Loader

Here we go! Empezamos a picar código.

Antes de pasar una imagen a una red neuronal hay que transformarla y procesarla. Creamos una función `get_transforms` en la cual definimos dos posibilidades:

- Si `augment==True` se aplica Data Augmentation. Con esto aplicamos variaciones a las fotos (las giramos, cambiamos su brillo, volteamos...). Con esto modificamos las fotos para poder darle más inputs al modelo y para evitar que la red aprenda de los únicos datos que tenemos produciendo overfitting.
- Si `augment==False` símplemente redimensionamos a 224x224 píxeles el cual es un requerimiento. Estas dimensiones son el estándar de modelos como ResNet. Tras dimensionarlos los convertimos en Tensores.

Ahora empezamos con la función más importante del Data_Loader: `get_loaders`.

Utilizamos `os`, un método que nos va a permitir encontrar si alguno de los datasets ya ha sido dividido en train, val y test. Utilizamos `has_splits` para encontrar, utilizando `os.path.isdir`, si existe un dataset en cuya ruta contenga `train`, `val` y `test`.

**Si `has_splits==True`**, hemos identificado que este dataset ya está dividido, por lo tanto nos encargaremos simplemente de cargar las imágenes y transformarlas.

A la función `dataset.ImageFolder` le adjuntamos la ruta en la que se encuentran las imágenes, y la función de transformación habiendo especificado en `get_loaders` si queremos una augmentation o no.

> ¿Cómo funciona esto? Simplemente la función identifica estas subcarpetas (tenedor, cuchara, ...) como las etiquetas (labels), nuestro objetivo. Después de identificar las etiquetas le asigna a cada una un índice (0, 1, 2...). Después crea una lista interna donde relaciona cada imagen con esta etiqueta.

Ahora vamos con el dato curioso. El transform no aplica los cambios a todas las imágenes diréctamente, sino que las prepara para que cada vez que se llamen se apliquen estos cambios.

Cuando nosotros estudiamos las CNN para el primer parcial, aprendimos a aplicar filtros a las imágenes, pooling... antes de pasar las imágenes a la red neuronal. Esto lo hacemos en el entrenamiento (`train.py`). Aquí simplemente con `ToTensor()` estamos generando matrices de números de dimensión `(C x H x W)` de cada imágen según la intensidad lumínica de cada Pixel. Es decir, nos estamos quedando con ese raw data numérico de tensores sin procesarlo ni aplicarle filtros.

- `train_dataset.classes` devuelve las etiquetas que encontró ImageFolder (cuchillos, cucharas...).
- `len(tran_dataset.classes)` obtiene el número total de etiquetas: **6**.
- `train_dataset.class_to_idx` devuelve un diccionario con los índices de cada etiqueta que ImageFolder había generado.

**Si `has_splits==False`** estamos ante un dataset sin split.

De este modo, primero crearemos el objeto `data` con `ImageFolder` para escanear qué hay en la carpeta sin aplicar transformaciones. `data.targets` devuelve una lista con el índice de la etiqueta a la que corresponde cada clase.

Ahora vamos a la división de fotos. Primero de todo tenemos que saber que con dividir SÓLO los índices es sufieciente ya que el objeto `data` las tiene conectadas a sus etiquetas correspondientes. Simplemente en el primer split dividimos también los labels para quedarnos con `train_val_labels` y luego usarlo para el `stratify` del segundo split. Tras hacer el primer split, ahora tenemos que dibidir el train y val en dos, y como el requerimiento del proyecto es que val también sea el 15 por ciento del total, esto se resume a `0.15/0.85`.

Ahora que ya tenemos la división correcta, aplicamos `ImageFolder` y sus transformaciones. Para el test de validación no queremos que haya un aumento.

> Tenemos que entender que la transofrmación no genera imágenes nuevas sino que transforma las existentes. Esto nos sirve para el entrenamiento ya que aleatoriamente en cada epoch tenemos imágenes distintas de las que aprender. Sin embargo en el modelo de validación preferimos que en cada epoch las imágenes sean las mismas para giarnos de una progresión real.

Llegó la hora de utilizar `Subset`. Tras crear dos objetos de `ImageFolder` correspondientes a los de train con sus transformaciones y a los de val y test sin ellas, haremos los subsets. A `Subset` se le pasan dos atributos: el dataset completo, y los índices que queremos separar. Es decir, adjuntamos el imagefolder que queremos, y los indices qeu queremos de todo el `data`.

Aquí por fin ya tenemos el objeto de train, val y test. Ahora solo falta obtener lo más importante, los loaders. Ya que tenemos todo el raw data en tensores, ahora queremos devolverlo para poder empezar a jugar con ello en el entrenamiento y poder pasárselo a un modelo y aplicar filtros y pooling.
`DataLoader` va a devolver una tupla. Por un lado el tensor de matrices numéricas corresponientes a cada imágen, y por otro lado un tensro con las etiquetas de cada imagen. Finalmente la función `get_loaders` devuelve todos los loaders y la infomación que hemos ido almacenando.  
**Autor:** `\Guillermo Álvarez González\`
---

## Models

### baseline
Creamos la clase BaselineCNN(). Heredamos de nn.Module lo cual es obligatorio en pytorch para que la clase herede todo lo necesario para funcionar como un ared neuronal.
`num_classes = 6`: definimos el número de clases que tenemos, y establecemos por defecto un `dropout_rate` del 0.5 para apagar el 50% de los canales y evitar el sobreajuste. 

Empezamos a definir las capas. Definiremos capas convolucionales, ya que estamos trabajando con imágenes.
A cada capa debemos pasarle como parámetros el in_channels y out_channels. Los canales de entrada de la primera capa serán 3, ya que las imágenes son RGB. El 32 es el número de filtros que la red aplica.
Con `padding=1` hacemos que la imagen no se encoja al pasar por los filtros añadiendo un borde de píxeles vacíos. `kernel_size` indica el tamaño del filtro. Establecemos 3 por lo tanto esa ventana/filtro que se aplica a la imagen es de tamaño 3x3

`BatchNorm2d(32)` normaliza los datos. Hace que el entrenamiento sea más estable y rápido, evitando que los valores se disparen.

Definimos los Pools para reducir el tamaño de las imágenes a la mitad. Si entra un 224x224 sale una 112x112. Se queda con sólo los píxeles más importantes de cada cuadrado 2x2

- Dropout2 apaga canales completos de la imagen.
- Dopout  apaga conexiones individuales. 
- Flatten convierte nuestra matriz (imagen) en una sola fila de números para pasárselos a la red nueronal.

Definimos la capa densa `nn.Linear(128 * 28 * 28, 256)`. Conecta los 128 canales de 28x28 píxeles con 256 neuronas.
Con la capa final reducimos esas 256 neuronas a 6 categorías finales.

Definimos el `forward`:
El modelo aplica 3 bloques de filtrado antes de pasar a la red neuronal. 
1. Primero `self.conv(x)` aplica el número de filtros correspondientes del tamaño correspondiente sobre la imagen. El kernel (filtro) se desliza sobre la matriz de la imagen realizando la suma ponderada de los píxeles.
Gracias a padding = 1 mantenemos el tamaño de la imagen.
2. `self.bn(x)`: batch_normalization. Se normaliza el aprendizaje (la salida del filtro) de la convolución para que tenga media 0 y varianza 1.
3. `F.relu(x)`: en las CNN cada número de la matrix es una neurona. Cuando aplicamos la convolución y la normalización la salida es una matriz llena de números (activaciones).
La ReLU se aplica a cada uno de esos números, al igual que se aplicaría a la salida de una neurona individual de una FC.
Con ReLU los valores negativos de la matriz se convierten a 0 y los positivos se quedan positivos.
4. `Pooling`: utilizamos Max Pooling. Como hemos definido un kernel de 2, el max pooling consistirá en analizar ventanas de 2x2 píxeles y quedándose con el valor máximo. Esto reduce las dimensiones a la mitad. Al definir un stride de 2 la ventana salta de dos en dos.
5. `dop2d`: dropout. Durante el entrenamiento apaga canales enteros aleatoriamente. Esto obliga al modelo a no aprender de un solo filtro para reconocer un objeto, mejorando la generalización. Este dropout no apaga sólo números de una matriz (canal) sino una matriz entera.
Como podemos ver, la primera capa convolucional tiene 3 canales ya que es RGB. Luego podemos aplicar el número de canales que queremos.
6. Tras aplicar estas 3 capas a las imágenes en el forward, terminamos flatten y la FC.
`Flatten()` básicamente lo que hace es convertir este volumen tridimensional 128x28x28 en un simple vector. 
Sabemos que 128 es el número de canales finales que hemos establecido, y 28x28 es el tamaño de cada canal (tamaño de cada matriz). Este flatten es necesario para poder utilkizar las capas finales FC.
7. `fc1`: capa lineal. Simplemente entramos en el mundo de las Fully connected utilizando la salida del aplanado Flatten() y de salida definimos 256 neuronas. Entonces tras las redes convolucionales el flatten convierte cada imagen en un tensor de x elementos. Cada elemento de ese tensor es una característica, teniendo x características. Esta será la entrada de la red neuronal y la salida de la primera capa será el número de neuronas estableciendo 256 neuronas.
8. Aplicamos `reLU` tras la primera capa. 
9. `Dropout` de alguna neurona tras la primera capa
10. Segunda y ultima capa full connected con las 6 clases como salida. 

Hemos de especificar que no hay que aplicar la fnción de activación final y debemos dejarlo como logits ya que la propia función de pérdida CCE calcula la función de activación `Softmax`.

Por último devolvemos la salida final en logits. Dado esto ya hemos definido toda la red del baseline utilizando la estructura pedida por Enrique. 

**Autor:** `\Guillermo Álvarez González\`

--

### improved_cnn
#### REGULARIZATION (Guillermo Álvarez)
1. AdaptiveAvgPool2d: En lugar del Flatten() directo del baseline, aplico un Global Average Pooling antes del clasificador. El baseline tenía fc1 = Linear(128*28*28, 256) con 25 millones de parámetros — una cantidad absurda para 452 imágenes de entrenamiento. Con AdaptiveAvgPool2d((1,1)) la capa queda fc1 = Linear(128, 256), reduciendo los parámetros totales de 25M a ~128K. Esto elimina el sobreajuste estructural de la arquitectura.
2. dropout_rate reducido (0.40 → 0.20): El baseline usaba un dropout del 40% tras la capa densa. Con un modelo ya regularizado por el pooling y el data augmentation, un dropout tan alto impedía aprender. Lo reduzco a 0.20 para mantener regularización sin bloquear la convergencia.
3. weight_decay aumentado (1e-4 → 1e-3): Configurado en el optimizador Adam en train.py. Penaliza los pesos grandes forzando al modelo a encontrar soluciones más simples y generalizables.
4. Data Augmentation (transformation=True en train.py): Activado a través del get_loaders. Aumenta artificialmente la diversidad del conjunto de entrenamiento sin recopilar más imágenes, haciendo el modelo más robusto a variaciones de orientación, color e iluminación.
5. ReduceLROnPlateau scheduler: Configurado en train.py con factor=0.5 y patience=5. Cuando el F1 de validación deja de mejorar durante 5 épocas consecutivas, divide el learning rate a la mitad automáticamente. Esto estabiliza la convergencia y elimina los saltos bruscos de la curva de validación.

#### OPTIMIZACION (Samuel Llamas)
1. Inicialización Kaiming (_initialize_weights): El baseline inicializa los pesos de forma aleatoria por defecto. Aplico kaiming_normal_ en todas las capas convolucionales, que es la inicialización matemáticamente óptima para activaciones ReLU. Garantiza que la varianza del gradiente se mantenga estable a lo largo de las capas desde la primera época, acelerando la convergencia inicial.
2. Optimizador AdamW (Adam → AdamW): Sustituyo Adam por AdamW en train.py. La diferencia clave es que AdamW aplica el weight decay de forma desacoplada del gradiente adaptativo, lo que en la práctica produce una regularización más efectiva y una generalización mejor que Adam con weight decay estándar.
3. CosineAnnealingLR scheduler: En lugar de un learning rate fijo, aplico una reducción coseno desde lr=3e-4 hasta eta_min=1e-6 a lo largo de todas las épocas. El lr cae suavemente siguiendo una curva coseno, lo que permite explorar bien el espacio de parámetros al principio y afinar con precisión al final del entrenamiento.
4. Learning rate ajustado (0.001 → 3e-4): El baseline usaba lr=0.001 con Adam, que en datasets pequeños produce oscilaciones. Reduzco a 3e-4, el valor estándar recomendado para AdamW en tareas de visión con datasets limitados.

#### FUNCION DE PÉRDIDA (Álvaro Vigueras)
1. La arquitectura es idéntica al baseline. Toda la mejora reside en cómo la red cuantifica y aprende de sus errores durante el entrenamiento.
2. Label Smoothing (0.1): La Cross-Entropy estándar entrena al modelo con etiquetas duras — clase correcta = 1.0, resto = 0.0. Con objetos visualmente similares como el menaje de cocina, esto hace que la red se vuelva excesivamente confiada y deje de extraer características discriminativas finas. Con label_smoothing=0.1 configurado en nn.CrossEntropyLoss, la clase correcta pasa a valer 0.90 y ese 0.10 restante se redistribuye entre las demás clases. Esto obliga al modelo a mantener cierta incertidumbre calibrada y a aprender representaciones más generalizables.
3. Focal Loss: Implementada como clase FocalLoss en improved_cnn.py. La Cross-Entropy trata todos los ejemplos por igual, pero en datasets pequeños y con clases difíciles (p.ej. cubiertos vs cuchillos), los ejemplos fáciles dominan el gradiente y el modelo apenas mejora en los casos complicados. La Focal Loss aplica un factor (1 - pt)^gamma que reduce el peso de los ejemplos que ya se clasifican correctamente con alta confianza (pt alto) y concentra el aprendizaje en los ejemplos difíciles. Con gamma=2.0, un ejemplo clasificado con 90% de confianza contribuye ~100 veces menos al gradiente que uno clasificado con 50%.
4. Selección automática en train.py: Al ejecutar improved_alvaro, el código sustituye automáticamente la función de pérdida por FocalLoss(gamma=2.0). La línea de Label Smoothing queda disponible comentada para facilitar la comparación experimental entre ambas variantes.
--

### advanced_model
Definiremos los modelos seleccionados en el estudio teórico

#### resnet50 (Guillermo Álvarez)
Definimos el modelo resnet50.
Primero definiremos el estudio teórico para comprender resnet50:
1. Preprocesamiento:
   * Las imágenes se transforman a 224x224 píxeles.
   * Los clores se normalizan y usamos técnicas transofmración de imágenes como voltearlas para darle más variedad al modelo
2. Descriptors(Como obtener características):
   * El modelo usa redes convolucionales (CNNs) pero con conexiones de salto (skip connections).
   * Esto actúa como recortes que permite que la información fluya más facilmente a traves de las 50 capas sin perderse o degradarse. Esto se deve que en el bacward, en el proceso de la cadena la informaión puede perderse entonces este modelo con las skip connections soluciona este problema.
3. Clasificación (tomar la decisión):
   * Al pasar por las 50 capas de extracción de características, los datos se promedian.
   * Al final, las capas softmax calculan matemáticamente la probabilidad de cada clase.

4. CODE:
- De resnet utilizamos IMAGENET1K_V1. Este es el conocimiento (pesos). El modelo ya ha visto millones de imágines y sabe distinguir mil categorías. Usamos pretrained=True para cargar este cerebro ya entrenado.
- Tras definir los pesos que vamos tenemos que heredar el modelo por supuesto. Llamamos model a resnet50(weights=pesos)
- `freeze_backbone`: el `backbone` son las cpaas convolucionales que extraen rasgos (bordes, texturas...)
Establecemos `requires_grad` a False ya que no queremos calcular los gradientes para estos parámetros. 
Esto lo queremos, ya que si ya sabe detectar platos, cucharas o cualquier otro utensilio de cocina, no queremos que lo olvide cuando empieze a entrenar nuestras fotos. Además, ahorramos mucha memoria y tiempo. 
model.parameters() devuelve una tupla para cada parámetro entrenado del modelo. 
- `name`: string. La ruta completa del parámetro.
- `param`: es un tensor. Este es el que almacena los pesos o sesgos, es decir, los valores del parámetro. 
- `param.requires_grad`: esto indica que para ese parámetro NO se haga el backward, es decir, no se calcule las derivadas de ese parámetro en respecto a la función de pérdida. 
- `"fc"`: en la arquitectura resnet las capas convolucionales se nombran como layer1, layer2... La única capa encargada de la clasificación final se llama `"fc"`. De esta forma, con esta condición, el condicional asegura que todas las capas de extracción de rasgos (las que no son la capa final) se congelen. 
Entonces: de esta forma, pasamos las imágenes por el backbone, sin aprender, simplemente pasando las imágenes por los pesos ya entrenados del modelo (convoluciones, normalización, ReLU...). En la última capa fc, recibe las x características y realiza la multiplicación final y=Wt*a+bpara generar las 6 salidas. Aquí es donde requires_grad=False tiene juego. Cuando calculamos la pérdida y llamamos a loss.backward, se calculan las derivadas desde la salida hacia la entrada usando la regla de la cadena. Haciendo lo que hemos hecho no se calculan estas derivadas. Es decir, en la actualización de parámetros el learning rate multiplica a 0 por lo tanto el parámetro antiguo suma a 0 quedando igual. 
Entonces cuando se llega a la última capa y se hace el backpropagation sólo se guardan los pesos y bias de la última capa, por lo tatno es de la única capa de la que se aprende. De este modo, todas las capas iniciales e intermedias han aprendido de bordes y formas con pesos ya entrenados de resnet y en la última capa es donde hacemos al modelo entrenar un poco del tipo de imagen que tenemos. 
¿Qué es lo que conseguimos con esto? Básicamente mantenemos la capacidad de visión de una red entrenada con millones de imágenes entonces no perdemos todo ese conocimiento aprendiendo sólo de lo nuestro. De esta forma, el modelo no necesita aprender qué es un reflejo metálico (ya lo sabe) simplemente necesita aprender que cuando detecta ese reflejho en nuestras fotos, probablemente está viendo un cuchillo o una cuchara. 
- Justo antes de la capa fc, la resNet-50 siempre entrega un vector de 2048 neuronas. Por eso, nosotros leemos ese valor con `model.fc.in_features` para asegurarnos de que nuestra nueva capa conecte perfectamente con lo que viene detrás. 
- Estamos trabajando con `Transfer Learning` por lo tanto a la arquitectura resnet50 de fábrica le hacemos una modificación en la última capa que es de la que vamos a querer cambiar los pesos. 
- Definimos la última capa de salida. Primero aplicamos un dropout para apagar el 30 % de las neuronas que llegan. Tras esto aplicamos la capa lineal con el numero de características de entrada y el número de neuronas de la capa que van a ser 6. 
- Finalmente devolvemos información del modelo que vamos a aplicar a nuestro conjunto de imágenes y devolvemos este modelo listo para usar.

**Autor:** `\Guillermo Álvarez González\`

#### efficientnet_v2 (Samuel Llamas)

Definimos el modelo EfficientNetV2, concretamente la versión Small(s) para mantener un equilibrio entre precisión y coste computacional.
- Estudio teórico para comprender EfficientNetV2:
  1. Compund Scaling(Escalado Compuesto): A diferencia de arquitecturas tradicionales como Resnet que mejoran simplemente añadiendo más y más capas(profunidad), EfficientNet fue diseñada por Google buscando la eficiencia matemática. Escala de forma inteligente y simultánea tres dimensiones: el ancho de la red (más canales), la profundidad (más capas) y la resolución de la imagen de entrada.
  2. Bloques Fused-MBConv: La gran novedad de la versión V2 es que utiliza unos bloques especiales en las primeras etapas de la red. Básicamente, fusiona varias operaciones convolucionales (que antes iban por separado) en una sola operación optimizada. Esto permite que la red aproveche al máximo la memoria caché del hardware, haciendo que entrene muchísimo más rápido que las CNNs tradicionales.

CODE:
    - Al igual que en los otros modelos, aplicamos Transfer Learning usando 'EfficientNet_V2_S_Weights.IMAGENET1K_V1'. Cargamos el 'cerebro' pre-entrenado para no partir de cero.
    - Congelación del Backbone: Usamos un bucle 'for param in model.parameters(): param.requires_grad = False' en cascada. Al congelar los gradientes, la red actúa como un embudo perfecto de extracción visual: la imagen entra, pasa por las capas convolucional eficientes sin alterar sus pesos, y sale convertida en un mapa de características altamente comprimido.
    - Modificación de la cabeza (Classifier): La arquitectura interna de EfficientNet nombra a su capa final como 'classifier' (a diferencia de ResNet que usa 'fc'). Esta capa es un bloque 'Sequential'.
    - Leemos cuántas características exactas entrega el backbone usando 'model.classifier[1].in_features' (suele ser 1280 en versión S).
    - Finalmente, aplastamos el clasificador orignal de 1000 clases y lo sustituimos por el nuestro: un bloque 'nn.Sequential' que primero inyecta un 'nn.Dropout(p=0.3)' para regularizar y apagar el 30% de las conexiones (forzando a la red a no depender de una sola característica de la foto), seguido de una capa 'nn.Linear' que condensa esa información matemática en nuestras 6 clases finales.

**Autor:** `\Samuel Llamas Martínez\`

#### transfer_model (Álvaro Vigueras)
Este modelo procesa la imagen dividiéndola en parches de pixeles de tamaño 16x16 y desde las primeras capas, usa mecanismos Self-Attention, lo que le permite entender la relacion espacial completa de los objetos.
La funcion build_vit_b_16 sigue principalmente tres fases:
1. Transfer learning e inicializacion. Esta corresponde a esta parte del código:
weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
model = models.vit_b_16(weights=weights)
Esta parte lo que hace es instanciar el modelo cargando los pesos pre entrenados más recientes del dataset llamado ImageNet-1K. Hay que hacerlo porque entrenar un transformer como este desde cero con nuestro volumen de datos limitado es inviable matemáticamente. La realización de esta parte, aporta a la red de capacidad robusta y previa para extraer características visuales de alto nivel.

2. Congelación del Backbone. Esta corresponde a la siguiente parte del código:
if freeze_backbone:
    for param in model.parameters():
        param.requires_grad = False
Necesitamos implementar esta parte ya que por defecto, la función congela todos los parámetros internos del modelo estableciendo requires_grad = False. Las razones de diseño son:
- Evitar que los gradientes iniciales, los cuales son ruidosos y altos, destruyan el conocimiento previo extraido de ImageNet.
- Reducir el coste de memoria en la GPU y el tiempo de computacion. Todo esto permite agilizar el ciclo de experimentación.

3. Rediseño de la cabeza de clasificación y regularización. Le corresponde esta parte del código:
in_features = model.heads.head.in_features
model.heads = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(in_features, num_classes)
)
El Vit tiene una arquitectura interna diferente a las CNNs estándar, ya que aquí la clasificación finañl recae sobre el atributo model.heads a diferencia de una capa fc (ResNet) o classifier (EfficientNet).
Nosotros tenemos 6 clases, y para adaptarlo a ellas y homogeneizar la estrategia de regularización con el resto del grupo, hemos reemplazado la capa original por un bloque nn.Sequential, que lo que hace es inyectar un dropout con p=0.3, es decir, apaga el 30% de las neuronas aleatoriamente duante la fase de entrenamiento, haciendo así que el modelo tenga que aprender representaciones robustas y reduantes evitando así el overfitting.

4. Integración en el patrón factoria. Toda esta lógica la registramos y encapsulamos dentro de build_tranfer_model. Logramos que podamos cambiar entre arquitecturas desde el archivo de configuración sin alterar el bucle de entrenamiento al mapear "vit_b_16" a la constructora.

**Autor:** `\Álvaro Vigueras Capablo\`

--

### free_model
#### ResNet-50 con Fine-Tuning Progresivo en 2 Fases (Guillermo Álvarez)

Este modelo parte de la misma arquitectura ResNet-50 del `advanced_model`, pero introduce una estrategia de entrenamiento en dos fases que permite adaptar las capas profundas del backbone al dominio específico del menaje de cocina, algo que el modelo avanzado no hace al mantener el backbone congelado permanentemente.

**Estudio teórico — Por qué el fine-tuning progresivo mejora al Transfer Learning estático:**

El problema del Transfer Learning clásico (el que usamos en `resnet50`) es que congela todo el backbone para siempre. Esto funciona bien si nuestro dominio es muy similar al de ImageNet, pero el menaje de cocina tiene particularidades visuales propias: reflejos metálicos en cubiertos, transparencia en vasos, formas muy similares entre clases. Las capas más profundas del backbone (`layer4`) aprenden features de alto nivel muy específicos — contornos complejos, texturas finas — que si se dejan fijos con pesos de ImageNet, pueden no ser los óptimos para nuestras clases.

El fine-tuning progresivo soluciona esto en dos fases:

- **Fase 1 (épocas 1–10):** el backbone sigue congelado y solo se entrena la nueva cabeza clasificadora. El objetivo es estabilizar los pesos de la cabeza antes de tocar nada más. Si se descongelase todo desde el principio, los gradientes de una cabeza inicializada aleatoriamente destruirían el conocimiento previo del backbone.
- **Fase 2 (épocas 11+):** una vez la cabeza está estabilizada, se descongela únicamente `layer4` — la última capa convolucional del backbone — y se entrena con un learning rate extremadamente bajo (`1e-5`). Esto permite ajustar los features de alto nivel al dominio sin olvidar lo aprendido en las capas anteriores.

**CODE:**

- Cargamos ResNet-50 con `ResNet50_Weights.IMAGENET1K_V1`, igual que en el advanced. En la Fase 1 congelamos **todos** los parámetros del modelo con `requires_grad = False`. Así, al construir el optimizer con `filter(lambda p: p.requires_grad, ...)`, solo se incluyen los parámetros entrenables en cada momento.

- La cabeza clasificadora se mejora respecto al `advanced_model`: en lugar de ir directamente de 2048 a 6 clases, añadimos una capa intermedia `Linear(2048, 512)` con `ReLU` y `Dropout(0.4)`. Este bottleneck fuerza al modelo a comprimir la información en una representación más abstracta antes de clasificar, mejorando la generalización con datasets pequeños.

- La función `unfreeze_layer4(model)` se llama desde `train.py` al llegar a la época 10. Itera sobre `model.named_parameters()` y activa `requires_grad = True` únicamente para los parámetros cuyo nombre contiene `"layer4"` o `"fc"`. Tras esto, `train.py` reinicia el optimizer con **learning rates diferenciados por grupo de parámetros**:

```python
optimizer = torch.optim.AdamW([
    {'params': model.layer4.parameters(), 'lr': 1e-5},  # backbone: lr mínimo
    {'params': model.fc.parameters(),     'lr': 1e-4},  # cabeza: lr normal
], weight_decay=1e-2)
```

Usar un lr mucho más bajo en el backbone que en la cabeza es fundamental — si ambos tuviesen el mismo lr, los gradientes de la cabeza destrozarían los pesos preentrenados de `layer4`.

- Como función de pérdida se usa `CrossEntropyLoss` con `label_smoothing=0.1`, que evita que el modelo se vuelva demasiado confiado durante el fine-tuning fino de la Fase 2.

- El scheduler `CosineAnnealingLR` se reinicia al comenzar la Fase 2 con `T_max = num_epochs - FASE2_EPOCH`, de forma que el lr decae suavemente desde el inicio del fine-tuning hasta el final del entrenamiento, independientemente de cuándo se activó la Fase 2.

**Autor:** `\Guillermo Álvarez González\`

---

## Train
Lets do the training. Comenzamos con el entrenamiento. Aquí juntamos todo: dataloader, models, datasets...
- Primero de todo importamos las funciones necesarias para el entrenamiento y los métodos que hemos implementado desde los otros scripts (modelos, loaders).
- Especificamos al usuario por pantalla que introduzca tanto el modelo que quiere usar para entrenar sus fotos, como el dataset que quiere utilizar. 
- Definimos el tamaño del batch especificado por Enrique, y el númeor de épocas.
- Utilizando `os` definiremos un par de directorios en los cuales almacenaremos los resultados y los parámetros.
- Como bien se explica en las especificaciones teóricas el proyecto, el experimento 1 correspondiente al baseline, no será sometido a transformaciones. En cualquier otro caso si que decidimos aplicarlas debido a las ventajas de estas frente al overfitting. 
- tras obtener los conjuntos de entrenamiento y validación definiremos la variable `model` correspondiente al tipo de modelo que deseamos y consecuentemente estableceremos la función de pérdida (La cual será siempre `CrossEntropyLoss()`) y la función de optimización.
- Utilizaremos AdamW para únicamente el transformer, y Adam para el resto de modelos.
  * `Adam` utiliza `weight decay` pero este se calcula dentro del cálculo del momento. Debido a que Adam adapta el ratio de aprendizaje para cada parámetro indicividualmente, el efecto del `weight decay` termina desapareciendo. Esto hace que la regularización no sea tan efectiva.
  * `AdamW` corrige separa la actualización del gradiente de la penalización del`Weight Decay`. Esto es mejor para la rgularización.
- Empezamos el bucle de entrenamiento para cada época y batch.
    1. Con `model.train` preparamos el modelo y parámetros para ser entrenado. 
    2. Para cada tensor de features con sus respectivas labels sacados de get_loaders, primero reestablecemos todos los gradientes a 0, ya que en el caso de los gradientes estos no se sobreescriben en cada iteración sino que se suman, por lo tanto al empoezar cada iteración queremos que valgan 0. Seguido de esto almacenamos los outputs que corresponden a las salidas (predicciones) del modelo pasándole al modelo las imágenes. Posteriormente guardamos el resultado de la función de pérdida pasándole al criterion las salidas frente a las etiquetas reales extraidas del loader. Una vez tenemos la función de pérdida calculada, ejecutamos el backward. (Quería añadir el dato de que en mi modelo -- resnet50 -- ocurre esto tan bonito llamado skip connections). Una vez hemos almacenado todos los gradientes con `backward`, ejecutamos los cambios en los parámetros (pesos y bias) de todas las capas utilizando `optimizer.step()`. Ahora que ya hemos actualizado los parámetros utilizando el optimizador Adam, ahora almacenamos la pérdida de ese batch para luego sumárselo al resto de batches y calcular la media final. 
    3. Ahora que ya tenemos la suma de todas las funciones de pérdida, dividimos este resultado entre el número de batches de entrenamiento. De esta forma añadimos a la lista de errores de cada época el resultado de esta época. 
    4. Empezamos con el conjunto de validación, para poder estudiar el overfitting y poder estudiar como se comporta el modelo para poder editar hiperparámetros y modificaciones para mejorar. Siempre queremos mejorar. Primero de todo hacemos lo mismo. Preparamos el modelo y parámetros para la validación con `mode.eval()`. Haremos exáctamente lo mismo que hicimos para el loader de entrenamiento, sin embargo, en este caso NO haremos la propagación hacia atrás ya que no queremos calcular los gradientes de los parámetros, ya que no queremos aprender ni modificar parámetros con este conjunto. Definiremos with torch.no_grad() para ahorrar memoria. utilizamos .item() para sacar la el resultado de la pérdida ya que loss no es simplemente un float sino un tensor con varias características almacenadas. `_, preds = torch.max(outputs, 1)` con esto obtendremos 2 cosas: el valor máximo (no lo necesitamos), y la posición (índice donde la predicción es la mayor). Como todos los cálculos los hacemos en tensores, para F1-score y otras métricas queremos pasar a numpy, que es lo que utilizan. Con `extend` vamos añadiendo al final de la lista los indices de las predicciones de cada batch sin generar lista de listas.
    5. Por último almacenamos la métrica que pide Enrique F1-score, y guardamos gráficas y resultados.
    6. Hemos implementado Early stopping para cuando el modelo sigue mejorando en train, pero deja de mejorar en val para de este modo, evitar el overfitting cuando se está aprendiendo del ruido y se está memorizando el conjunto de entrenamiento.

**Autor:** `\Guillermo Álvarez González\`



## Evaluate
Este archivo centraliza la validación final de cada uno de los modelos sobre el conjunto de los datos que nunca ha visto, es decir, el test.
Se divide en 7 fases claras:
1. Menú de configuración: está totalmente desacoplado del de entrenamiento. 
Este se inicia y solicita al usuario la arquitectura (model_type) y el dataset (dataset_name) que quiere evaluar. Así logramos poder testear la combinación que el usuario quiera de forma sencilla y rápida.
2. Carga del test DataLoader: invocamos la función get_loaders apuntando unicamente al conjunto de test. Es importante forzar transformation=False, los datos no deben sufrir data augmentation ya que lo que buscamos es medir el rendimiento que tiene el modelo sobre imágenes sin modificar, imágenes puras.
3. Instanciar el modelo exacto que el usuario ha elegido: mediante el patron de diseño Factory (usando build_transfer_model) o bien llamando de forma directa a BaselineCNN/ImprovedCNN, el script reconstruye matemáticamente la arquitectura que el usuario ha seleccionado previamente.
4. Cargar los pesos entrenados: aquí el modelo lo que hace es buscar directamente en el directorio saved_models/ el archivo con extensión .pth correspondiente, el cual se ha guardado gracias al early stopping usado durante el entrenamiento. 
Además, se implementa try-except que lo que hace es capturar errores de FileNotFoundError y que si el usuario está intentando evaluar un modelo que aún no ha sido entrenado evita su bloqueo.
5. Bucle de inferencia con medición de tiempos: a la medición de tiempos y recursos también se le llama profiling. Esta evaluación se ejecuta de forma estricta bajo with torch.no_grad():, lo cual deshabilita el motor Autograd, con esto lo que conseguimos es que reducimos mucho el consumo de memoria VRAM y aceleramos de forma drástica el proceso. 
A su vez aplicamos técnica de profiling usando la librería time para así cronometrar el bucle entero y sacar el tiempo medio de inferencia por batch en milisegundos.
6. Cálculo de métricas y guardado de reporte: aquí lo que hacemos es que usando sklearn.metrics extraemos las métricas globales. También imprimimos por pantalla distintas métricas: accuracy y el classification report (precision, recall y f1-score por clase). Guarda automáticamente un txt en la carpeta results, lo cual asegura que los datos con los que experimentamos el modelo no se pierdan una vez cerremos sesión y sean inmutables.
7. Matriz de confusión visual: esta es la parte final del evaluate.py, aquí se computa y renderiza la matriz de confusión multiclase, exportandose como un .png en la carpeta de los resultados.
Esta es la herramienta principal que vamos a necesitar para cuando hagamos posteriormente nuestro análisis crítico, ya que permite ver los sesgos de la red y detectar qué clases en concreto tienden a confundirse entre si por similitudes de iluminación, lo cual son brillos, sombras o reflejos; o morfológicas, lo cual es la forma física y geométrica de las cosas. Ambas bastante comunes en el menaje.

**Autor:** `\Álvaro Vigueras Capablo\`

---

## Resultados
0. IMPORTANTE: los resultados F1-Score son los resultantes de `train.py` (basándose en el conjunto de validación). Aún así, también hemos incluido los resultados de accuracy y F1-score del conjunto de test (evaluate.py) en los .txt finales.
Hemos utilizado una `paciencia` de `12` para darle tiempo al modelo.


1. Primero de todo hemos empezado probando para el dataset `G00_dataset_split_Reference` que ha proporcionado Enrique para comprobar si llegamos a ese Macro F1-Score de 0.40.
Al ver que en un primer momento nos ha dado poco más de 0.20, nos damos cuenta de que el modelo no estaba aprendiendo bien.
Al haber seleccionado un dopout2D (el que actúa sobre canales antes de llegar al flatten()) y estar utilizando un dataset de solo unas 600 imágenes, nos damos cuenta de que el modelo no estaba aprendiendo lo suficiente.
Por lo tanto, hemos eliminado por completo este `drop2D` y simplemente hemos mantenido el dropout con el `dropout_rate` que ya teníamos (el cual aplica sobre el tensor resultante del flatten()).
Al probart sin este `drop2D` el F-1 score ha sido de 0.37. Finalmente, hemos decidido disminuir el `dropout_rate` de 0.5 a 0.4. Gracias a esto hemos obtenido un macro F1-score de 0.4247

### Baseline
2. Modelo: Baseline, Dataset: Group_Dataset (tiempo de ejecución(colab) = 3:40 min)
 - En la Época 1 (Época 0 en las gráficas), el modelo parte de una pérdida altísima (Train Loss: 14.68) y una precisión muy baja. Sin embargo, en la Época 2, vemos una caída drástica del error. Esto nos indica que el optimizador está funcionando y el modelo está encontrando rápidamente patrones básicos en las imágenes.
 - A partir de la Época 3, si miras la primera gráfica (Curvas de Pérdida), la línea azul (Train Loss) baja de forma suave y constante. Sin embargo, la línea roja (Val Loss) pega unos saltos enormes (ej. de 1.43 a 2.11 entre la época 11 y 12).
-> Esto se ve reflejado en el F1-Score (gráfica verde), pasando de picos altos a valles profundos.
el modelo es demasiado sencillo y los datos de validación (fotos de utensilios de cocina) son complejos (por los reflejos, los fondos, etc.)
 - En las últimas épocas (de la 15 a la 20), la línea azul de Train (tanto en Loss como en Accuracy) sigue mejorando (llega al 51% de Accuracy). El modelo se está memorizando los platos y cucharas que ve.
 - La línea roja de Validation Loss empieza a separarse y a subir ligeramente (fíjate en el pico final de 2.26 en la última época).
Conclusión: El modelo ha dejado de aprender concpetos generales y ha empezado a memorizar los detalles exactos de las fotos de entrenamiento. Cuanto le pasemos fotos nuevas, falla.
Descripción Matriz de confusión
 - Clases fuertes:
    Vasos (Glasses): Es la clase estrella. De 15 fotos de vasos, ha acertado 13 (un 87% de recall). Esto tiene sentido lógico: los vasos suelen ser transparentes y cilíndricos, una característica visual muy distinta al resto de la cubertería.
    Tazas (Cups): También se defienden bien, con 11 aciertos de 17. Probablemente, la red neuronal esté detectando el "asa" de la taza como característica principal.
 - Problema principal:
    Knives (Cuchillos): Esta es la clase que rompe el modelo. La columna de "Knives" predichos tiene números muy altos en filas que no le corresponden.
    Se ha predicho que había un cuchillo 6 veces cuando en realidad era una cuchara (Spoons), 5 veces cuando era un plato (Plates) y 4 veces cuando era un tenedor (Forks)
    En las características visuales de vuestras fotos. Seguramente los cuchillos, tenedores y cucharas compartan el mismo material (metal brillante), tengan un mango similar y estén fotografiados sobre fondos parecidos. Al ser un modelo "Baseline" muy simple, no tiene la profundidad suficiente para distinguir los dientes de un tenedor o la curva de una cuchara, y ante la duda, lo clasifica todo como "cuchillo" (el objeto alargado metálico genérico).
 - Problema secundario:
    De las 17 cucharas que había, la red solo ha acertado 3. Ha clasificado 3 de ellas como tenedores, y 4 como vasos (posiblemente por los reflejos de la luz en el metal que se parecen a los reflejos del cristal).
 - 
3. Modelo: baseline, Dataset: Kitchenware_EVEN_pooled_Homogeneous (tiempo de ejecución(colab) = 9:35 min)
Aquí vemos como afecta el fondo de las imágenes al rendimiento de una IA, este dataset tiene fondos homogéneos y se ha notado muchísimo.
- Curvas de aprendizaje
    Si miramos la evolución, la pérdida de validación(Val Loss) baja de una forma suave y se estabiliza en 1.13
    Importante. El error de validación es menor que el del entrenamienot(1.13vs1.20) Esto lo vemos en: primero , porque al tener los fondos lisos, las imagénes de validación son muy limpias y fáciles de leer para la red. Segundo, porque durante el entrenamiento el modelo sufre los cortes del Dropout(apagando neuronas para aprender mejor), pero en Validación el Dropout se desactiva, rindiendo al 100%. No hay Overfitting aquí.
- Puntos que hace bien el modelo:
    Mirando el reporte de clasificación: La clase de Platos tiene una precisión de 1.00, esto significa que el modelo ha tenido un 100% de las veces que el modelo ha dicho que esto es un plato, ha acertado.
    Cups y glasses también han tenido un rendimiento muy bueno, con un F1-Score de 0.72 y 0.65 respectivamente. Al no haber fondos que distraigan, la red detecta a la perfección los contornos redondos y los reflejos del cristal.
- Fallos más repetitivos:
    Los cuchillos: A pesar de tener fondos limpios, el modelo sigue fallando al intentar separar los cubiertos
    La precisión es de tan solo un 0.21, la red neuronal cuando ve un trozo de metal alargado, ante la duda, siempre apuesta por 'Cuchillo'. Sigue clasificando un montón de tenedores y cucharas erróneamente como cuchillos.
Conclusión: Los fondos lisos ayudan a aislar al objeto(mejorando el F1-Score del 45.77% a 51.87%) pero la arquitectura BaselineCNN es demasiado poco profunda(solo tiene 3 capas convolucionales). No tiene la capacidad matemática para extraer características 'finas' como loos dientes de un tenedor o la concavidad de una cuchara
    
4. Modelo: Baseline, Dataset: Kitchenware_Combinado_Splits (tiempo de ejecución(colab) = 18:23 min)

5. Modelo: Baseline, Dataset: public_kaggle:
- Mejor macro F1-Score = 0.3584
- Tiempo de ejecución(colab) = 15:37 min

6. Modelo: Baseline, Dataset: odd_pooled:
- Mejor macro F1-Score = 0.1926
  - Tiempo de ejecución(colab) = 10 min


### Improved Guillermo Álvarez
1. Modelo: improved_guille, Dataset: group_dataset:
- Mejor Macro F1:   0.6368
- Tiempo de ejecución(cuda) = 2m 55s
Se han completado las 50 épocas.

2. Modelo: improved_guille, Dataset: even_pooled:
- Mejor Macro F1: 0.5886
- Tiempo de ejecución(colab) = 18m 7s
Se ha activado el `early stopping`. El modelo no mejora en 12 épocas consecutivas.

3. Modelo: improved_guille, Dataset: odd_pooled:
- mejor Macro F1: 0.3992
- Tiempo de ejecución(cuda) = 16m 45s
Se han completado las 50 épocas.

4. Modelo: improved_guille, Dataset: public_kaggle:
- mejor Macro F1: 0.4932
- Tiempo de ejecución(cuda) = 35m 29s
Se han ejecutado las 50 épocas.

5. Modelo: improved_guille, Dataset: combined:
- mejor Macro F1: 0.4760
- Tiempo de ejecución(cuda) = 45m 8s

### Improved Samuel Llamas

1. Modelo: improved_samuel, Dataset: group_dataset:
- mejor Macro F1: 0.6302
- Tiempo de ejecución(cuda) = 17m 58s
Se han completado las 50 épocas

2. Modelo: improved_samuel, Dataset: odd_pooled:
- mejor Macro F1: 0.6527
- Tiempo de ejecución(cuda) = 21m 36s
Se han completado las 50 épocas

3. Modelo: improved_samuel, Dataset: even_pooled:
- mejor Macro F1: 0.7784
- Tiempo de ejecución(cuda) = 347m 28s

4. Modelo: improved_samuel, Dataset: public_kaggle:
- mejor Macro F1: 0.7421
- Tiempo de ejecución(cuda) = 249m 53s
    
5. Modelo: improved_samuel, Dataset: combined:
- mejor Macro F1: 0.6606
- Tiempo de ejecución(cuda) = 74m 28s
Se han completado las 50 épocas.

### Improved Álvaro Vigueras
1. group dataset
- Mejor Macro F1: 0.5504
- Tiempo de ejecución(colab) = 6min 20s
Se han completado las 50 épocas.
- 
2. EVEN_pooled 
- Mejor Macro F1: 0.5983
- Tiempo de ejecución(colab) = 32min 23s
Se han completado las 50 épocas.
- 
3. ODD_pooled 
- Mejor Macro F1: 0.4128
- Tiempo de ejecución(colab) = 27min 45s
Se han completado las 50 épocas.
-
4. Combined
- Mejor Macro F1:  0.4701
- Tiempo de ejecución(CPU) = 336min 53s
Se han completado las 50 épocas.
-
5. Kaggle
- Mejor Macro F1: 0.4684
- Tiempo de ejecución(CPU) = 464min 42s
Se han completado las 50 épocas.


### ResNet50
1. Modelo: resnet50, Dataset: group_dataset:
- mejor Macro F1: 0.9080
- Tiempo de ejecución(cuda) = 23m 54s

2. Modelo: resnet50, Dataset: odd_pooled:
- mejor Macro F1: 0.8875
- Tiempo de ejecución(cpu) = 125m 12s
Early stopping activado.

3. Modelo: resnet50, Dataset: even_pooled:
- mejor Macro F1: 0.9646
- Tiempo de ejecución(CPU) = 332m 43s

4. Modelo: resnet50, Dataset: public_kaggle:
- mejor Macro F1: 0.9189
- Tiempo de ejecución(cuda) = 15m 25s
Early stopping activado.

5. Modelo: resnet50, Dataset: combined:
- mejor Macro F1: 0.9240
- Tiempo de ejecución(cuda) = 24m 48
Early Stopping activado. No hubo mejoras en 12 epocas consecutivas.

### EfficientNet-V2
1. Modelo: efficientnet_v2_s, Dataset: group_dataset:
- mejor Macro F1: 0.8533
- Tiempo de ejecución(cuda) = 4m 2s
Early Stopping activado. No hubo mejoras en 12 epocas consecutivas.

2. Modelo: efficientnet_v2_s, Dataset: odd_pooled:
- mejor Macro F1: 0.8959
- Tiempo de ejecución(cuda) = 16m 14s
Early Stopping activado. No hubo mejoras en 12 epocas consecutivas.

3. Modelo: efficientnet_v2_s, Dataset: even_pooled:
- mejor Macro F1: 0.9437
- Tiempo de ejecución(cpu) = 469m 52s
Early Stopping activado. No hubo mejoras en 12 epocas consecutivas.

4. Modelo: efficientnet_v2_s, Dataset: public_kaggle:
- mejor Macro F1: 0.9401
- Tiempo de ejecución(cuda) = 23m 45s
Early Stopping activado. No hubo mejoras en 12 epocas consecutivas.

5. Modelo: efficientnet_v2_s, Dataset: combined:
- mejor Macro F1: 0.9151
- Tiempo de ejecución(cuda) = 194m 37s
Early Stopping activado. No hubo mejoras en 12 epocas consecutivas.

### Transformer VitB16
1. Modelo: vit_b_16, Dataset: group_dataset:
- mejor Macro F1: 0.9395
- Tiempo de ejecución(cuda) = 41m 36s
Early stopping activado en la época 29. No hubo mejoras en 12 épocas consecutivas

2. Modelo: vit_b_16, Dataset: odd_pooled:
- mejor Macro F1:
- Tiempo de ejecución(cuda) =

3. Modelo: vit_b_16, Dataset: even_pooled:
- mejor Macro F1:
- Tiempo de ejecución(cuda) =

4. Modelo: vit_b_16, Dataset: public_kaggle:
- mejor Macro F1: 0.9415
- Tiempo de ejecución(cuda) = 74m 38s
Se han completado las 50 épocas.

5. Modelo: vit_b_16, Dataset: combined:
- mejor Macro F1: 0.9357
  - Tiempo de ejecución(cuda) = 1496m 27s

### Free Model
* Modelo: free_model
* Dataset: public_kaggle
- Mejor Macro F1-Score: 0.9687
- Tiempo de ejecución (cuda): 45m 43s