# Класифікація зображень: CNN та Transfer Learning

Проєкт досліджує класифікацію зображень на PyTorch: власні згорткові архітектури, порівняння Global Average Pooling з Flatten і transfer learning на попередньо навчених моделях.

## Датасет

Набір містить **3659** зображень у **150** класах (кожен клас — окрема підпапка). Структура каталогу:

```
dataset/
├── cofagrigus/
├── combee/
├── combusken/
└── ...
```

Розподіл на train/validation: **80/20** (фіксований seed `42`).

| Параметр        | Значення |
|-----------------|----------|
| Класів          | 150      |
| Навчальна вибірка | 2928   |
| Валідаційна вибірка | 731  |

## Що реалізовано

- Завантаження даних через `ImageFolder` та `DataLoader`
- Власна CNN з трьома згортковими блоками (Conv → BatchNorm → ReLU → MaxPool)
- Дві варіації класифікатора:
  - **CNNFlatten** — класичний `Flatten` перед повнозв'язним шаром
  - **CNNGAP** — `Global Average Pooling` перед класифікатором
- Навчання з `Adam` та `StepLR` scheduler
- Transfer learning на **ResNet18** та **MobileNetV2**:
  - **Feature Extraction** — навчається лише останній шар
  - **Fine-tuning** — розморожуються останні згорткові блоки
- Порівняння експериментів: accuracy, precision, recall, F1, час навчання, криві loss/accuracy, confusion matrix

## Вимоги

- Python 3.10+
- PyTorch
- torchvision
- numpy, pandas, matplotlib, scikit-learn
- Jupyter Notebook або JupyterLab

```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn jupyter
```

## Запуск

1. Клонуйте або розпакуйте проєкт так, щоб поруч із ноутбуком була папка `dataset/`.
2. Відкрийте `notebook.ipynb` і виконуйте клітинки зверху донизу.
3. Запускайте ноутбук з кореня проєкту — шлях до даних задається як `Path("dataset")`.

Якщо ноутбук відкрито з іншої директорії, змініть змінну `DATA_DIR` на повний шлях до `dataset`.

При першому запуску transfer learning PyTorch автоматично завантажить ваги ResNet18 та MobileNetV2 (ImageNet).

## Порівнювані моделі

| Модель | Розмір вхідного зображення | Опис |
|--------|----------------------------|------|
| CNNFlatten | 32×32 | Власна CNN з Flatten |
| CNNGAP | 32×32 | Власна CNN з Global Average Pooling |
| ResNet18 FE | 224×224 | Feature extraction |
| ResNet18 FT | 224×224 | Fine-tuning layer4 + fc |
| MobileNetV2 FE | 224×224 | Feature extraction |
| MobileNetV2 FT | 224×224 | Fine-tuning останнього блоку + classifier |

## Орієнтовні результати (val accuracy)

| Експеримент | Val accuracy |
|-------------|--------------|
| CNNFlatten | ~90.7% |
| CNNGAP | ~50.2% |
| ResNet18 FE | ~88.0% |
| ResNet18 FT | ~95.2% |
| MobileNetV2 FE | ~86.5% |
| MobileNetV2 FT | ~69.4% |

ResNet18 з fine-tuning показує найкращу точність на цьому наборі. CNNFlatten перевершує CNNGAP за accuracy, хоча GAP має суттво менше параметрів (~113K проти ~401K).

## Структура проєкту

```
.
├── dataset/          # зображення за класами
├── notebook.ipynb    # код, навчання та візуалізації
└── README.md
```

## Примітки

- Для CNN-архітектур зображення масштабуються до 32×32; для transfer learning — до 224×224 з нормалізацією ImageNet.
- Навчання використовує GPU, якщо доступна (`cuda`), інакше — CPU.
- Параметри навчання: 10 епох, batch size 32, learning rate `1e-3` (для fine-tuning — `1e-4`).
