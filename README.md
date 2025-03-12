## Функції, які виконує програма

1. **Підготовка даних**:
   - Розділення даних на множини `train`, `val` та `test`.
   - Завантаження зображень з вказаних директорій та їх обробка за допомогою трансформацій для тренувальної та тестової множин.
   - Аугментація даних для тренувальної множини, що включає випадкове обертання, горизонтальний переворот та зміни яскравості/контрасту.

2. **Трансформації зображень**:
   - Використовуються стандартні трансформації для обробки зображень:
     - `RandomHorizontalFlip()` — випадковий горизонтальний переворот.
     - `RandomRotation(15)` — випадкове обертання зображень на 15 градусів.
     - `ColorJitter(brightness=0.2, contrast=0.2)` — випадкові зміни яскравості та контрасту.
     - `Resize((128, 128))` — зміна розміру зображень до 128x128 пікселів.
     - `Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])` — нормалізація зображень з використанням середніх значень та стандартних відхилень.

3. **Завантаження даних за допомогою `DataLoader`**:
   - Створення об'єктів `DataLoader` для тренувальної, валідаційної та тестової множин з розміром батчу 32 .
   
4. **Денормалізація зображень**:
   - Функція `denormalize()` використовується для відновлення зображень до їх початкового вигляду після нормалізації.

5. **Візуалізація зображень**:
   - Функція `show_images()` використовується для відображення декількох прикладів зображень з тренувального набору разом з їх класами.
   
6. **Аналіз розподілу зображень по класах**:
   - Підрахунок кількості зображень у кожному класі для тренувальних, валідаційних та тестових множин.
   - Побудова гістограми для візуалізації кількості зображень у кожному класі.

7. **Підрахунок загальної кількості зображень**:
   - Підрахунок загальної кількості зображень у кожній з множин та виведення відсоткового співвідношення для тренувальних, валідаційних та тестових даних.

---

## Інструкція по налаштуванню середовища

1. **Потрібні бібліотеки**:
   Для роботи програми необхідно мати встановлені наступні бібліотеки:
   - `torch` — для роботи з PyTorch.
   - `torchvision` — для роботи з трансформаціями зображень.
   - `matplotlib` — для візуалізації зображень та графіків.
   - `numpy` — для роботи з масивами та числовими операціями.
   - `seaborn` (не обов'язково, можна замінити на matplotlib) — для побудови графіків.

   Встановити всі бібліотеки можна за допомогою pip:
   ```bash
   pip install torch torchvision matplotlib numpy 
   
##  Структура даних 

Animal-10-split/
├── train/
│   ├── class1/
│   ├── class2/
│   └── ...
├── val/
│   ├── class1/
│   ├── class2/
│   └── ...
└── test/
    ├── class1/
    ├── class2/
    └── ...

