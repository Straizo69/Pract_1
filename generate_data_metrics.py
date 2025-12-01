# generate_data_metrics.py

import pandas as pd
import numpy as np

# Встановимо випадковий seed для відтворюваності
np.random.seed(42)

# Кількість зразків
n_samples = 1000

# Генеруємо фактичні мітки (0 або 1)
actual_label = np.random.binomial(1, 0.5, n_samples)

# Генеруємо ймовірності для моделі RF
# Якщо фактична мітка 1, ймовірність більш висока
model_RF = np.clip(np.random.normal(0.7, 0.2, n_samples) * actual_label + 
                   np.random.normal(0.3, 0.2, n_samples) * (1 - actual_label), 0, 1)

# Генеруємо ймовірності для моделі LR
model_LR = np.clip(np.random.normal(0.65, 0.25, n_samples) * actual_label + 
                   np.random.normal(0.35, 0.25, n_samples) * (1 - actual_label), 0, 1)

# Створюємо DataFrame
df = pd.DataFrame({
    'actual_label': actual_label,
    'model_RF': model_RF,
    'model_LR': model_LR
})

# Зберігаємо у CSV
df.to_csv('data_metrics.csv', index=False)

print("Файл 'data_metrics.csv' створено успішно!")
