# LR_1_task_1.py
# Попередня обробка даних: бінаризація, виключення середнього, масштабування, нормалізація, кодування міток

import numpy as np
from sklearn import preprocessing

# 2.1 – Вхідні дані

input_data = np.array([
    [5.1, -2.9, 3.3],
    [-1.2, 7.8, -6.1],
    [3.9, 0.4, 2.1],
    [7.3, -9.9, -4.5]
])

print("Input data:")
print(input_data)

# 2.1.1 Бінаризція

data_binarized = preprocessing.Binarizer(threshold=2.1).transform(input_data)
print("\nBinarized data:\n", data_binarized)

# 2.1.2 Викл. середнього

print("\nBEFORE:")
print("Mean =", input_data.mean(axis=0))
print("Std deviation =", input_data.std(axis=0))

data_scaled = preprocessing.scale(input_data)

print("\nAFTER:")
print("Mean =", data_scaled.mean(axis=0))
print("Std deviation =", data_scaled.std(axis=0))

# 2.1.3 Мастабуання(Min–Max)

data_scaler_minmax = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_scaled_minmax = data_scaler_minmax.fit_transform(input_data)

print("\nMin–Max scaled data:\n", data_scaled_minmax)

# 2.1.4 Нормалізація(L1 та L2)

data_normalized_l1 = preprocessing.normalize(input_data, norm='l1')
data_normalized_l2 = preprocessing.normalize(input_data, norm='l2')

print("\nL1 normalized data:\n", data_normalized_l1)
print("\nL2 normalized data:\n", data_normalized_l2)

# 2.1.5 Кодуання міток

input_labels = ['red', 'black', 'red', 'green', 'black', 'yellow', 'white']

encoder = preprocessing.LabelEncoder()
encoder.fit(input_labels)

print("\nLabel mapping:")
for i, item in enumerate(encoder.classes_):
    print(item, "-->", i)

# Перетворення тестових міток
test_labels = ['green', 'red', 'black']
encoded_values = encoder.transform(test_labels)

print("\nTest labels =", test_labels)
print("Encoded values =", list(encoded_values))

# Декодування
encoded_values2 = [3, 0, 4, 1]
decoded_list = encoder.inverse_transform(encoded_values2)

print("\nEncoded values =", encoded_values2)
print("Decoded labels =", list(decoded_list))
