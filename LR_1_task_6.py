# LR_1_task_6.py

import numpy as np
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
from utilities import visualize_classifier

# Завантаження даних
input_file = 'data_multivar_nb.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# Розбиття на тренувальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Naive Bayes ---
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)
y_pred_nb = nb_classifier.predict(X_test)
accuracy_nb = 100.0 * np.mean(y_pred_nb == y_test)
print("Naive Bayes Accuracy:", round(accuracy_nb, 2), "%")
visualize_classifier(nb_classifier, X_test, y_test)

# --- SVM ---
svm_classifier = SVC(kernel='rbf', probability=True)
svm_classifier.fit(X_train, y_train)
y_pred_svm = svm_classifier.predict(X_test)
accuracy_svm = 100.0 * np.mean(y_pred_svm == y_test)
print("SVM Accuracy:", round(accuracy_svm, 2), "%")
visualize_classifier(svm_classifier, X_test, y_test)

# --- Порівняння моделей ---
if accuracy_svm > accuracy_nb:
    print("SVM краща модель за точністю")
else:
    print("Naive Bayes краща модель за точністю")
