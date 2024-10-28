import os
import keras
from keras.src.datasets import mnist  # Импортируем набор данных MNIST
from keras.src.layers import Flatten, Dense  # Импортируем слои Flatten и Dense для создания модели

# Отключаем вывод предупреждений и информационных сообщений от TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np  # Импортируем библиотеку для работы с массивами
import matplotlib.pyplot as plt  # Импортируем библиотеку для визуализации данных

# Загружаем данные MNIST (рукописные цифры)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Стандартизация входных данных: приводим значения пикселей к диапазону [0, 1]
x_train = x_train / 255  # Обучающая выборка
x_test = x_test / 255  # Тестовая выборка

# Преобразуем метки классов в категории (векторное кодирование), так как у нас 10 классов (цифры от 0 до 9)
y_train_cat = keras.utils.to_categorical(y_train, 10)  # Обучающие метки
y_test_cat = keras.utils.to_categorical(y_test, 10)  # Тестовые метки

# Отображение первых 25 изображений из обучающей выборки
plt.figure(figsize=(5, 5))  # Размер рисунка
for i in range(25):  # Цикл для 25 изображений
    plt.subplot(5, 5, i + 1)  # Размещение изображений в сетке 5x5
    plt.xticks([])  # Убираем отметки по оси X
    plt.yticks([])  # Убираем отметки по оси Y
    plt.imshow(x_train[i], cmap=plt.cm.binary)  # Отображаем изображение в оттенках серого
plt.show()  # Показываем график

# Создаем модель
model = keras.Sequential([
    Flatten(input_shape=(28, 28, 1)),  # Преобразует 28x28 изображение в одномерный вектор
    Dense(128, activation='relu'),  # Полносвязный слой с 128 нейронами и функцией активации ReLU
    Dense(10, activation='softmax')  # Выходной слой с 10 нейронами и функцией активации softmax (по количеству классов)
])

# Вывод структуры модели в консоль
print(model.summary())

# Компилируем модель, задаем параметры обучения
model.compile(optimizer='adam',  # Оптимизатор Adam для обновления весов
              loss='categorical_crossentropy',  # Функция потерь для многоклассовой классификации
              metrics=['accuracy'])  # Метрика — точность предсказания

# Обучаем модель на обучающей выборке
model.fit(x_train, y_train_cat, batch_size=32, epochs=5, validation_split=0.2)  # Разделяем на обучающую и проверочную выборку

# Оцениваем точность модели на тестовой выборке
model.evaluate(x_test, y_test_cat)

# Пример распознавания одного изображения
n = 1  # Индекс изображения из тестовой выборки
x = np.expand_dims(x_test[n], axis=0)  # Расширяем измерение, чтобы подать изображение в модель
res = model.predict(x)  # Предсказываем цифру
print(res)  # Выводим вероятности для всех классов (0-9)
print(np.argmax(res))  # Находим индекс с максимальной вероятностью — это предсказанное число

# Отображаем изображение, для которого сделали предсказание
plt.imshow(x_test[n], cmap=plt.cm.binary)
plt.show()

# Распознавание всех изображений тестовой выборки
pred = model.predict(x_test)  # Предсказываем классы для всех изображений
pred = np.argmax(pred, axis=1)  # Преобразуем вероятности в метки классов

print(pred.shape)  # Размер предсказанного массива

# Вывод первых 20 предсказанных значений и реальных меток для сравнения
print(pred[:20])  # Первые 20 предсказанных значений
print(y_test[:20])  # Первые 20 реальных значений

# Выделение неверных предсказаний
mask = pred == y_test  # Создаем маску, где True — правильные предсказания
print(mask[:10])  # Показываем первые 10 элементов маски

x_false = x_test[~mask]  # Извлекаем изображения с неправильными предсказаниями
y_false = y_test[~mask]  # Извлекаем реальные метки для неправильных предсказаний

print(x_false.shape)  # Размер массива неверно предсказанных изображений

# Отображаем первые 25 неверных предсказаний
plt.figure(figsize=(10, 5))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_false[i], cmap=plt.cm.binary)  # Отображаем изображение в оттенках серого
plt.show()  # Показываем график
