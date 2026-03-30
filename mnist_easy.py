import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

# 1. ЗАГРУЗКА ДАННЫХ
print("Загрузка данных MNIST...")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 2. ПРЕДОБРАБОТКА
# Нормализация (0-255 -> 0-1)
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Преобразование из 28x28 в 784 пикселя
x_train = x_train.reshape(x_train.shape[0], 784)
x_test = x_test.reshape(x_test.shape[0], 784)

# Преобразование меток в one-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

print(f"Обучающая выборка: {x_train.shape}")
print(f"Тестовая выборка: {x_test.shape}")

# 3. СОЗДАНИЕ МОДЕЛИ
model = keras.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# 4. КОМПИЛЯЦИЯ
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Вывод архитектуры
print("\nАрхитектура модели:")
model.summary()

# 5. ОБУЧЕНИЕ
print("\nНачало обучения...")
history = model.fit(
    x_train, y_train,
    batch_size=32,
    epochs=5,
    validation_split=0.2,
    verbose=1
)

# 6. ОЦЕНКА
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"\nТочность на тестовых данных: {test_accuracy:.4f}")
print(f"Потери на тестовых данных: {test_loss:.4f}")

# 7. ПРИМЕР ПРЕДСКАЗАНИЯ
# Берем первое изображение из тестовой выборки
sample = x_test[0:1]  # форма (1, 784)
prediction = model.predict(sample, verbose=0)
predicted_digit = np.argmax(prediction)
true_digit = np.argmax(y_test[0])

print(f"\nПример предсказания:")
print(f"Предсказанная цифра: {predicted_digit}")
print(f"Истинная цифра: {true_digit}")
