#######################################
##### В связи с тем, что после устранения ряда ошибок RStudio стала вылетать, попробовал написать код на Python
##### С помощью библиотек numpy и pandas в JupyterLab
##### Final_Homework
##### USA_SP500 - S&P 500 Index
#######################################

# Импорт библиотек
import numpy as np
import pandas as pd

# Установка начального числа для генератора псевдослучайных чисел
np.random.seed(100)

# Загрузка исходные данные
SP_500 = pd.read_excel('SP500.xlsx')

################SM-модель################
# Импорт необходимых модулей
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

# Подготовка данных - Деление выборок на тренировочную и тестовую
train_price = SP_500[0:1000]
test_price =  SP_500[1000:1201]
train_x =  (SP_500.ClosingPrice[0:1000], SP_500.Open[0:1000], SP_500.DailyHigh[0:1000], SP_500.DailyLow[0:1000])
test_x =  (SP_500.ClosingPrice[1000:1201], SP_500.Open[1000:1201], SP_500.DailyHigh[1000:1201], SP_500.DailyLow[1000:1201])

# Изменение типа - Создание массивов данных из DataFrame и кортежей для дальнейшей работы
train_price = np.asarray(train_price)
test_price = np.asarray(test_price)
train_x = np.asarray(train_x)
test_x = np.asarray(test_x)

# Изменение типа - Изменение типа данных столбцов массивов данных на число с плав. точкой для последующей нормализации
train_price = train_price.astype('float32')
test_price = test_price.astype('float32')

# Нормализация данных методом Мин-Макс
train_price = (train_price - np.min("ClosingPrice")) / (np.max("ClosingPrice") - np.min("ClosingPrice"))
test_price = (test_price - np.min("ClosingPrice")) / (np.max("ClosingPrice") - np.min("ClosingPrice"))

# Предварительная обработка меток классов для Keras
train_x = np_utils.to_categorical(train_x, 10)
test_x = np_utils.to_categorical(test_x, 10)

# Задача архитектуры модели
sm = Sequential()
# Добавление слоёв
sm.add(Conv2D(64, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
sm.add(MaxPooling2D(pool_size=(2, 2)))
sm.add(Dropout(0.2))
sm.add(Conv2D(100, (3, 3), activation='relu'))
sm.add(MaxPooling2D(pool_size=(2, 2)))
sm.add(Dropout(0.2))
sm.add(Flatten())
sm.add(Dense(500, activation='relu'))
sm.add(Dropout(0.5))
sm.add(Dense(10, activation='softmax'))

# Компиляция модели (SM, mse, rmsprop) и вывод итога
sm.compile(loss="mse", optimizer="rmsprop", metrics=["accuracy"])
print(sm.summary())

# Обучение модели
sm.fit(train_price, train_x, batch_size=125, epochs=5, validation_split=0.2, verbose=2)

# Оценка качества работы модели на тестовых данных
scores = sm.evaluate(test_price, test_x, verbose=0)
print("Точность работы на тестовых данных: %.2f%%" % (scores[1]*100))

# Компиляция модели (SM, mse, adam) и вывод итога
sm.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
print(sm.summary())

# Обучение модели
sm.fit(train_price, train_x, batch_size=125, epochs=5, validation_split=0.2, verbose=2)

# Оценка качества работы модели на тестовых данных
scores = sm.evaluate(test_price, test_x, verbose=0)
print("Точность работы на тестовых данных: %.2f%%" % (scores[1]*100))

# Компиляция модели (SM, mape, rmsprop) и вывод итога
sm.compile(loss="mape", optimizer="rmsprop", metrics=["accuracy"])
print(sm.summary())

# Обучение модели
sm.fit(train_price, train_x, batch_size=125, epochs=5, validation_split=0.2, verbose=2)

# Оценка качества работы модели на тестовых данных
scores = sm.evaluate(test_price, test_x, verbose=0)
print("Точность работы на тестовых данных: %.2f%%" % (scores[1]*100))

# Компиляция модели (SM, mape, adam) и вывод итога
sm.compile(loss="mape", optimizer="adam", metrics=["accuracy"])
print(sm.summary())

# Обучение модели
sm.fit(train_price, train_x, batch_size=125, epochs=5, validation_split=0.2, verbose=2)

# Оценка качества работы модели на тестовых данных
scores = sm.evaluate(test_price, test_x, verbose=0)
print("Точность работы на тестовых данных: %.2f%%" % (scores[1]*100))
