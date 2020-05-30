############################
##### Final homework
##### USA - S&P 500 (^GSPC)
############################

# Установка необходимых пакетов
install.packages('BatchGetSymbols')
install.packages('plotly')
install.packages('keras')
install.packages('tensorflow')
install.packages('minimax')

# Обращение к установленным пакетам
library('BatchGetSymbols')
library('plotly')
library('keras')
library('tensorflow')
library('minimax')

# Задание values - Индекс S&P 500 (^GSPC) в период 25.06.15 - 29.05.20  
tickers <- c('%%5EGSPC')
first.date <- Sys.Date() - 360*5
last.date <- Sys.Date()

# Загрузка данных (Got 100% of valid prices | Nice!)
yts <- BatchGetSymbols(tickers = tickers,
                       first.date = first.date,
                       last.date = last.date,
                       cache.folder = file.path(tempdir(),
                                                'BGS_Cache') )

# Этап подготовки данных
y <-  yts$df.tickers$price.close
myts <-  data.frame(index = yts$df.tickers$ref.date, price = y, vol = yts$df.tickers$volume)
myts <-  myts[complete.cases(myts), ]
myts <-  myts[-seq(nrow(myts) - 1200), ]
myts$index <-  seq(nrow(myts))

# Вывод на экран графика - Цена закрытия индекса S&P 500 (^GSPC) за рассматриваемый период
plot_ly(myts, x = ~index, y = ~price, type = "scatter", mode = "markers", color = ~vol)

# Стандартизация данных методом min-max
myts <- data.frame(index = rminimax(myts$index), price = rminimax(myts$price), vol= rminimax(myts$vol))
myts

# Деление выборки на тренировочную и тестовую (1000 - тренировочная, 200 - тестовая)
datalags = 20
train <-  myts[seq(1000 + datalags), ]
test <-  myts[1000 + datalags + seq(200 + datalags), ]
batch.size <- 125

# Создание массивов для последующей работы с ними
# Тренировочная выборка 
x.train <-  array(data = lag(cbind(train$price, train$vol), datalags)[-(1:datalags), ], dim = c(nrow(train) - datalags, datalags))
y.train = array(data = train$price[-(1:datalags)], dim = c(nrow(train)-datalags))
# Тестовая выборка
x.test <-  array(data = lag(cbind(test$vol, test$price), datalags)[-(1:datalags), ], dim = c(nrow(test) - datalags, datalags))
y.test <-  array(data = test$price[-(1:datalags)], dim = c(nrow(test) - datalags))

### SM-модель ###

# Создание архитектуры нейронной сети 
# Два слоя: в первом ф-ция активации - relu, во втором - sigmoid
sm <- keras_model_sequential() %>%
  layer_dense(units = 1000, activation = 'relu') %>%
  layer_dense(units = 5, activation = 'sigmoid')

# Добавляем оптимизатор и функцию потерь для нейронной сети. Точность не имеет смысла выводить на экран, т.к. она стремится к 0
### Mae_Adam ###
sm %>% compile(
  optimizer = 'adam',
  loss = 'mae')

# Обучаем нейронную сеть
sm %>% fit(x.train, y.train, epochs = 10, batch_size = 125)

# Получаем ответ
SM_Mae_Adam <- 0.2493 # величина ошибки SM-модели при ф-ции потерь Mae и оптимизаторе Adam

# Далее будем повторять алгоритм для остальных вариаций SM-модели

### Mae_Rmsprop ###
sm %>% compile(
  optimizer = 'rmsprop',
  loss = 'mae')
sm %>% fit(x.train, y.train, epochs = 10, batch_size = 125)
SM_Mae_RmsProp <- 0.2485 # величина ошибки SM-модели при ф-ции потерь Mae и оптимизаторе Rmsprop

### Mape_Adam ###
sm %>% compile(
  optimizer = 'adam',
  loss = 'mape')
sm %>% fit(x.train, y.train, epochs = 10, batch_size = 125)
SM_Mape_Adam <- 96.0233 # величина ошибки SM-модели при ф-ции потерь Mape и оптимизаторе Adam

### Mape_Rmsprop ###
sm %>% compile(
  optimizer = 'rmsprop',
  loss = 'mape')
sm %>% fit(x.train, y.train, epochs = 10, batch_size = 125)
SM_Mape_Rmsprop <- 95.5099 # величина ошибки SM-модели при ф-ции потерь Mape и оптимизаторе Rmsprop

### Mse_Adam ###
sm %>% compile(
  optimizer = 'adam',
  loss = 'mse')
sm %>% fit(x.train, y.train, epochs = 10, batch_size = 125)
SM_Mse_Adam <- 0.0848 # величина ошибки SM-модели при ф-ции потерь Mse и оптимизаторе Adam

### Mse_Rmsprop ###
sm %>% compile(
  optimizer = 'rmsprop',
  loss = 'mse')
sm %>% fit(x.train, y.train, epochs = 10, batch_size = 125)
SM_Mse_Rmsprop <- 0.0849 # величина ошибки SM-модели при ф-ции потерь Mse и оптимизаторе Rmsprop

# Немного изменим архитектуру сити, поменяв местами функции активации relu и sigmoid в слоях
sm <- keras_model_sequential() %>%
  layer_dense(units = 1000, activation = 'sigmoid') %>%
  layer_dense(units = 5, activation = 'relu')

# Проделаем аналогичные действия
### Mae_Adam ###
sm %>% compile(
  optimizer = 'adam',
  loss = 'mae')
sm %>% fit(x.train, y.train, epochs = 10, batch_size = 125)
SM_Mae_Adam_1 <- 0.3463 # величина ошибки SM-модели при ф-ции потерь Mae и оптимизаторе Adam

### Mae_Rmsprop ###
sm %>% compile(
  optimizer = 'rmsprop',
  loss = 'mae')
sm %>% fit(x.train, y.train, epochs = 10, batch_size = 125)
SM_Mae_Rmsprop_1 <- 0.4906 # величина ошибки SM-модели при ф-ции потерь Mae и оптимизаторе Rmsprop

### Mape_Adam ###
sm %>% compile(
  optimizer = 'adam',
  loss = 'mape')
sm %>% fit(x.train, y.train, epochs = 10, batch_size = 125)
SM_Mape_Adam_1 <- 100.0000 # величина ошибки SM-модели при ф-ции потерь Mape и оптимизаторе Adam

### Mape_Rmsprop ###
sm %>% compile(
  optimizer = 'rmsprop',
  loss = 'mape')
sm %>% fit(x.train, y.train, epochs = 10, batch_size = 125)
SM_Mape_Rmsprop_1 <- 100.0000 # величина ошибки SM-модели при ф-ции потерь Mape и оптимизаторе Rmsprop

### Mse_Adam ###
sm %>% compile(
  optimizer = 'adam',
  loss = 'mse')
sm %>% fit(x.train, y.train, epochs = 10, batch_size = 125)
SM_Mse_Adam_1 <- 0.3236 # величина ошибки SM-модели при ф-ции потерь Mse и оптимизаторе Adam

### Mse_Rmsprop ###
sm %>% compile(
  optimizer = 'rmsprop',
  loss = 'mse')
sm %>% fit(x.train, y.train, epochs = 10, batch_size = 125)
SM_Mse_Rmsprop_1 <- 0.3236 # величина ошибки SM-модели при ф-ции потерь Mse и оптимизаторе Rmsprop

### LSTM-модель ###
### По сравнению с SM-моделью изменения начинаются с этапа стандартизации данных

# Стандартизация данных посредством z-оценки
msd.price <-  c(mean(myts$price), sd(myts$price))
msd.vol <-  c(mean(myts$vol), sd(myts$vol))
myts$price <-  (myts$price - msd.price[1])/msd.price[2]
myts$vol <-  (myts$vol - msd.vol[1])/msd.vol[2]
summary(myts)

# Деление выборки на тренировочную и тестовую (1000 - тренировочная, 200 - тестовая)
datalags = 20
train <-  myts[seq(1000 + datalags), ]
test <-  myts[1000 + datalags + seq(200 + datalags), ]
batch.size <- 125

# Создание массивов для последующей работы с ними
# Тренировочная выборка 
x.train <-  array(data = lag(cbind(train$price, train$vol), datalags)[-(1:datalags), ], dim = c(nrow(train) - datalags, datalags, 2))
y.train = array(data = train$price[-(1:datalags)], dim = c(nrow(train)-datalags, 1))
# Тестовая выборка
x.test <-  array(data = lag(cbind(test$vol, test$price), datalags)[-(1:datalags), ], dim = c(nrow(test) - datalags, datalags, 2))
y.test <-  array(data = test$price[-(1:datalags)], dim = c(nrow(test) - datalags, 1))

# Создание архитектуры нейронной сети 
# Двухслойная LSTM-модель
model <- keras_model_sequential()  %>%
  layer_lstm(units = 100,
             input_shape = c(datalags, 2),
             batch_size = batch.size,
             return_sequences = TRUE,
             stateful = TRUE) %>%
  layer_dropout(rate = 0.5) %>%
  layer_lstm(units = 50,
             return_sequences = FALSE,
             stateful = TRUE) %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 1)

# Добавляем оптимизатор и функцию потерь для нейронной сети. Точность не имеет смысла выводить на экран, т.к. она стремится к 0
### Mae_Adam ###
model %>% compile(
  optimizer = 'adam',
  loss = 'mae')

# Обучаем нейронную сеть
model %>% fit(x.train, y.train, epochs = 10, batch_size = 125)

# Получаем ответ
LSTM_Mae_Adam <- 0.4300 # величина ошибки LSTM-модели при ф-ции потерь Mae и оптимизаторе Adam

# Далее будем повторять алгоритм для остальных вариаций LSTM-модели

### Mae_Rmsprop ###
model %>% compile(
  optimizer = 'rmsprop',
  loss = 'mae')
model %>% fit(x.train, y.train, epochs = 10, batch_size = 125)
LSTM_Mae_Rmsprop <- 0.3693 # величина ошибки LSTM-модели при ф-ции потерь Mae и оптимизаторе Rmsprop

### Mape_Adam ###
model %>% compile(
  optimizer = 'adam',
  loss = 'mape')
model %>% fit(x.train, y.train, epochs = 10, batch_size = 125)
LSTM_Mape_Adam <- 98.8686 # величина ошибки LSTM-модели при ф-ции потерь Mape и оптимизаторе Adam

### Mape_Rmsprop ###
model %>% compile(
  optimizer = 'rmsprop',
  loss = 'mape')
model %>% fit(x.train, y.train, epochs = 10, batch_size = 125)
LSTM_Mape_Rmsprop <- 102.1346 # величина ошибки LSTM-модели при ф-ции потерь Mape и оптимизаторе Rmsprop

### Mse_Adam ###
model %>% compile(
  optimizer = 'adam',
  loss = 'mse')
model %>% fit(x.train, y.train, epochs = 10, batch_size = 125)
LSTM_Mse_Adam <- 0.2044 # величина ошибки LSTM-модели при ф-ции потерь Mse и оптимизаторе Adam

### Mse_Rmsprop ###
model %>% compile(
  optimizer = 'rmsprop',
  loss = 'mse')
model %>% fit(x.train, y.train, epochs = 10, batch_size = 125)
LSTM_Mse_Rmsprop <- 0.1612 # величина ошибки LSTM-модели при ф-ции потерь Mse и оптимизаторе Rmsprop

### RNN-модель ###
### Изменения начинаются с этапа стандартизации данных

# Стандартизация данных методом min-max
myts <- data.frame(index = rminimax(myts$index), price = rminimax(myts$price), vol= rminimax(myts$vol))
myts

# Деление выборки на тренировочную и тестовую (1000 - тренировочная, 200 - тестовая)
datalags = 20
train <-  myts[seq(1000 + datalags), ]
test <-  myts[1000 + datalags + seq(200 + datalags), ]
batch.size <- 125

# Создание массивов для последующей работы с ними
# Тренировочная выборка 
x.train <-  array(data = lag(cbind(train$price, train$vol), datalags)[-(1:datalags), ], dim = c(nrow(train) - datalags, datalags))
y.train = array(data = train$price[-(1:datalags)], dim = c(nrow(train)-datalags))
# Тестовая выборка
x.test <-  array(data = lag(cbind(test$vol, test$price), datalags)[-(1:datalags), ], dim = c(nrow(test) - datalags, datalags))
y.test <-  array(data = test$price[-(1:datalags)], dim = c(nrow(test) - datalags))

# Создание архитектуры нейронной сети 
# Обучение RNN-моделей с функцией активации relu
model <- keras_model_sequential() %>%
  layer_embedding(input_dim = 10000, output_dim = 50) %>%
  layer_simple_rnn(units = 50) %>%
  layer_dense(units = 1, activation = "relu")

# Добавляем оптимизатор и функцию потерь для нейронной сети. Точность не имеет смысла выводить на экран, т.к. она стремится к 0
### Mae_Adam ###
model %>% compile(
  optimizer = "adam",
  loss = "mae",
)

# Обучаем нейронную сеть
# Будем пробовать для каждой модели разные validation_split = 0.00; 0.12; 0.15; 0.20; 0.25
history <- model %>% fit(
  x.train, y.train,
  epochs = 10,
  batch_size = 125,
  validation_split = 0.12)

# Получаем ответ
RNN_Mae_Adam <- 0.4861 # величина ошибки RNN-модели при ф-ции потерь Mae и оптимизаторе Adam

# validation split = 0.15
history <- model %>% fit(
  x.train, y.train,
  epochs = 10,
  batch_size = 125,
  validation_split = 0.15)

# Получаем ответ
RNN_Mae_Adam_1 <- 0.4801 # величина ошибки RNN-модели при ф-ции потерь Mae и оптимизаторе Adam

# validation split = 0.20
history <- model %>% fit(
  x.train, y.train,
  epochs = 10,
  batch_size = 125,
  validation_split = 0.20)

# Получаем ответ
RNN_Mae_Adam_2 <- 0.4733 # величина ошибки RNN-модели при ф-ции потерь Mae и оптимизаторе Adam

# validation split = 0.25
history <- model %>% fit(
  x.train, y.train,
  epochs = 10,
  batch_size = 125,
  validation_split = 0.25)

# Получаем ответ
RNN_Mae_Adam_3 <- 0.4888 # величина ошибки RNN-модели при ф-ции потерь Mae и оптимизаторе Adam

# validation split = 0.00
history <- model %>% fit(
  x.train, y.train,
  epochs = 10,
  batch_size = 125)

# Получаем ответ
RNN_Mae_Adam_4 <- 0.4942 # величина ошибки RNN-модели при ф-ции потерь Mae и оптимизаторе Adam

# Далее будем повторять алгоритм для остальных вариаций RNN-модели

### Mae_Rmsprop ###
model %>% compile(
  optimizer = "rmsprop",
  loss = "mae",
)

# validation split = 0.12
history <- model %>% fit(
  x.train, y.train,
  epochs = 10,
  batch_size = 125,
  validation_split = 0.12)

# Получаем ответ
RNN_Mae_Rmsprop <- 0.4861 # величина ошибки RNN-модели при ф-ции потерь Mae и оптимизаторе Rmsprop

# validation split = 0.15
history <- model %>% fit(
  x.train, y.train,
  epochs = 10,
  batch_size = 125,
  validation_split = 0.15)

# Получаем ответ
RNN_Mae_Rmsprop_1 <- 0.4801 # величина ошибки RNN-модели при ф-ции потерь Mae и оптимизаторе Rmsprop

# validation split = 0.20
history <- model %>% fit(
  x.train, y.train,
  epochs = 10,
  batch_size = 125,
  validation_split = 0.20)

# Получаем ответ
RNN_Mae_Rmsprop_2 <- 0.4733 # величина ошибки RNN-модели при ф-ции потерь Mae и оптимизаторе Rmsprop

# validation split = 0.25
history <- model %>% fit(
  x.train, y.train,
  epochs = 10,
  batch_size = 125,
  validation_split = 0.25)

# Получаем ответ
RNN_Mae_Rmsprop_3 <- 0.4888 # величина ошибки RNN-модели при ф-ции потерь Mae и оптимизаторе Rmsprop

# validation split = 0.00
history <- model %>% fit(
  x.train, y.train,
  epochs = 10,
  batch_size = 125)

# Получаем ответ
RNN_Mae_Rmsprop_4 <- 0.4942 # величина ошибки RNN-модели при ф-ции потерь Mae и оптимизаторе Rmsprop

### Mape_Adam ###
model %>% compile(
  optimizer = "adam",
  loss = "mape",
)

# validation split = 0.12
history <- model %>% fit(
  x.train, y.train,
  epochs = 10,
  batch_size = 125,
  validation_split = 0.12)

# Получаем ответ
RNN_Mape_Adam <- 100.0000 # величина ошибки RNN-модели при ф-ции потерь Mape и оптимизаторе Adam

# validation split = 0.15
history <- model %>% fit(
  x.train, y.train,
  epochs = 10,
  batch_size = 125,
  validation_split = 0.15)

# Получаем ответ
RNN_Mape_Adam_1 <- 100.0000 # величина ошибки RNN-модели при ф-ции потерь Mape и оптимизаторе Adam

# validation split = 0.20
history <- model %>% fit(
  x.train, y.train,
  epochs = 10,
  batch_size = 125,
  validation_split = 0.20)

# Получаем ответ
RNN_Mape_Adam_2 <- 100.0000 # величина ошибки RNN-модели при ф-ции потерь Mape и оптимизаторе Adam

# validation split = 0.25
history <- model %>% fit(
  x.train, y.train,
  epochs = 10,
  batch_size = 125,
  validation_split = 0.25)

# Получаем ответ
RNN_Mape_Adam_3 <- 100.0000 # величина ошибки RNN-модели при ф-ции потерь Mape и оптимизаторе Adam

# validation split = 0.00
history <- model %>% fit(
  x.train, y.train,
  epochs = 10,
  batch_size = 125)

# Получаем ответ
RNN_Mape_Adam_4 <- 100.0000 # величина ошибки RNN-модели при ф-ции потерь Mape и оптимизаторе Adam

### Mape_Rmsprop ###
model %>% compile(
  optimizer = "rmsprop",
  loss = "mape",
)

# validation split = 0.12
history <- model %>% fit(
  x.train, y.train,
  epochs = 10,
  batch_size = 125,
  validation_split = 0.12)

# Получаем ответ
RNN_Mape_Rmsprop <- 100.0000 # величина ошибки RNN-модели при ф-ции потерь Mape и оптимизаторе Rmsprop

# validation split = 0.15
history <- model %>% fit(
  x.train, y.train,
  epochs = 10,
  batch_size = 125,
  validation_split = 0.15)

# Получаем ответ
RNN_Mape_Rmsprop_1 <- 100.0000 # величина ошибки RNN-модели при ф-ции потерь Mape и оптимизаторе Rmsprop

# validation split = 0.20
history <- model %>% fit(
  x.train, y.train,
  epochs = 10,
  batch_size = 125,
  validation_split = 0.20)

# Получаем ответ
RNN_Mape_Rmsprop_2 <- 100.0000 # величина ошибки RNN-модели при ф-ции потерь Mape и оптимизаторе Rmsprop

# validation split = 0.25
history <- model %>% fit(
  x.train, y.train,
  epochs = 10,
  batch_size = 125,
  validation_split = 0.25)

# Получаем ответ
RNN_Mape_Rmsprop_3 <- 100.0000 # величина ошибки RNN-модели при ф-ции потерь Mape и оптимизаторе Rmsprop

# validation split = 0.00
history <- model %>% fit(
  x.train, y.train,
  epochs = 10,
  batch_size = 125)

# Получаем ответ
RNN_Mape_Rmsprop_4 <- 100.0000 # величина ошибки RNN-модели при ф-ции потерь Mape и оптимизаторе Rmsprop

### Mse_Adam ###
model %>% compile(
  optimizer = "adam",
  loss = "mse",
)

# validation split = 0.12
history <- model %>% fit(
  x.train, y.train,
  epochs = 10,
  batch_size = 125,
  validation_split = 0.12)

# Получаем ответ
RNN_Mse_Adam <- 0.3268 # величина ошибки RNN-модели при ф-ции потерь Mse и оптимизаторе Adam

# validation split = 0.15
history <- model %>% fit(
  x.train, y.train,
  epochs = 10,
  batch_size = 125,
  validation_split = 0.15)

# Получаем ответ
RNN_Mse_Adam_1 <- 0.3217 # величина ошибки RNN-модели при ф-ции потерь Mse и оптимизаторе Adam

# validation split = 0.20
history <- model %>% fit(
  x.train, y.train,
  epochs = 10,
  batch_size = 125,
  validation_split = 0.20)

# Получаем ответ
RNN_Mse_Adam_2 <- 0.3123 # величина ошибки RNN-модели при ф-ции потерь Mse и оптимизаторе Adam

# validation split = 0.25
history <- model %>% fit(
  x.train, y.train,
  epochs = 10,
  batch_size = 125,
  validation_split = 0.25)

# Получаем ответ
RNN_Mse_Adam_3 <- 0.3339 # величина ошибки RNN-модели при ф-ции потерь Mse и оптимизаторе Adam

# validation split = 0.00
history <- model %>% fit(
  x.train, y.train,
  epochs = 10,
  batch_size = 125)

# Получаем ответ
RNN_Mse_Adam_4 <- 0.3293 # величина ошибки RNN-модели при ф-ции потерь Mse и оптимизаторе Adam

### Mse_Rmsprop ###
model %>% compile(
  optimizer = "rmsprop",
  loss = "mse",
)

# validation split = 0.12
history <- model %>% fit(
  x.train, y.train,
  epochs = 10,
  batch_size = 125,
  validation_split = 0.12)

# Получаем ответ
RNN_Mse_Rmsprop <- 0.3268 # величина ошибки RNN-модели при ф-ции потерь Mse и оптимизаторе Rmsprop

# validation split = 0.15
history <- model %>% fit(
  x.train, y.train,
  epochs = 10,
  batch_size = 125,
  validation_split = 0.15)

# Получаем ответ
RNN_Mse_Rmsprop_1 <- 0.3217 # величина ошибки RNN-модели при ф-ции потерь Mse и оптимизаторе Rmsprop

# validation split = 0.20
history <- model %>% fit(
  x.train, y.train,
  epochs = 10,
  batch_size = 125,
  validation_split = 0.20)

# Получаем ответ
RNN_Mse_Rmsprop_2 <- 0.3123 # величина ошибки RNN-модели при ф-ции потерь Mse и оптимизаторе Rmsprop

# validation split = 0.25
history <- model %>% fit(
  x.train, y.train,
  epochs = 10,
  batch_size = 125,
  validation_split = 0.25)

# Получаем ответ
RNN_Mse_Rmsprop_3 <- 0.3339 # величина ошибки RNN-модели при ф-ции потерь Mse и оптимизаторе Rmsprop

# validation split = 0.00
history <- model %>% fit(
  x.train, y.train,
  epochs = 10,
  batch_size = 125)

# Получаем ответ
RNN_Mse_Rmsprop_4 <- 0.3293 # величина ошибки RNN-модели при ф-ции потерь Mse и оптимизаторе Rmsprop

# Обучение RNN-моделей с функцией активации sigmoid
model <- keras_model_sequential() %>%
  layer_embedding(input_dim = 10000, output_dim = 50) %>%
  layer_simple_rnn(units = 50) %>%
  layer_dense(units = 1, activation = "sigmoid")

### Mae_Adam ###
model %>% compile(
  optimizer = "adam",
  loss = "mae",
)

# Будем пробовать для каждой модели разные validation_split = 0.00; 0.12; 0.15; 0.20; 0.25
history <- model %>% fit(
  x.train, y.train,
  epochs = 10,
  batch_size = 125,
  validation_split = 0.12)

# Получаем ответ
RNN_Mae_Adam_Sg <- 0.2692 # величина ошибки RNN-модели при ф-ции потерь Mae и оптимизаторе Adam

# validation split = 0.15
history <- model %>% fit(
  x.train, y.train,
  epochs = 10,
  batch_size = 125,
  validation_split = 0.15)

# Получаем ответ
RNN_Mae_Adam_Sg_1 <- 0.2693 # величина ошибки RNN-модели при ф-ции потерь Mae и оптимизаторе Adam

# validation split = 0.20
history <- model %>% fit(
  x.train, y.train,
  epochs = 10,
  batch_size = 125,
  validation_split = 0.20)

# Получаем ответ
RNN_Mae_Adam_Sg_2 <- 0.2614 # величина ошибки RNN-модели при ф-ции потерь Mae и оптимизаторе Adam

# validation split = 0.25
history <- model %>% fit(
  x.train, y.train,
  epochs = 10,
  batch_size = 125,
  validation_split = 0.25)

# Получаем ответ
RNN_Mae_Adam_Sg_3 <- 0.2720 # величина ошибки RNN-модели при ф-ции потерь Mae и оптимизаторе Adam

# validation split = 0.00
history <- model %>% fit(
  x.train, y.train,
  epochs = 10,
  batch_size = 125)

# Получаем ответ
RNN_Mae_Adam_Sg_4 <- 0.2522 # величина ошибки RNN-модели при ф-ции потерь Mae и оптимизаторе Adam

# Далее будем повторять алгоритм для остальных вариаций RNN-модели

### Mae_Rmsprop ###
model %>% compile(
  optimizer = "rmsprop",
  loss = "mae",
)

# validation split = 0.12
history <- model %>% fit(
  x.train, y.train,
  epochs = 10,
  batch_size = 125,
  validation_split = 0.12)

# Получаем ответ
RNN_Mae_Rmsprop_Sg <- 0.2692 # величина ошибки RNN-модели при ф-ции потерь Mae и оптимизаторе Rmsprop

# validation split = 0.15
history <- model %>% fit(
  x.train, y.train,
  epochs = 10,
  batch_size = 125,
  validation_split = 0.15)

# Получаем ответ
RNN_Mae_Rmsprop_Sg_1 <- 0.2693 # величина ошибки RNN-модели при ф-ции потерь Mae и оптимизаторе Rmsprop

# validation split = 0.20
history <- model %>% fit(
  x.train, y.train,
  epochs = 10,
  batch_size = 125,
  validation_split = 0.20)

# Получаем ответ
RNN_Mae_Rmsprop_Sg_2 <- 0.2614 # величина ошибки RNN-модели при ф-ции потерь Mae и оптимизаторе Rmsprop

# validation split = 0.25
history <- model %>% fit(
  x.train, y.train,
  epochs = 10,
  batch_size = 125,
  validation_split = 0.25)

# Получаем ответ
RNN_Mae_Rmsprop_Sg_3 <- 0.2720 # величина ошибки RNN-модели при ф-ции потерь Mae и оптимизаторе Rmsprop

# validation split = 0.00
history <- model %>% fit(
  x.train, y.train,
  epochs = 10,
  batch_size = 125)

# Получаем ответ
RNN_Mae_Rmsprop_Sg_4 <- 0.2521 # величина ошибки RNN-модели при ф-ции потерь Mae и оптимизаторе Rmsprop

### Mape_Adam ###
model %>% compile(
  optimizer = "adam",
  loss = "mape",
)

# validation split = 0.12
history <- model %>% fit(
  x.train, y.train,
  epochs = 10,
  batch_size = 125,
  validation_split = 0.12)

# Получаем ответ
RNN_Mape_Adam_Sg <- 94.4158 # величина ошибки RNN-модели при ф-ции потерь Mape и оптимизаторе Adam

# validation split = 0.15
history <- model %>% fit(
  x.train, y.train,
  epochs = 10,
  batch_size = 125,
  validation_split = 0.15)

# Получаем ответ
RNN_Mape_Adam_Sg_1 <- 95.2362 # величина ошибки RNN-модели при ф-ции потерь Mape и оптимизаторе Adam

# validation split = 0.20
history <- model %>% fit(
  x.train, y.train,
  epochs = 10,
  batch_size = 125,
  validation_split = 0.20)

# Получаем ответ
RNN_Mape_Adam_Sg_2 <- 95.9408 # величина ошибки RNN-модели при ф-ции потерь Mape и оптимизаторе Adam

# validation split = 0.25
history <- model %>% fit(
  x.train, y.train,
  epochs = 10,
  batch_size = 125,
  validation_split = 0.25)

# Получаем ответ
RNN_Mape_Adam_Sg_3 <- 96.2572 # величина ошибки RNN-модели при ф-ции потерь Mape и оптимизаторе Adam

# validation split = 0.00
history <- model %>% fit(
  x.train, y.train,
  epochs = 10,
  batch_size = 125)

# Получаем ответ
RNN_Mape_Adam_Sg_4 <- 99.1607 # величина ошибки RNN-модели при ф-ции потерь Mape и оптимизаторе Adam

### Mape_Rmsprop ###
model %>% compile(
  optimizer = "rmsprop",
  loss = "mape",
)

# validation split = 0.12
history <- model %>% fit(
  x.train, y.train,
  epochs = 10,
  batch_size = 125,
  validation_split = 0.12)

# Получаем ответ
RNN_Mape_Rmsprop_Sg <- 96.9550 # величина ошибки RNN-модели при ф-ции потерь Mape и оптимизаторе Rmsprop

# validation split = 0.15
history <- model %>% fit(
  x.train, y.train,
  epochs = 10,
  batch_size = 125,
  validation_split = 0.15)

# Получаем ответ
RNN_Mape_Rmsprop_Sg_1 <- 97.6969 # величина ошибки RNN-модели при ф-ции потерь Mape и оптимизаторе Rmsprop

# validation split = 0.20
history <- model %>% fit(
  x.train, y.train,
  epochs = 10,
  batch_size = 125,
  validation_split = 0.20)

# Получаем ответ
RNN_Mape_Rmsprop_Sg_2 <- 98.1924 # величина ошибки RNN-модели при ф-ции потерь Mape и оптимизаторе Rmsprop

# validation split = 0.25
history <- model %>% fit(
  x.train, y.train,
  epochs = 10,
  batch_size = 125,
  validation_split = 0.25)

# Получаем ответ
RNN_Mape_Rmsprop_Sg_3 <- 97.7242 # величина ошибки RNN-модели при ф-ции потерь Mape и оптимизаторе Rmsprop

# validation split = 0.00
history <- model %>% fit(
  x.train, y.train,
  epochs = 10,
  batch_size = 125)

# Получаем ответ
RNN_Mape_Rmsprop_Sg_4 <- 98.5460 # величина ошибки RNN-модели при ф-ции потерь Mape и оптимизаторе Rmsprop

### Mse_Adam ###
model %>% compile(
  optimizer = "adam",
  loss = "mse",
)

# validation split = 0.12
history <- model %>% fit(
  x.train, y.train,
  epochs = 10,
  batch_size = 125,
  validation_split = 0.12)

# Получаем ответ
RNN_Mse_Adam_Sg <- 0.0906 # величина ошибки RNN-модели при ф-ции потерь Mse и оптимизаторе Adam

# validation split = 0.15
history <- model %>% fit(
  x.train, y.train,
  epochs = 10,
  batch_size = 125,
  validation_split = 0.15)

# Получаем ответ
RNN_Mse_Adam_Sg_1 <- 0.0920 # величина ошибки RNN-модели при ф-ции потерь Mse и оптимизаторе Adam

# validation split = 0.20
history <- model %>% fit(
  x.train, y.train,
  epochs = 10,
  batch_size = 125,
  validation_split = 0.20)

# Получаем ответ
RNN_Mse_Adam_Sg_2 <- 0.0893 # величина ошибки RNN-модели при ф-ции потерь Mse и оптимизаторе Adam

# validation split = 0.25
history <- model %>% fit(
  x.train, y.train,
  epochs = 10,
  batch_size = 125,
  validation_split = 0.25)

# Получаем ответ
RNN_Mse_Adam_Sg_3 <- 0.0950 # величина ошибки RNN-модели при ф-ции потерь Mse и оптимизаторе Adam

# validation split = 0.00
history <- model %>% fit(
  x.train, y.train,
  epochs = 10,
  batch_size = 125)

# Получаем ответ
RNN_Mse_Adam_Sg_4 <- 0.0857 # величина ошибки RNN-модели при ф-ции потерь Mse и оптимизаторе Adam

### Mse_Rmsprop ###
model %>% compile(
  optimizer = "rmsprop",
  loss = "mse",
)

# validation split = 0.12
history <- model %>% fit(
  x.train, y.train,
  epochs = 10,
  batch_size = 125,
  validation_split = 0.12)

# Получаем ответ
RNN_Mse_Rmsprop_Sg <- 0.1188 # величина ошибки RNN-модели при ф-ции потерь Mse и оптимизаторе Rmsprop

# validation split = 0.15
history <- model %>% fit(
  x.train, y.train,
  epochs = 10,
  batch_size = 125,
  validation_split = 0.15)

# Получаем ответ
RNN_Mse_Rmsprop_Sg_1 <- 0.0952 # величина ошибки RNN-модели при ф-ции потерь Mse и оптимизаторе Rmsprop

# validation split = 0.20
history <- model %>% fit(
  x.train, y.train,
  epochs = 10,
  batch_size = 125,
  validation_split = 0.20)

# Получаем ответ
RNN_Mse_Rmsprop_Sg_2 <- 0.0888 # величина ошибки RNN-модели при ф-ции потерь Mse и оптимизаторе Rmsprop

# validation split = 0.25
history <- model %>% fit(
  x.train, y.train,
  epochs = 10,
  batch_size = 125,
  validation_split = 0.25)

# Получаем ответ
RNN_Mse_Rmsprop_Sg_3 <- 0.0951 # величина ошибки RNN-модели при ф-ции потерь Mse и оптимизаторе Rmsprop

# validation split = 0.00
history <- model %>% fit(
  x.train, y.train,
  epochs = 10,
  batch_size = 125)

# Получаем ответ
RNN_Mse_Rmsprop_Sg_4 <- 0.0856 # величина ошибки RNN-модели при ф-ции потерь Mse и оптимизаторе Rmsprop

# Подготавливаем данные для итоговой сводной таблицы - по 26 моделей на каждый оптимизатор
mae <- c(SM_Mae_Adam, SM_Mae_RmsProp, SM_Mae_Adam_1, SM_Mae_Rmsprop_1, LSTM_Mae_Adam, LSTM_Mae_Rmsprop, 
RNN_Mae_Adam, RNN_Mae_Adam_1, RNN_Mae_Adam_2, RNN_Mae_Adam_3, RNN_Mae_Adam_4, RNN_Mae_Rmsprop, RNN_Mae_Rmsprop_1, 
RNN_Mae_Rmsprop_2, RNN_Mae_Rmsprop_3, RNN_Mae_Rmsprop_4, RNN_Mae_Adam_Sg, RNN_Mae_Adam_Sg_1, RNN_Mae_Adam_Sg_2,
RNN_Mae_Adam_Sg_3, RNN_Mae_Adam_Sg_4, RNN_Mae_Rmsprop_Sg, RNN_Mae_Rmsprop_Sg_1, RNN_Mae_Rmsprop_Sg_2, 
RNN_Mae_Rmsprop_Sg_3, RNN_Mae_Rmsprop_Sg_4)

mape <- c(SM_Mape_Adam, SM_Mape_Rmsprop, SM_Mape_Adam_1, SM_Mape_Rmsprop_1, LSTM_Mape_Adam, LSTM_Mape_Rmsprop, 
RNN_Mape_Adam, RNN_Mape_Adam_1, RNN_Mape_Adam_2, RNN_Mape_Adam_3, RNN_Mape_Adam_4, RNN_Mape_Rmsprop, RNN_Mape_Rmsprop_1, 
RNN_Mape_Rmsprop_2, RNN_Mape_Rmsprop_3, RNN_Mape_Rmsprop_4, RNN_Mape_Adam_Sg, RNN_Mape_Adam_Sg_1, RNN_Mape_Adam_Sg_2,
RNN_Mape_Adam_Sg_3, RNN_Mape_Adam_Sg_4, RNN_Mape_Rmsprop_Sg, RNN_Mape_Rmsprop_Sg_1, RNN_Mape_Rmsprop_Sg_2, 
RNN_Mape_Rmsprop_Sg_3, RNN_Mape_Rmsprop_Sg_4)

mse <- c(SM_Mse_Adam, SM_Mse_Rmsprop, SM_Mse_Adam_1, SM_Mse_Rmsprop_1, LSTM_Mse_Adam, LSTM_Mse_Rmsprop, 
RNN_Mse_Adam, RNN_Mse_Adam_1, RNN_Mse_Adam_2, RNN_Mse_Adam_3, RNN_Mse_Adam_4, RNN_Mse_Rmsprop, RNN_Mse_Rmsprop_1, 
RNN_Mse_Rmsprop_2, RNN_Mse_Rmsprop_3, RNN_Mse_Rmsprop_4, RNN_Mse_Adam_Sg, RNN_Mse_Adam_Sg_1, RNN_Mse_Adam_Sg_2,
RNN_Mse_Adam_Sg_3, RNN_Mse_Adam_Sg_4, RNN_Mse_Rmsprop_Sg, RNN_Mse_Rmsprop_Sg_1, RNN_Mse_Rmsprop_Sg_2, 
RNN_Mse_Rmsprop_Sg_3, RNN_Mse_Rmsprop_Sg_4)

# Классификация моделей по алгоритмам
SM <- c("SM_Adam", "SM_Rmsprop", "SM_Adam_1", "SM_Rmsprop_1")

LSTM <- c("LSTM_Adam", "LSTM_Rmsprop")

RNN <- c("RNN_Adam", "RNN_Adam_1", "RNN_Adam_2", "RNN_Adam_3", "RNN_Adam_4", 
"RNN_Rmsprop", "RNN_Rmsprop_1", "RNN_Rmsprop_2", "RNN_Rmsprop_3", "RNN_Rmsprop_4",
"RNN_Adam_Sg", "RNN_Adam_Sg_1", "RNN_Adam_Sg_2", "RNN_Adam_Sg_3", "RNN_Adam_Sg_4",
"RNN_Rmsprop_Sg", "RNN_Rmsprop_Sg_1", "RNN_Rmsprop_Sg_2", "RNN_Rmsprop_Sg_3", "RNN_Rmsprop_Sg_4")

Model <- c(SM, LSTM, RNN)

# Сводим в единую таблицу полученные результаты
data.frame(MODEL = Model, MAE = mae, MAPE = mape, MSE = mse)
### Делаем вывод, что лучшая модель - SM с функцией потерь mse и оптимизатором adam
### loss = 0.0848
### SM, mse, adam - лучшая модель

# Делаем прогноз по тестовой выборке для лучшей модели
pred_out <- sm %>% predict(x.test, batch_size = batch.size) %>% .[,1]

# Строим графики отклонений для прогноза
plot(y.test - pred_out, type = 'line')
plot(x = y.test, y = pred_out)
