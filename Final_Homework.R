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

# Далее будем повторять алгоритм для остальных вариаций SM модели

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
SM_Mse_Adam <- 0.0829 # величина ошибки SM-модели при ф-ции потерь Mse и оптимизаторе Adam

### Mse_Rmsprop ###
sm %>% compile(
  optimizer = 'rmsprop',
  loss = 'mse')
sm %>% fit(x.train, y.train, epochs = 10, batch_size = 125)
SM_Mse_Rmsprop <- 0.0829 # величина ошибки SM-модели при ф-ции потерь Mse и оптимизаторе Rmsprop

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
batch.size <- 50

# Создание массивов для последующей работы с ними
# Тренировочная выборка 
x.train <-  array(data = lag(cbind(train$price, train$vol), datalags)[-(1:datalags), ], dim = c(nrow(train) - datalags, datalags, 2))
y.train = array(data = train$price[-(1:datalags)], dim = c(nrow(train)-datalags, 1))
# Тестовая выборка
x.test <-  array(data = lag(cbind(test$vol, test$price), datalags)[-(1:datalags), ], dim = c(nrow(test) - datalags, datalags, 2))
y.test <-  array(data = test$price[-(1:datalags)], dim = c(nrow(test) - datalags, 1))
