##### USA - Dow Jones Industrial Average (^DJI)
############################

# Установка необходимых пакетов
install.packages('BatchGetSymbols')
install.packages('plotly')
install.packages('keras')
install.packages('tensorflow')

# Обращение к установленным пакетам
library('BatchGetSymbols')
library('plotly')
library('keras')
library('tensorflow')

# Задание values - Индекс Dow Jones Industrial Average (^DJI) в период 26.06.15 - 30.05.20  
tickers <- c('%5EDJI')
first.date <- Sys.Date() - 360*5
last.date <- Sys.Date()

# Загрузка данных (Got 100% of valid prices | Feels good!)
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

# Вывод на экран графика - Цена закрытия индекса Dow Jones Industrial Average (^DJI) за рассматриваемый период
plot_ly(myts, x = ~index, y = ~price, type = "scatter", mode = "markers", color = ~vol)

# Проверка автокорреляции ряда
acf(myts$price, lag.max = 3000)

# стандартизация данных посредством z-оценки
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
model %>% fit(x.train, y.train, epochs = 10, batch_size = batch.size)

# Получаем ответ
LSTM_Mae_Adam <- 0.2809 # величина ошибки LSTM-модели при ф-ции потерь Mae и оптимизаторе Adam

### Mae_Rmsprop ###
model %>% compile(
  optimizer = 'rmsprop',
  loss = 'mae')
model %>% fit(x.train, y.train, epochs = 10, batch_size = batch.size)
LSTM_Mae_Rmsprop <- 0.2764 # величина ошибки LSTM-модели при ф-ции потерь Mae и оптимизаторе Rmsprop

### Mape_Adam ###
model %>% compile(
  optimizer = 'adam',
  loss = 'mape')
model %>% fit(x.train, y.train, epochs = 10, batch_size = 125)
LSTM_Mape_Adam <- 55.6054 # величина ошибки LSTM-модели при ф-ции потерь Mape и оптимизаторе Adam

### Mape_Rmsprop ###
model %>% compile(
  optimizer = 'rmsprop',
  loss = 'mape')
model %>% fit(x.train, y.train, epochs = 10, batch_size = 125)
LSTM_Mape_Rmsprop <- 52.6158 # величина ошибки LSTM-модели при ф-ции потерь Mape и оптимизаторе Rmsprop

### Mse_Adam ###
model %>% compile(
  optimizer = 'adam',
  loss = 'mse')
model %>% fit(x.train, y.train, epochs = 10, batch_size = batch.size)
LSTM_Mse_Adam <- 0.1060 # величина ошибки LSTM-модели при ф-ции потерь Mse и оптимизаторе Adam

### Mse_Rmsprop ###
model %>% compile(
  optimizer = 'rmsprop',
  loss = 'mse')
model %>% fit(x.train, y.train, epochs = 10, batch_size = batch.size)
LSTM_Mse_Rmsprop <- 0.1059 # величина ошибки LSTM-модели при ф-ции потерь Mse и оптимизаторе Rmsprop

# Подготавливаем данные для итоговой сводной таблицы - по 2 модели на каждый оптимизатор
mae <- c(LSTM_Mae_Adam, LSTM_Mae_Rmsprop)
mape <- c(LSTM_Mape_Adam, LSTM_Mape_Rmsprop)
mse <- c(LSTM_Mse_Adam, LSTM_Mse_Rmsprop)

# Классификация моделей по алгоритмам
LSTM <- c('LSTM_Adam', 'LSTM_Rmsprop')

# Сводим в единую таблицу полученные результаты
data.frame(MODEL = LSTM, MAE = mae, MAPE = mape, MSE = mse)
### Делаем вывод, что лучшая модель - LSTM с функцией потерь mse и оптимизатором rmsprop
### loss = 0.1059
### LSTM, mse, rmsprop - лучшая модель

# Делаем прогноз по тестовой выборке для лучшей модели
pred_out <- model %>% predict(x.test, batch_size = batch.size) %>% .[,1]

# Визуализируем прогноз модели
plot_ly(myts, x = ~index, y = ~price, type = "scatter", mode = "markers", color = ~vol) %>%
    add_trace(y = c(rep(NA, 2000), pred_out), x = myts$index, name = "LSTM prediction", color = 'black')

# Строим графики отклонений для прогноза
plot(y.test - pred_out, type = 'line')
plot(x = y.test, y = pred_out)
