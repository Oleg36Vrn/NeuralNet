# Устанавливаем пакеты keras и tensorflow с CRAN и запускаем их
install.packages('tensorflow')
install.packages('keras')
library('keras')
library('tensorflow')

# Устанавливаем Keras
install_keras()

#####
# При ошибке "AttributeError: module 'tensorflow' has no attribute 'VERSION'"
# Делаем откат до одной из предыдущих версий: 1.12
tensorflow::install_tensorflow(version = '1.12')

# Установим число признаков, по которым ищется разбиение
max_features <- 10000

# Загружаем данные
mnist <- dataset_mnist()
str(mnist) #Смотрим на загруженный массив

# Разбиваем исходные данные на 4 переменные
input_train <- mnist$train$x
y_train <- mnist$train$y
input_test <- mnist$test$x
y_test <- mnist$test$y

#####
# Меняем размерность тренировочной выборки
train_images <- array_reshape(input_train, c(60000, 28*28))
# Меняем область значений
train_images <- input_train/255
str(train_images) #Смотрим на изменённую выборку

#####
# Переходим к тренировке рекуррентной сети
model <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_features, output_dim = 32) %>%
  layer_simple_rnn(units = 32) %>%
  layer_dense(units = 1, activation = "sigmoid")
model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("acc")
)

history <- model %>% fit(
  input_train, y_train,
  epochs = 5,
  batch_size = 128,
  validation_split = 0.2
)

# Вывод графика-результата
plot(history)


