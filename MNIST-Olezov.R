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

# Загружаем данные
mnist <- dataset_mnist()
# Разбиваем исходные данные на 4 переменные
train_images <- mnist$train$x
train_labels <- mnist$train$y
test_images <- mnist$test$x
test_labels <- mnist$test$y

# Строим архитектуру нейронной сети
network <- keras_model_sequential() %>%
  layer_dense(units = 512, activation = 'relu', input_shape = c(28*28)) %>%
  layer_dense(units = 10, activation = 'softmax')
# Добавляем для нейронной сети оптимизатор, функцию потерь и метрики для вывода на экран
network %>% compile(
  optimizer = 'rmsprop',
  loss = 'categorical_crossentropy',
  metrics = c('accuracy'))
#####
# Изначально массивы имеют размерность 60000, 28, 28, сами значения изменяются в пределах от 0 до 255
# Для обучения нейронной сети необходимо преобразовать форму 60000, 28*28, а значения перевести в размерность от 0 до 1

# Меняем размерность тренировочной выборки:
train_images <- array_reshape(train_images, c(60000, 28*28))
# Меняем область значений:
train_images <- train_images/255
str(train_images)

# Меняем размерность тестовой выборки:
test_images <- array_reshape(test_images, c(10000, 28*28))
# Меняем область значений:
test_images <- test_images/255

# Cоздаём категории для ярлыков
train_labels <- to_categorical(train_labels)
test_labels <- to_categorical(test_labels)


# Тренируем нейронную сеть с условием, что число эпох >=10
network %>% fit(train_images, train_labels, epochs = 10, batch_size = 128)
# Точность модели составила 99,3%

# Аналогично по тестовой выборке
metric <- network %>% evaluate(test_images, test_labels)
metric
# Точность модели по тестоовой выборке составила 98,1%

##### Предсказываем наблюдения
# Для этого производим выборку по тестовым наблюдениям (первые 10; последние 10)
a_test <- rbind (test_images[1:10,], test_images[9991:10000,])
predict <- network %>% predict_classes(a_test)
test_labels1 <- mnist$test$y
a_test_lables <- rbind (test_labels1[1:10], test_labels1[9991:10000])        
trans <- t(a_test_lables)
# Преобразуем в числовой вектор
data <- as.numeric(t) 

# Сравним полученные и расчётные данные с помощью функции compare
compare <- cbind(data, predict)
compare

# Произведём обучение сетей с добавлением валидации (появятся интерактивные графики с точками каждой эпохи)
history <- network %>% fit(train_images, train_labels,
                           epochs = 10, batch_size = 128,
                           validation_split = 0.2)

#####
# Теперь отобразим числа из массива (представлены в последних 10 матрицах - 9991:10000)
par(mfrow = c(3, 3))
#1
a1 <- mnist$test$x[9991, 1:28, 1:28]
a1
image(as.matrix(a1))
#2
a2 <- mnist$test$x[9992, 1:28, 1:28]
image(as.matrix(a2))
#3
a3 <- mnist$test$x[9993, 1:28, 1:28]
image(as.matrix(a3))
#4
a4 <- mnist$test$x[9994, 1:28, 1:28]
image(as.matrix(a4))
#5
a5 <- mnist$test$x[9995, 1:28, 1:28]
image(as.matrix(a5))
#6
a6 <- mnist$test$x[9996, 1:28, 1:28]
image(as.matrix(a6))
#7
a7 <- mnist$test$x[9997, 1:28, 1:28]
image(as.matrix(a7))
#8
a8 <- mnist$test$x[9998, 1:28, 1:28]
image(as.matrix(a8))
#9
a9 <- mnist$test$x[9999, 1:28, 1:28]
image(as.matrix(a9))
par(mfrow = c(1, 1))
#10 (изгой)
a10 <- mnist$test$x[10000, 1:28, 1:28]
image(as.matrix(a10))  
