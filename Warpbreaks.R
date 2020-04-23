install.packages("kohonen")
library('kohonen')
data('warpbreaks')
str(warpbreaks)
head(warpbreaks)
View(warpbreaks)
set.seed(1)
som.warpbreaks <- som(scale(warpbreaks), grid = somgrid(5, 5, 'hexagonal'))
as.numeric(warpbreaks)
as.numeric(wool)
as.numeric(warpbreaks$wool)
as.numeric(warpbreaks$tension)
som.warpbreaks <- som(scale(warpbreaks), grid = somgrid(5, 5, 'hexagonal'))
warpbreaks$wool<- as.numeric(warpbreaks$wool)
warpbreaks$tension<- as.numeric(warpbreaks$tension)#РџСЂРµРѕР±СЂР°Р·РѕРІР°Р»Рё С„Р°РєС‚РѕСЂС‹ РІ С‡РёСЃР»Р°
som.warpbreaks <- som(scale(warpbreaks), grid = somgrid(5, 5, 'hexagonal'))
som.warpbreaks
dim(getCodes(som.warpbreaks))
plot(som.warpbreaks, main = 'Warpbreaks data Kohonen SOM')
graphics.off()#РЎС‚С‘СЂР»Рё РіСЂР°С„РёРє
par(mfrow = c(1,1))
plot(som.warpbreaks, type = 'changes', main = 'Warpbreaks data SOM')
train <- sample(nrow(warpbreaks), 40)
X_train <- scale(warpbreaks[train,])
X_test <- scale(warpbreaks[-train,],
                center = attr(X_train, "scaled:center"),
                scale = attr(X_train, "scaled:center"))
level <- factor(warpbreaks[1])
level
train_data <- list(measurements = X_train,
                   level = level[train])
test_data <- list(measurements = X_test,
                  level = level[-train])
mygrid <- somgrid(5, 5, 'hexagonal')
som.warpbreaks <- supersom(train_data, grid = mygrid)     
som.warpbreaks
mygrid <- somgrid(4, 4, 'hexagonal')
som.warpbreaks <- supersom(train_data, grid = mygrid)     
mygrid <- somgrid(3, 3, 'hexagonal')
som.warpbreaks <- supersom(train_data, grid = mygrid)
mygrid
mygrid <- somgrid(5, 5, 'hexagonal')
som.warpbreaks <- supersom(train_data, grid = mygrid)
som.predict <- predict(som.warpbreaks, newdata = test_data)
table(level[-train], som.predict$predictions[['level']])
map(som.warpbreaks)
plot(som.warpbreaks, main = 'Warpbreaks data Kohonen SOM')
train_data
level
