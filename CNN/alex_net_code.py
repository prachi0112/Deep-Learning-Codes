import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
# from keras.datasets import cifar10

# x,y = mnist.load_data(one_hot = True)
# (x_train,y_train),(x_test,y_test) = cifar10.load_data()

model = Sequential()

# 1st conv layer
model.add(Conv2D(filters = 96, input_shape = (227,227,3), kernel_size=(11,11),
                 strides = (4,4), padding='valid', activation = 'relu'))
# max pooling
model.add(MaxPooling2D(pool_size=(2,2), strides = (2,2), padding='valid'))

# 2nd conv layer
model.add(Conv2D(filters = 256, kernel_size=(11,11),
                 strides = (1,1), padding='valid', activation = 'relu'))
# max pooling
model.add(MaxPooling2D(pool_size=(2,2), strides = (2,2), padding='valid'))

# 3rd conv layer
model.add(Conv2D(filters = 384, kernel_size=(3,3),
                 strides = (1,1), padding='valid', activation = 'relu'))

# 4th conv layer
model.add(Conv2D(filters = 384, kernel_size=(3,3),
                 strides = (1,1), padding='valid', activation = 'relu'))
# 5th conv layer
model.add(Conv2D(filters = 256, kernel_size=(3,3),
                 strides = (1,1), padding='valid', activation = 'relu'))

# max pooling
model.add(MaxPooling2D(pool_size=(2,2), strides = (2,2), padding='valid'))


# flatten layer
model.add(Flatten())

#1st fullly connected layer
model.add(Dense(4096, input_shape=(227,227,3)))
model.add(Activation('relu'))
# add dropout layer to prevent overfitting
model.add(Dropout(0.4))

# 2nd fully connected layer
model.add(Dense(4096))
model.add(Activation('relu'))
model.add(Dropout(0.4))

# 3rd fully connected layer
model.add(Dense(1000))
model.add(Activation('relu'))
model.add(Dropout(0.4))

# output layer
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam',
              metrics = ['accuracy'])

# model.fit(x_train,y_train,batch_size=64, epochs=5,verbose=1,validation_split = 0.2,
#           shuffle = True)


from keras.preprocessing.image import ImageDataGenerator

train = ImageDataGenerator(rescale=1/255., horizontal_flip=True,
                          zoom_range=0.2)
test = ImageDataGenerator(rescale=1/255.)
training_set = train.flow_from_directory('CNN/dog_cat/training_set/',
                                        target_size=(227,227),
                                        batch_size=32, class_mode='binary')

test_set = test.flow_from_directory('CNN/dog_cat/test_set/',
                                    target_size=(227,227),
                                   batch_size=32, class_mode='binary')

model.fit_generator(training_set,epochs=10,validation_data=test_set,
                        steps_per_epoch=20)

model.save('model.h5')















