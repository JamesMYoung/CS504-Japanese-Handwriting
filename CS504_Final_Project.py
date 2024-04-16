#CS504 final project

from datareader import load_hiragana
from datareader import increase_images
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import random
import math
from sklearn.preprocessing import StandardScaler

#shuffles both sets in the same way
def shuffle_data(X, Y):
	#for i in range(6112):
	#	k = random.randrange(6112)
	for i in range(len(X)):
		k = random.randrange(len(X))
		temp = X[i]
		X[i] = X[k]
		X[k] = temp
		
		temp = Y[i]
		Y[i] = Y[k]
		Y[k] = temp

def simple_model():
	model = Sequential()
	
	model.add(Conv2D(32, (5,5), activation='relu', input_shape=(76,72,1)))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	
	return model
		
def complex_model_1():
	model = Sequential()
	
	model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(76,72,1),padding='same'))
	model.add(LeakyReLU(alpha=0.1))
	model.add(MaxPooling2D((2, 2),padding='same'))
	model.add(Dropout(0.25))
	model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
	model.add(LeakyReLU(alpha=0.1))
	model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
	model.add(Dropout(0.25))
	model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
	model.add(LeakyReLU(alpha=0.1))                  
	model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
	model.add(Dropout(0.4))
	model.add(Flatten())
	model.add(Dense(128, activation='linear'))
	model.add(LeakyReLU(alpha=0.1))              
	model.add(Dropout(0.3))    
	model.add(Dense(num_classes, activation='softmax'))
	
	return model

def complex_model_2():
	model = Sequential()
	
	model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(76,72,1),padding='same'))
	model.add(LeakyReLU(alpha=0.1))
	model.add(MaxPooling2D((2, 2),padding='same'))
	model.add(Dropout(0.25))
	model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
	model.add(LeakyReLU(alpha=0.1))
	model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
	model.add(Dropout(0.25))
	model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
	model.add(LeakyReLU(alpha=0.1))                  
	model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
	model.add(Dropout(0.4))
	model.add(Conv2D(256, (3, 3), activation='linear',padding='same'))
	model.add(LeakyReLU(alpha=0.1))   #new               
	model.add(MaxPooling2D(pool_size=(2, 2),padding='same')) #new
	model.add(Dropout(0.4)) #new
	model.add(Flatten())
	model.add(Dense(256, activation='linear'))
	model.add(LeakyReLU(alpha=0.1))              
	model.add(Dropout(0.3))    
	model.add(Dense(num_classes, activation='softmax'))
	
	return model
	
def complex_model_3():
	model = Sequential()
	
	model.add(Conv2D(64, kernel_size=(3, 3),activation='linear',input_shape=(76,72,1),padding='same'))
	model.add(LeakyReLU(alpha=0.1)) 
	model.add(MaxPooling2D((2, 2),padding='same'))
	model.add(Conv2D(128, kernel_size=(3, 3),activation='linear',input_shape=(76,72,1),padding='same'))
	model.add(LeakyReLU(alpha=0.1)) 
	model.add(MaxPooling2D((1, 2),padding='same'))
	model.add(Conv2D(192, kernel_size=(3, 3),activation='linear',input_shape=(76,72,1),padding='same'))
	model.add(LeakyReLU(alpha=0.1)) 
	model.add(MaxPooling2D((2, 1),padding='same'))
	model.add(Conv2D(256, kernel_size=(3, 3),activation='linear',input_shape=(76,72,1),padding='same'))
	model.add(LeakyReLU(alpha=0.1)) 
	model.add(MaxPooling2D((4, 4),padding='same'))
	model.add(Flatten())
	model.add(Dense(256, activation='linear'))
	model.add(LeakyReLU(alpha=0.1)) 
	model.add(Dense(256, activation='linear'))
	model.add(LeakyReLU(alpha=0.1)) 
	model.add(Dense(num_classes, activation='softmax'))
	
	return model
	
#These need to be run to create the
#dataset that is used in this assignment
#create_hiragana()
#increase_images()
#exit()

		
X, Y = load_hiragana()

shuffle_data(X, Y)

#split on 80% - 20%
X_train = X[:math.floor((len(X)) * 0.8)]
Y_train = Y[:math.floor((len(Y)) * 0.8)]

X_test = X[math.floor((len(X)) * 0.8):]
Y_test = Y[math.floor((len(Y)) * 0.8):]


print('Training data shape : ', X_train.shape, Y_train.shape)

print('Testing data shape : ', X_test.shape, Y_test.shape)

classes = np.unique(Y_train)
nClasses = len(classes)

print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)


X_train = X_train.reshape(-1, 76, 72, 1)
X_test = X_test.reshape(-1, 76, 72, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

#normalize input
X_train = X_train / 255.
X_test = X_test / 255.


Y_train_one_hot = to_categorical(Y_train)
Y_test_one_hot = to_categorical(Y_test)

print('Original label:', Y_train[0])
print('After conversion to one-hot:', Y_train_one_hot[0])


import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU


batch_size = 256
epochs = 40
num_classes = 51

#Choose model to use
#model = simple_model()
#model = complex_model_1()
#model = complex_model_2()
model = complex_model_3()


model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

model.summary()


X_valid = X_train[math.floor(len(X) * 0.7):]
X_train = X_train[:math.floor(len(X) * 0.7)]

valid_label = Y_train_one_hot[math.floor(len(Y) * 0.7):]
Y_train_one_hot = Y_train_one_hot[:math.floor(len(Y) * 0.7)]




print(X_valid.shape, X_train.shape, valid_label.shape, Y_train_one_hot.shape)

train = model.fit(X_train, Y_train_one_hot, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(X_valid, valid_label))
#train = model.fit(X_train, Y_train_one_hot, batch_size=batch_size,epochs=epochs,verbose=1)
test_eval = model.evaluate(X_test, Y_test_one_hot, verbose=0)

print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])


accuracy = train.history['acc']
val_accuracy = train.history['val_acc']
loss = train.history['loss']
val_loss = train.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()














