from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D , Flatten ,Dense
import cv2
#Start Model 
model = Sequential()
#frist ConV laye
model.add(Conv2D(32,(3,3), input_shape=(50,50,3) , activation = 'relu'))
#Pooling Layer
model.add(MaxPool2D(pool_size = (2,2)))
#second ConV layer
model.add(Conv2D(30,(3,3),activation ='relu'))
model.add(Flatten())

model.add(Dense(units =128, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2 , zoom_range=0.2 , horizontal_flip=True)
test_data = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('train/',(50,50),batch_size=32,class_mode='binary')
model.fit_generator(training_set , steps_per_epoch = 8000,epochs = 25 , validation_steps =2000)
img = cv2.imread("/test/1.jpg")
img =cv2.resize(img,50,50)
img = img.reshape(1,52,50,3)
print(model.predict(img))

