from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
import progressbar
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta, Adamax, Nadam
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU 
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
import sys
import matplotlib.pyplot as plt

#####################################
# Loading images into memory
#####################################

image_width = 24
image_height = 24
channels = 3

dirName = "resized_data"

print("creating dataset")
dataset = np.ndarray(shape=(len(os.listdir(dirName)), image_height, image_width, channels),
                     dtype=np.float32)
y_list = [0] * len(os.listdir(dirName))

print("loading images:")
i = 0
for filename in progressbar.progressbar(os.listdir(dirName)):
    if filename.endswith(".jpg"):
        img = load_img(dirName + "\\" + filename)  # this is a PIL image,       remember to change \\ for Windows
        # Convert to Numpy Array
        x = img_to_array(img)
        x = x/255
        # Normalize
        #x = (x - 128.0) / 128.0
        dataset[i] = x
        if filename[0] == 'p':
            y_list[i] = 1 		# 1 corresponds to positive
        else:
            y_list[i] = 0		# 0 corresponds to negative
        i += 1
print("All images to array")

print("splitting:")
X_train, X_test, Y_train, Y_test = train_test_split(dataset, y_list, test_size=0.2, random_state=33)
X_train, X_vizTest, Y_train, Y_vizTest = train_test_split(X_train, Y_train, test_size=0.03, random_state=33)

print("Train set size: %i, Test set size: %i, Final test set size: %i" % (len(X_train), len(X_test), len(X_vizTest)))

data_labels = ["negative", "positive"] # so 0 corresponds to negative and 1 corresponds to positive

# stops execution if we want to test data formatting
#sys.exit("execution halted")

#####################################
# Model Training
#####################################
number_of_classes = 2

Y_train = to_categorical(Y_train, number_of_classes)
Y_test = to_categorical(Y_test, number_of_classes)

# Three steps to Convolution
# 1. Convolution
# 2. Activation
# 3. Polling
# Repeat Steps 1,2,3 for adding more hidden layers

# 4. After that make a fully connected network
# This fully connected network gives ability to the CNN
# to classify the samples

inShape = 24 # dimension of square image for input shape of conv layer

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(inShape,inShape,channels)))
model.add(Activation('relu'))
BatchNormalization(axis=-1)
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

BatchNormalization(axis=-1)
model.add(Conv2D(64,(3, 3)))
model.add(Activation('relu'))
BatchNormalization(axis=-1)
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
#Fully connected layer

BatchNormalization()
model.add(Dense(512))
model.add(Activation('relu'))
BatchNormalization()
model.add(Dropout(0.2)) #remove dropout to reduce underfitting
model.add(Dense(number_of_classes))

model.add(Activation('softmax'))

model.summary()
#plot_model(model, to_file='model.png')

#model compilation
model.compile(loss='categorical_crossentropy', optimizer=Nadam(), metrics=['accuracy'])

#data augmentation
gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,
                         height_shift_range=0.08, zoom_range=0.08)
						 
test_gen = ImageDataGenerator()

train_generator = gen.flow(X_train, Y_train, batch_size=64)
test_generator = test_gen.flow(X_test, Y_test, batch_size=64)

print('training...')
model.fit_generator(train_generator, steps_per_epoch=60000//64, epochs=7, 
                    validation_data=test_generator, validation_steps=10000//64)
					
score = model.evaluate(X_test, Y_test)
print()
print('Test accuracy: ', score[1])

#####################################
# Model evaluation, prediction visualization
#####################################
Y_hat = model.predict(X_vizTest)

# Plot a random sample of 10 test images, their predicted labels and ground truth
figure = plt.figure(figsize=(20, 8))
for i, index in enumerate(np.random.choice(X_vizTest.shape[0], size=15, replace=False)):
    ax = figure.add_subplot(3, 5, i + 1, xticks=[], yticks=[])
    # Display each image
    ax.imshow(np.squeeze(X_vizTest[index]))
    predict_index = np.argmax(Y_hat[index])
    true_index = Y_vizTest[index]
    # Set the title for each image
    ax.set_title("{} ({})".format(data_labels[predict_index], 
                                  data_labels[true_index]),
                                  color=("green" if predict_index == true_index else "red"))
#plt.savefig('model_eval.png')

model.save('cnnModel.h5')



