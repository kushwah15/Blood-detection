
# import the libraries as shown below

from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
# re-size all the images to this
IMAGE_SIZE = [224, 224]

train_path = 'Datasets/train'
# Import the Vgg 16 library as shown below and add preprocessing layer to the front of VGG
# Here we will be using imagenet weights

InceptionV3 = InceptionV3(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
# don't train existing weights
for layer in InceptionV3.layers:
    layer.trainable = False

  # useful for getting number of output classes
folders = glob('Datasets/train/*')
# our layers - you can add more if you want
x = Flatten()(InceptionV3.output)
prediction = Dense(len(folders), activation='softmax')(x)

# create a model object
model = Model(inputs=InceptionV3.input, outputs=prediction)
# tell the model what cost and optimization method to use
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)
# Use the Image Data Generator to import the images from the dataset


train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)
# Make sure you provide the same target size as initialied for the image size
training_set = train_datagen.flow_from_directory('Datasets/train',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')
# fit the model
# Run the cell. It will take some time to execute
r = model.fit_generator(
  training_set,
  epochs=10,
  steps_per_epoch=len(training_set)
)
model.save("model2.h5")

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
model = load_model('model2.h5')
test_image = image.load_img(r"Datasets\train\Normal\img12.png", target_size = (224,224))
test_image = image.img_to_array(test_image)
test_image=test_image/255
test_image = np.expand_dims(test_image,axis=0)
result = model.predict(test_image)
print(result)
