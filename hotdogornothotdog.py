
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image

from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions
import numpy as np

model = VGG16(weights='imagenet', include_top=True)

img_path = 'coffee.JPG'
def predict(file):
	img = image.load_img(file, target_size=(224, 224))
	img = image.img_to_array(img)
	img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
	img = preprocess_input(img)
	yhat = model.predict(img)
	label = decode_predictions(yhat)
	label = label[0][0]
	result = '%s' % (label[1])
	if result == "hotdog":
		print(result + "!")
	else:
		print("not hotdog!")

predict('coffee.JPG')
predict('hotdog.jpg')