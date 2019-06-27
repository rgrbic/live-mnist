"""
use the MNIST set + model to predict on the real world data:
take an image and do adaptive tresholding and contours to find the digits
then apply a CNN trained on MNIST for each rectangle found:

great resource + credits for contours and adaptive tresholds:
http://hanzratech.in/2015/02/24/handwritten-digit-recognition-using-opencv-sklearn-and-python.html

author: Alexandru Papiu, December, 2016, alex.papiu@gmail.com, @apapiu
Jacob Rafati fixed the bug and trained the MNIST model again and correctly.

"""

import sys
import cv2
import numpy as np
from keras.models import load_model
from keras import backend as K
from sklearn.preprocessing import LabelEncoder
from subprocess import call

font = cv2.FONT_HERSHEY_SIMPLEX

cp = cv2.VideoCapture(0)
cp.set(3, 5*128)
cp.set(4, 5*128)
SIZE = 28
img_rows, img_cols = 28, 28

kernel = np.ones((5,5), np.uint8) 

if K.image_data_format() == 'channels_first':
    input_shape = (1, img_rows, img_cols)
    first_dim = 0
    second_dim = 1
else:
    input_shape = (img_rows, img_cols, 1)
    first_dim = 0
    second_dim = 3


def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
 
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
 
	# return the edged image
	return edged

def annotate(frame, label, location = (20,30)):
    #writes label on image#

    cv2.putText(frame, label, location, font,
                fontScale = 0.5,
                color = (255, 255, 0),
                thickness =  1,
                lineType =  cv2.LINE_AA)


def extract_digit(frame, rect, pad = 10):
    x, y, w, h = rect
    cropped_digit = final_img[y-pad:y+h+pad, x-pad:x+w+pad]
    cropped_digit = cropped_digit/255.0

    #only look at images that are somewhat big:
    if cropped_digit.shape[0] >= 32 and cropped_digit.shape[1] >= 32:
        cropped_digit = cv2.resize(cropped_digit, (SIZE, SIZE))
    else:
        return
    return cropped_digit


def img_to_mnist(frame, tresh = 70):
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

    #adaptive here does better with variable lighting:
    gray_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, blockSize = 321, C = 28)

    return gray_img

def img_to_mnist_dilate(frame):
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    edge_img = auto_canny(gray_img)
    img_dilation = cv2.dilate(edge_img, kernel, iterations=1)

    return img_dilation


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print("loading model")
model = load_model("full_model.mnist")

labelz = dict(enumerate(["zero", "one", "two", "three", "four",
                         "five", "six", "seven", "eight", "nine"]))


while True:
    ret, frame = cp.read(0)

    final_img = img_to_mnist_dilate(frame)
    image_shown = frame
    _, contours, _ = cv2.findContours(final_img.copy(), cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)

    rects = [cv2.boundingRect(contour) for contour in contours]
    rects = [rect for rect in rects if rect[2] >= 3 and rect[3] >= 8]

    #draw rectangles and predict:
    for rect in rects:

        x, y, w, h = rect
        
        mnist_frame = extract_digit(frame, rect, pad = 15)
        
        if mnist_frame is not None: #and i % 25 == 0:
            mnist_frame = np.expand_dims(mnist_frame, first_dim) #needed for keras
            mnist_frame = np.expand_dims(mnist_frame, second_dim) #needed for keras
            
            class_prediction = model.predict_classes(mnist_frame, verbose = False)[0]
            prediction = np.around(np.max(model.predict(mnist_frame, verbose = False)), 2)
            label = str(prediction) # if you want probabilities
            
            cv2.rectangle(image_shown, (x - 15, y - 15), (x + 15 + w, y + 15 + h), color = (255, 255, 0))
            
            labelClass = labelz[class_prediction]
            
            annotate(image_shown, labelClass, location = (rect[0], rect[1]))

    cv2.imshow('frame', image_shown)
    cv2.imshow('thresholded', final_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break