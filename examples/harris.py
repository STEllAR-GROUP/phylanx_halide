from phylanx import Phylanx
import cv2
import numpy

@Phylanx
def add(img):
    return harris(img)

img = cv2.imread("rgba.png")
data = numpy.asarray(img)

print(data.shape)

add(data)
