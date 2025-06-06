import numpy as np
import cv2

therm = cv2.imread(r"D:\git\learning-cv\camera fusion\thermal.jpg")

ir = cv2.imread(r"D:\git\learning-cv\camera fusion\infrared.jpg")

therm = cv2.resize(therm, (480, 480))
ir = cv2.resize(ir, (480, 480))

cv2.imshow('thermal', therm)