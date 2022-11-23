import os
import cv2

yourDirectory = '/home/taozhi/datasets/ds/room/'

for filename in os.listdir(yourDirectory):
    if filename.endswith(".jpeg"):
        img = cv2.imread(yourDirectory+filename)
        # cv2.imwrite(yourDirectory+filename, img)