# import the necessary packages
import numpy as np
import argparse
import cv2
import numpy as np

# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", help="path to the image file")
# args = vars(ap.parse_args())

# # load the image
# image = cv2.imread(args["image"])

# # find all the 'black' shapes in the image
# lower = np.array([0, 0, 0])
# upper = np.array([15, 15, 15])
# shapeMask = cv2.inRange(image, lower, upper)

# # find the contours in the mask
# (cnts, _, _) = cv2.findContours(shapeMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# print("I found %d black shapes" % (len(cnts)))
# cv2.imshow("Mask", shapeMask)

# # loop over the contours
# for c in cnts:
# 	# draw the contour and show it
#     c = np.array(c).reshape((-1, 1, 2)).astype(np.int32)
#     cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
#     cv2.imshow("Image", image)
#     cv2.waitKey(0)

img = cv2.imread("/home/seba/Desktop/ISIC_0012377.jpg")

# convert to hsv
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# brown
mask1 = cv2.inRange(hsv, (0, 100, 0), (20, 255, 255))
# magenta
mask2 = cv2.inRange(hsv, (390, 100, 0), (310, 255, 255))

## final mask and masked
mask = cv2.bitwise_or(mask1, mask2)
target = cv2.bitwise_and(img, img, mask=mask)

cv2.imwrite("target.png", target)
