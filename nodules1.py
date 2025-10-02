import cv2
import numpy as np


img_path = "a.jpg"
image = cv2.imread(img_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY, 0.7)

cv2.imshow("Gray", gray)
cv2.waitKey(0)

(T, thresh) = cv2.threshold(gray, 155, 255, cv2.THRESH_BINARY)
cv2.imshow("thresh", thresh)
cv2.waitKey(0)

(T, threshInv) = cv2.threshold(gray, 155, 255,cv2.THRESH_BINARY_INV)
cv2.imshow("thresh", threshInv)
cv2.waitKey(0)


kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
cv2.imshow("closed", closed)
cv2.waitKey(0)

closed = cv2.erode(closed, None, iterations = 14)
cv2.imshow("closed1", closed)
cv2.waitKey(0)
closed = cv2.dilate(closed, None, iterations = 13)
cv2.imshow("closed1", closed)
cv2.waitKey(0)

t_lower = 50  # Lower Threshold 
t_upper = 150  # Upper threshold 
  
# Applying the Canny Edge filter 
canny = cv2.Canny(image, t_lower, t_upper)

#(cnts, _) = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts, hierarchy = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
cv2.drawContours(image, cnts, -1, (0, 0, 255), 2)
cv2.imshow("closed1", image)
cv2.waitKey(0)
