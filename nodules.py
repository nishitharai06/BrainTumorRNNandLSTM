import cv2
import numpy as np
img_path = "a.jpg"
imgfile=img_path
'''
img_path = "a.jpg"
image = cv2.imread(img_path)
print("width: {} pixels".format(image.shape[1]))
print("height: {} pixels".format(image.shape[0]))
print("channels: {}".format(image.shape[2]))
dim=(500,590)
image=cv2.resize(image, dim)
cv2.imshow("OP",image)
cv2.waitKey(1)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY, 0.7)
#cv2_imshow(gray)
(T, thresh) = cv2.threshold(gray, 155, 255, cv2.THRESH_BINARY)
#cv2_imshow(thresh)
(T, threshInv) = cv2.threshold(gray, 155, 255,cv2.THRESH_BINARY_INV)
#cv2.imshow(threshInv)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
#cv2_imshow(closed)
cv2.imshow("OP",closed)
cv2.waitKey(1)

closed = cv2.erode(closed, None, iterations = 14)
closed = cv2.dilate(closed, None, iterations = 13)

edged = cv2.Canny(image, lower, upper)
canny = auto_canny(closed)
cv2.imshow("OP",canny)
cv2.waitKey(1)
cv2.imshow("OP",edged)
cv2.waitKey(1)
'''







from PIL import Image

img = cv2.imread(img_path)
img=cv2.resize(img, (300, 300))
y=50
x=165
h=130
w=100
crop_img = img[y:y+h, x:x+w]

crop_img1=cv2.resize(crop_img, (300, 300))
cv2.imshow("Right Side Image", crop_img1)

cv2.waitKey(0)

#color_img = cv2.cvtColor(crop_img,cv2.COLOR_BGR2RGB)

#cv2.imshow("cropped", color_img)

#cv2.waitKey(0)


gray = cv2.cvtColor(crop_img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# noise removal
kernel = np.ones((2,2),np.uint8)

closing = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel, iterations = 2)

# sure background area
sure_bg = cv2.dilate(closing,kernel,iterations=3)
sure_bg=cv2.resize(sure_bg, (300, 300))
cv2.imshow("Right Side Image Crop", sure_bg)


#cv2.imshow("cropped", sure_bg)

cv2.waitKey(0)





img = cv2.imread(imgfile)
img=cv2.resize(img, (300, 300))
y=50
x=60
h=130
w=100
crop_img = img[y:y+h, x:x+w]
cv2.imshow("cropped", crop_img)

crop_img1=cv2.resize(crop_img, (300, 300))
cv2.imshow("Left Side Image", crop_img1)
cv2.waitKey(0)

#color_img = cv2.cvtColor(crop_img,cv2.COLOR_BGR2RGB)

#cv2.imshow("cropped", color_img)

#cv2.waitKey(0)


gray = cv2.cvtColor(crop_img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# noise removal
kernel = np.ones((2,2),np.uint8)

closing = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel, iterations = 2)

# sure background area
sure_bg = cv2.dilate(closing,kernel,iterations=3)

sure_bg=cv2.resize(sure_bg, (300, 300))
cv2.imshow("Left Side Image Crop", sure_bg)

cv2.waitKey(0)




