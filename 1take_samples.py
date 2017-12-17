import cv2
import numpy as np
import time
cap = cv2.VideoCapture(0)
#cap.set(3,320)
#cap.set(4,240)
cv2.namedWindow("input",1)
cv2.namedWindow("hand",2)
i=1
imgName='F'
imgName=input("Enter image name\n")
i=input("Enter starting number\n")
j=1000
j=input("Enter ending number\n")
folder='/home/pi/Sign_samples/new_data/'
imgName1=folder+imgName
while( cap.isOpened() ) :
    ret,img = cap.read()
    img1=img[0:200,0:180]
    gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    cv2.rectangle(img,(0,0),(200,180),(0,255,0),3)
    cv2.imshow("input",img)
    cv2.imshow("hand",gray)
    fileName="%s%d.jpg"%(imgName1,i)
    cv2.imwrite(fileName,gray)
    print(i)
    i=i+1
    k = cv2.waitKey(10)
    if k == 27:
        break
    if i == j+1:
        break
