import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import color
from sklearn import svm
from sklearn.externals import joblib
from subprocess import call
cap = cv2.VideoCapture(0)
cv2.namedWindow("ISL",cv2.CV_WINDOW_AUTOSIZE)
cv2.namedWindow("text",cv2.CV_WINDOW_AUTOSIZE)
clf=joblib.load('ZSVM_FULL.pkl')
alpha_text = "."
index = 0.0
blank_image = np.zeros((100,100,3), np.uint8)
def a():
    res=call(["aplay","-q","/home/pi/Sign_samples/alpha_sound/New/A.wav"])
    global alpha_text
    alpha_text="A"
def b():
    res=call(["aplay","-q","/home/pi/Sign_samples/alpha_sound/New/B.wav"])
    global alpha_text
    alpha_text="B"
def c():
    res=call(["aplay","-q","/home/pi/Sign_samples/alpha_sound/New/C.wav"])
    global alpha_text
    alpha_text="C"
def d():
    res=call(["aplay","-q","/home/pi/Sign_samples/alpha_sound/New/D.wav"])
    global alpha_text
    alpha_text="D"
def e():
    res=call(["aplay","-q","/home/pi/Sign_samples/alpha_sound/New/E.wav"])
    global alpha_text
    alpha_text="E"
def f():
    res=call(["aplay","-q","/home/pi/Sign_samples/alpha_sound/New/F.wav"])
    global alpha_text
    alpha_text="F"
def g():
    res=call(["aplay","-q","/home/pi/Sign_samples/alpha_sound/New/G.wav"])
    global alpha_text
    alpha_text="G"
def h():
    res=call(["aplay","-q","/home/pi/Sign_samples/alpha_sound/New/H.wav"])
    global alpha_text
    alpha_text="H"
def i():
    res=call(["aplay","-q","/home/pi/Sign_samples/alpha_sound/New/I.wav"])
    global alpha_text
    alpha_text="I"
def k():
    res=call(["aplay","-q","/home/pi/Sign_samples/alpha_sound/New/K.wav"])
    global alpha_text
    alpha_text="K"
def l():
    res=call(["aplay","-q","/home/pi/Sign_samples/alpha_sound/New/L.wav"])
    global alpha_text
    alpha_text="L"
def m():
    res=call(["aplay","-q","/home/pi/Sign_samples/alpha_sound/New/M.wav"])
    global alpha_text
    alpha_text="M"
def n():
    res=call(["aplay","-q","/home/pi/Sign_samples/alpha_sound/New/N.wav"])
    global alpha_text
    alpha_text="N"
def o():
    res=call(["aplay","-q","/home/pi/Sign_samples/alpha_sound/New/O.wav"])
    global alpha_text
    alpha_text="O"
def p():
    res=call(["aplay","-q","/home/pi/Sign_samples/alpha_sound/New/P.wav"])
    global alpha_text
    alpha_text="P"
def q():
    res=call(["aplay","-q","/home/pi/Sign_samples/alpha_sound/New/Q.wav"])
    global alpha_text
    alpha_text="Q"
def r():
    res=call(["aplay","-q","/home/pi/Sign_samples/alpha_sound/New/R.wav"])
    global alpha_text
    alpha_text="R"
def s():
    res=call(["aplay","-q","/home/pi/Sign_samples/alpha_sound/New/S.wav"])
    global alpha_text
    alpha_text="S"
def t():
    res=call(["aplay","-q","/home/pi/Sign_samples/alpha_sound/New/T.wav"])
    global alpha_text
    alpha_text="T"
def u():
    res=call(["aplay","-q","/home/pi/Sign_samples/alpha_sound/New/U.wav"])
    global alpha_text
    alpha_text="U"
def v():
    res=call(["aplay","-q","/home/pi/Sign_samples/alpha_sound/New/V.wav"])
    global alpha_text
    alpha_text="V"
def w():
    res=call(["aplay","-q","/home/pi/Sign_samples/alpha_sound/New/W.wav"])
    global alpha_text
    alpha_text="W"
def x():
    res=call(["aplay","-q","/home/pi/Sign_samples/alpha_sound/New/X.wav"])
    global alpha_text
    alpha_text="X"
def y():
    res=call(["aplay","-q","/home/pi/Sign_samples/alpha_sound/New/Y.wav"])
    global alpha_text
    alpha_text="Y"
def err():
    global alpha_text
    alpha_text="*****"

sign_alphabets = { 0.0 : err,
                   1.0 : a,
                   2.0 : b,
                   3.0 : c,
                   4.0 : d,
                   5.0 : e,
                   6.0 : f,
                   7.0 : g,
                   8.0 : h,
                   9.0 : i,
                   10.0 : k,
                   11.0 : l,
                   12.0 : m,
                   13.0 : n,
                   14.0 : o,
                   15.0 : p,
                   16.0 : q,
                   17.0 : r,
                   18.0 : s,
                   19.0 : t,
                   20.0 : u,
                   21.0 : v,
                   22.0 : w,
                   23.0 : x,
                   24.0 : y,
}

print('SVM train success\n')
j=1
global alpha_text 
alpha_text = "."
while( cap.isOpened() ) :
    blank_image = np.zeros((100,100,3), np.uint8)
    ret,img = cap.read()
    img1=img[0:200,0:180]
    gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    imageA=color.rgb2gray(gray)
    if j==10:
        j=0
        fA= hog(imageA,orientations=8,pixels_per_cell=(16,16),cells_per_block=(1,1),visualise=False)
        result=clf.predict(fA)
        index=result[0]
        sign_alphabets.get(index,err)()
        
    j=j+1
    cv2.putText(blank_image,alpha_text, (10,80), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,255),5)
    img=cv2.resize(img,(480,320))
    cv2.rectangle(img,(0,0),(153,123),(0,255,0),3)
    cv2.imshow("ISL",img)
    cv2.imshow("text",blank_image)
    k = cv2.waitKey(10)
    if k == 27:
        break
