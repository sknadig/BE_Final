import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import color
from sklearn import svm
from sklearn.externals import joblib

i=1
imgNameA='K'
folderName='/home/pi/Sign_samples/new_data/'
imgName=folderName+imgNameA
hogArr=joblib.load('HOGFAI.pkl')
while(i<1001) :
    fileNameA="%s%d.jpg"%(imgName,i) 
    aImg=cv2.imread(fileNameA)
    imageA=color.rgb2gray(aImg)
    fA = hog(imageA,orientations=8,pixels_per_cell=(16,16),cells_per_block=(1,1),visualise=False)
    size=fA.shape
    if size[0]!=1056 :
       fa=np.resize(fa,1056)
    hogArr=np.vstack((hogArr,fA))
    print(fileNameA)
    i=i+1
joblib.dump(hogArr,'hogFABCDEFGHIK.pkl')

i=1
imgNameA='L'
folderName='/home/pi/Sign_samples/new_data/'
imgName=folderName+imgNameA
hogArr=joblib.load('hogFABCDEFGHIK.pkl')
while(i<1001) :
    fileNameA="%s%d.jpg"%(imgName,i) 
    aImg=cv2.imread(fileNameA)
    imageA=color.rgb2gray(aImg)
    fA = hog(imageA,orientations=8,pixels_per_cell=(16,16),cells_per_block=(1,1),visualise=False)
    size=fA.shape
    if size[0]!=1056 :
       fa=np.resize(fa,1056)
    hogArr=np.vstack((hogArr,fA))
    print(fileNameA)
    i=i+1
joblib.dump(hogArr,'hogFABCDEFGHIKL.pkl')

i=1
imgNameA='M'
folderName='/home/pi/Sign_samples/new_data/'
imgName=folderName+imgNameA
hogArr=joblib.load('hogFABCDEFGHIKL.pkl')
while(i<1001) :
    fileNameA="%s%d.jpg"%(imgName,i) 
    aImg=cv2.imread(fileNameA)
    imageA=color.rgb2gray(aImg)
    fA = hog(imageA,orientations=8,pixels_per_cell=(16,16),cells_per_block=(1,1),visualise=False)
    size=fA.shape
    if size[0]!=1056 :
       fa=np.resize(fa,1056)
    hogArr=np.vstack((hogArr,fA))
    print(fileNameA)
    i=i+1
joblib.dump(hogArr,'hogFABCDEFGHIKLM.pkl')

i=1
imgNameA='N'
folderName='/home/pi/Sign_samples/new_data/'
imgName=folderName+imgNameA
hogArr=joblib.load('hogFABCDEFGHIKLM.pkl')
while(i<1001) :
    fileNameA="%s%d.jpg"%(imgName,i) 
    aImg=cv2.imread(fileNameA)
    imageA=color.rgb2gray(aImg)
    fA = hog(imageA,orientations=8,pixels_per_cell=(16,16),cells_per_block=(1,1),visualise=False)
    size=fA.shape
    if size[0]!=1056 :
       fa=np.resize(fa,1056)
    hogArr=np.vstack((hogArr,fA))
    print(fileNameA)
    i=i+1
joblib.dump(hogArr,'hogFABCDEFGHIKLMN.pkl')

i=1
imgNameA='O'
folderName='/home/pi/Sign_samples/new_data/'
imgName=folderName+imgNameA
hogArr=joblib.load('hogFABCDEFGHIKLMN.pkl')
while(i<1001) :
    fileNameA="%s%d.jpg"%(imgName,i) 
    aImg=cv2.imread(fileNameA)
    imageA=color.rgb2gray(aImg)
    fA = hog(imageA,orientations=8,pixels_per_cell=(16,16),cells_per_block=(1,1),visualise=False)
    size=fA.shape
    if size[0]!=1056 :
       fa=np.resize(fa,1056)
    hogArr=np.vstack((hogArr,fA))
    print(fileNameA)
    i=i+1
joblib.dump(hogArr,'hogFABCDEFGHIKLMNO.pkl')

i=1
imgNameA='P'
folderName='/home/pi/Sign_samples/new_data/'
imgName=folderName+imgNameA
hogArr=joblib.load('hogFABCDEFGHIKLMNO.pkl')
while(i<1001) :
    fileNameA="%s%d.jpg"%(imgName,i) 
    aImg=cv2.imread(fileNameA)
    imageA=color.rgb2gray(aImg)
    fA = hog(imageA,orientations=8,pixels_per_cell=(16,16),cells_per_block=(1,1),visualise=False)
    size=fA.shape
    if size[0]!=1056 :
       fa=np.resize(fa,1056)
    hogArr=np.vstack((hogArr,fA))
    print(fileNameA)
    i=i+1
joblib.dump(hogArr,'hogFABCDEFGHIKLMNOP.pkl')

i=1
imgNameA='Q'
folderName='/home/pi/Sign_samples/new_data/'
imgName=folderName+imgNameA
hogArr=joblib.load('hogFABCDEFGHIKLMNOP.pkl')
while(i<1001) :
    fileNameA="%s%d.jpg"%(imgName,i) 
    aImg=cv2.imread(fileNameA)
    imageA=color.rgb2gray(aImg)
    fA = hog(imageA,orientations=8,pixels_per_cell=(16,16),cells_per_block=(1,1),visualise=False)
    size=fA.shape
    if size[0]!=1056 :
       fa=np.resize(fa,1056)
    hogArr=np.vstack((hogArr,fA))
    print(fileNameA)
    i=i+1
joblib.dump(hogArr,'hogFABCDEFGHIKLMNOPQ.pkl')

i=1
imgNameA='R'
folderName='/home/pi/Sign_samples/new_data/'
imgName=folderName+imgNameA
hogArr=joblib.load('hogFABCDEFGHIKLMNOPQ.pkl')
while(i<1001) :
    fileNameA="%s%d.jpg"%(imgName,i) 
    aImg=cv2.imread(fileNameA)
    imageA=color.rgb2gray(aImg)
    fA = hog(imageA,orientations=8,pixels_per_cell=(16,16),cells_per_block=(1,1),visualise=False)
    size=fA.shape
    if size[0]!=1056 :
       fa=np.resize(fa,1056)
    hogArr=np.vstack((hogArr,fA))
    print(fileNameA)
    i=i+1
joblib.dump(hogArr,'hogFABCDEFGHIKLMNOPQR.pkl')

i=1
imgNameA='S'
folderName='/home/pi/Sign_samples/new_data/'
imgName=folderName+imgNameA
hogArr=joblib.load('hogFABCDEFGHIKLMNOPQR.pkl')
while(i<1001) :
    fileNameA="%s%d.jpg"%(imgName,i) 
    aImg=cv2.imread(fileNameA)
    imageA=color.rgb2gray(aImg)
    fA = hog(imageA,orientations=8,pixels_per_cell=(16,16),cells_per_block=(1,1),visualise=False)
    size=fA.shape
    if size[0]!=1056 :
       fa=np.resize(fa,1056)
    hogArr=np.vstack((hogArr,fA))
    print(fileNameA)
    i=i+1
joblib.dump(hogArr,'hogFABCDEFGHIKLMNOPQRS.pkl')

i=1
imgNameA='T'
folderName='/home/pi/Sign_samples/new_data/'
imgName=folderName+imgNameA
hogArr=joblib.load('hogFABCDEFGHIKLMNOPQRS.pkl')
while(i<1001) :
    fileNameA="%s%d.jpg"%(imgName,i) 
    aImg=cv2.imread(fileNameA)
    imageA=color.rgb2gray(aImg)
    fA = hog(imageA,orientations=8,pixels_per_cell=(16,16),cells_per_block=(1,1),visualise=False)
    size=fA.shape
    if size[0]!=1056 :
       fa=np.resize(fa,1056)
    hogArr=np.vstack((hogArr,fA))
    print(fileNameA)
    i=i+1
joblib.dump(hogArr,'hogFABCDEFGHIKLMNOPQRST.pkl')


i=1
imgNameA='U'
folderName='/home/pi/Sign_samples/new_data/'
imgName=folderName+imgNameA
hogArr=joblib.load('hogFABCDEFGHIKLMNOPQRST.pkl')
while(i<1001) :
    fileNameA="%s%d.jpg"%(imgName,i) 
    aImg=cv2.imread(fileNameA)
    imageA=color.rgb2gray(aImg)
    fA = hog(imageA,orientations=8,pixels_per_cell=(16,16),cells_per_block=(1,1),visualise=False)
    size=fA.shape
    if size[0]!=1056 :
       fa=np.resize(fa,1056)
    hogArr=np.vstack((hogArr,fA))
    print(fileNameA)
    i=i+1
joblib.dump(hogArr,'hogFABCDEFGHIKLMNOPQRSTU.pkl')

i=1
imgNameA='V'
folderName='/home/pi/Sign_samples/new_data/'
imgName=folderName+imgNameA
hogArr=joblib.load('hogFABCDEFGHIKLMNOPQRSTU.pkl')
while(i<1001) :
    fileNameA="%s%d.jpg"%(imgName,i) 
    aImg=cv2.imread(fileNameA)
    imageA=color.rgb2gray(aImg)
    fA = hog(imageA,orientations=8,pixels_per_cell=(16,16),cells_per_block=(1,1),visualise=False)
    size=fA.shape
    if size[0]!=1056 :
       fa=np.resize(fa,1056)
    hogArr=np.vstack((hogArr,fA))
    print(fileNameA)
    i=i+1
joblib.dump(hogArr,'hogFABCDEFGHIKLMNOPQRSTUV.pkl')


i=1
imgNameA='W'
folderName='/home/pi/Sign_samples/new_data/'
imgName=folderName+imgNameA
hogArr=joblib.load('hogFABCDEFGHIKLMNOPQRSTUV.pkl')
while(i<1001) :
    fileNameA="%s%d.jpg"%(imgName,i) 
    aImg=cv2.imread(fileNameA)
    imageA=color.rgb2gray(aImg)
    fA = hog(imageA,orientations=8,pixels_per_cell=(16,16),cells_per_block=(1,1),visualise=False)
    size=fA.shape
    if size[0]!=1056 :
       fa=np.resize(fa,1056)
    hogArr=np.vstack((hogArr,fA))
    print(fileNameA)
    i=i+1
joblib.dump(hogArr,'hogFABCDEFGHIKLMNOPQRSTUVW.pkl')

i=1
imgNameA='X'
folderName='/home/pi/Sign_samples/new_data/'
imgName=folderName+imgNameA
hogArr=joblib.load('hogFABCDEFGHIKLMNOPQRSTUVW.pkl')
while(i<1001) :
    fileNameA="%s%d.jpg"%(imgName,i) 
    aImg=cv2.imread(fileNameA)
    imageA=color.rgb2gray(aImg)
    fA = hog(imageA,orientations=8,pixels_per_cell=(16,16),cells_per_block=(1,1),visualise=False)
    size=fA.shape
    if size[0]!=1056 :
       fa=np.resize(fa,1056)
    hogArr=np.vstack((hogArr,fA))
    print(fileNameA)
    i=i+1
joblib.dump(hogArr,'hogFABCDEFGHIKLMNOPQRSTUVWX.pkl')


i=1
imgNameA='Y'
folderName='/home/pi/Sign_samples/new_data/'
imgName=folderName+imgNameA
hogArr=joblib.load('hogFABCDEFGHIKLMNOPQRSTUVWX.pkl')
while(i<1001) :
    fileNameA="%s%d.jpg"%(imgName,i) 
    aImg=cv2.imread(fileNameA)
    imageA=color.rgb2gray(aImg)
    fA = hog(imageA,orientations=8,pixels_per_cell=(16,16),cells_per_block=(1,1),visualise=False)
    size=fA.shape
    if size[0]!=1056 :
       fa=np.resize(fa,1056)
    hogArr=np.vstack((hogArr,fA))
    print(fileNameA)
    i=i+1
joblib.dump(hogArr,'hogFABCDEFGHIKLMNOPQRSTUVWXY.pkl')





svm_Full=svm.SVC(gamma=0.0001,C=100)
target=[1]*1000+[2]*1000+[3]*1000+[4]*1000+[5]*1000+[6]*1000+[7]*1000+[8]*1000+[9]*1000+[10]*1000+[11]*1000+[12]*1000+[13]*1000+[14]*1000+[15]*1000+[16]*1000+[17]*1000+[18]*1000+[19]*1000+[20]*1000+[21]*1000+[22]*1000+[23]*1000+[24]*1000
svm_Full.fit(hogArr,target)
joblib.dump(svm_Full,'ZSVM_FULL.pkl')
