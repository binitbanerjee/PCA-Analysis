imgPath = '{'ImagePath}'
import numpy as np
import matplotlib.image as image
from numpy import linalg as LA
imgsCollection =[]
imgVectorCollection = []
imgVectorTransposeCollection =[]
imgMatrixCorrelationCollection =[]
from PIL import Image

def checkForDuplicates(rowColumnCollection,row,col):
    duplicateFound=False
    if len(rowColumnCollection)!=0:
        for items in rowColumnCollection:
            trow=items[0]
            tcol=items[1]
            if(trow==row and tcol==col):
                duplicateFound=True
                break
    return duplicateFound

def LoadRandomImg(countOfImage,rowDimension,colDimension):
    img = np.float64(image.imread(imgPath))
    sum = np.zeros((rowDimension * rowDimension, rowDimension * rowDimension))
    rowsStart = np.random.choice({'imagedimension-height'} - rowDimension, 1000, replace=False)
    colsStart = np.random.choice({'Imagedimension-col'} - colDimension, 1000, replace=False)
    imageCount = 0
    for index in range(len(rowsStart)):
        imageCount=imageCount+1
        tempImage=[]
        for i in range(rowDimension):
            t=[]
            for j in range (colDimension):
                t.append(img[(i+rowsStart[index]),(j+colsStart[index])][0])
            tempImage.append(t)
        sum = calculateCorrelationMatrix(tempImage,rowDimension,colDimension,sum)
        imgsCollection.append('random_%02d.png' % imageCount)
    return sum

def calculateCorrelationMatrix(img, rowMax,colMax,sum):
    dimensionalVector = []
    for i in range(rowMax):
        for j in range(colMax):
            dimensionalVector.append([int(img[i][j])])
    dimensionalVectorTranspose = np.transpose(dimensionalVector)
    tempMatrix = np.matmul(dimensionalVector, dimensionalVectorTranspose)
    sum = np.add(sum, tempMatrix)
    return sum

def sortEigenValuesNVectorsInDesc(eigenValues,eigenVectors):
    ids = eigenValues.argsort()[::-1]
    eigenValues = eigenValues[ids]
    eigenVectors = eigenVectors[:,ids]
    return eigenValues,eigenVectors



def createCompressedImage(eigenValues,eigenVectors):
    maxLength = 64
    numberOfEigenValues = len(eigenValues)
    if maxLength<=numberOfEigenValues:
        counter=maxLength
    else:
        counter = numberOfEigenValues
    vrow,vcol = eigenVectors.shape
    allImages=[]
    vCount=0
    for i in  range(64):
        temp=[]
        if(vCount>counter):
            break;
        vCount=vCount+1
        for j in range(vrow):
            temp.append(eigenVectors[j,i])

        newImage = np.array(temp).reshape(16,16)
        image.imsave('imageFormed_%02d.png' %i,newImage,cmap='gray')
        allImages.append(newImage)
    y=0
    result = Image.new("RGB", (256, 256))
    lcounter = 0
    for i in range(64):
        if(i<10):
            path='imageFormed_0'+ str(i)+'.png'
        else:
            path='imageFormed_'+str(i)+'.png'
        img=Image.open(path)
        img.thumbnail((18,18),Image.ANTIALIAS)
        if lcounter==0:
            x=0
        else:
            x = lcounter*18
        lcounter=lcounter+1
        w,h = img.size
        result.paste(img, (x, y, x + w, y + h))
        if lcounter==8:
            y=y+18
            lcounter=0
    result.save("ImageOut.jpg")


row = col = {'patch sizes'}
correlationMatrix = LoadRandomImg({'numberofSamples'},row,col)
eigenValues,eigenVectors = LA.eigh(correlationMatrix)
eigenValues,eigenVectors = sortEigenValuesNVectorsInDesc(eigenValues,eigenVectors)
a = eigenValues.sort()
createCompressedImage(eigenValues,eigenVectors)

