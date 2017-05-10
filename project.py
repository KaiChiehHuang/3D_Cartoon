"""CS231A Project baseline 

"""

import numpy as np
import random as rand
import cv2
import matplotlib.pyplot as plt
import pdb
import sys


"""
This function calculates the normalized feature vector 
for a given point
---------------------------------------------------------
@ points: a point location in the image [x,y]
@ originImg: the gray scale original image 
return: the feature vector of this point
"""
def caclPointNormFeature(point, originImg, windowSize = 5):
    #TODO: check for corner 
    surroundPoints = np.array([originImg[point[0],point[1]]])
    for i in range(-windowSize,windowSize+1):
        for j in range(-windowSize,windowSize+1):
            if (i == 0 and j == 0): continue 
            surrPointValue = originImg[point[0]+i,point[1]+j]
            surroundPoints = np.append(surroundPoints,surrPointValue)
    f_bar = np.mean(surroundPoints)
    f_normFactor = np.linalg.norm(surroundPoints - f_bar)

    return (surroundPoints - f_bar) / f_normFactor


def tooClose(point, point2Feature, threshold=5):
    for k,_ in point2Feature.iteritems():
        dist = abs(k[0]-point[0]) + abs(k[1]-point[1])
        if dist < threshold: return True
    return False
"""
def outOfBound(point, originImg):
    x, y = 
"""

"""
This function constructs the normalized features of each 
point detected by the harris corner algorithm
---------------------------------------------------------
@ points: a list of points in the image
@ originImg: the gray scale original image
@ numFeaturePoints: number of feature points to be extracted
return: a point to feature dict
"""
def extractFeature(points,originImg,numFeaturePoints): 
    numPointsFound = 0
    numPoints, _ = points.shape
    point2Feature = {}
    while numPointsFound < numFeaturePoints:
        rand_idx = rand.randint(0,numPoints-1) 
        point = points[rand_idx,:]
        #if outOfBound(point,originImg): continue
        if tooClose(point, point2Feature): continue
        feature = caclPointNormFeature(point, originImg)
        point2Feature[(point[0],point[1])] = feature
        numPointsFound += 1
    
    return point2Feature

"""
This function finds the corresponding points from 
two points to feature dict and returns a list of 
tuples that specifies the corresponding pairs 
---------------------------------------------------------

"""
def findCorrPairs(pTof_1, pTof_2):
    pairs = []
    for k1,v1 in pTof_1.iteritems():
        bestMatch = [0,0]
        maxCorr = -sys.maxint
        for k2,v2 in pTof_2.iteritems():
            corr = v1.dot(v2)
            if corr > maxCorr:
                bestMatch = k2
                maxCorr = corr
        pairs.append((k1,bestMatch,maxCorr))
    return pairs



"""
This function finds the corner points in a given image
using the harris corner detector
---------------------------------------------------------

"""
def getCornerPoints(grayImg): 
    gray = np.float32(grayImg)
    dst = cv2.cornerHarris(gray,15,7,0.24)
    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)

    # Threshold for an optimal value, it may vary depending on the image.
    ys = []
    xs = []
    row, col = dst.shape 
    #pdb.set_trace()
    maxDst = 0.01*dst.max()
    for y in range(row):
        for x in range(col):
            if dst[y,x] > maxDst:
                ys.append(y)
                xs.append(x)
                
    ys = np.array(ys).reshape(len(ys),1)
    xs = np.array(xs).reshape(len(xs),1)

    return np.c_[ys,xs]


"""
=================TEST========================================
"""
if __name__ == "__main__":

    #filename = 'checkboard.png'
    img1 = cv2.imread("guy1.jpg")
    gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    img2 = cv2.imread("guy2.jpg")
    gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    #for i in range(2):
    cornerPoints1 = getCornerPoints(gray1)
    cornerPoints2 = getCornerPoints(gray2)

    numFeaturePoints = 50
    pTof_1 = extractFeature(cornerPoints1,gray1,numFeaturePoints)
    pTof_2 = extractFeature(cornerPoints2,gray2,numFeaturePoints)

    corrPoints = findCorrPairs(pTof_1, pTof_2)
    corrPoints = sorted(corrPoints, key=lambda pair: pair[2])
    corrPoints = corrPoints[::-1]
        
    #print corners
    imgs = [img1,img2]
    color = ['b','g','r','c','m','y','k','w']
    for i in range(2):
        plt.subplot(2,1,i+1)
        plt.imshow(imgs[0])
        for j in range(8):
            pair = corrPoints[j]
            plt.plot(pair[i][1],pair[i][0],marker='.',ms=10,c=color[j])

    plt.show()
