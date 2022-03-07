#!/usr/bin/env python
# coding: utf-8
import cv2
import os
import glob
from cv2 import sort
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate    
from scipy.spatial import distance as dist  

orientation_count = 0
prev_orientation = 0
prev_id = 0

def order_points(pts):
    xSorted = pts[np.argsort(pts[:, 0]), :]
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]
    sorted = [tl,tr,br,bl]
    return sorted

def generate_mask(frame):
    cur_frame = np.zeros(frame.shape)
    points = []
    mask = np.zeros(frame.shape)
    mask.fill(255)
    white_paper = np.where(frame > 185)
    cur_frame[white_paper] = 255
    x = np.array(white_paper[0])
    y = np.array(white_paper[1])
    # print(np.amin(x), np.amax(y), np.amax(x), np.amin(y))
    points.append([y[np.where(x == np.amin(x))][0],x[np.where(x == np.amin(x))][0]])
    points.append([y[np.where(y == np.amax(y))][0],x[np.where(y == np.amax(y))][0]])
    points.append([y[np.where(x == np.amax(x))][0],x[np.where(x == np.amax(x))][0]])
    points.append([y[np.where(y == np.amin(y))][0],x[np.where(y == np.amin(y))][0]])
    points = np.array(points, np.int32)
    points = points.reshape((-1,1,2))
    mask = cv2.fillPoly(mask, pts=[points], color=(0, 0, 0))
    mask = mask.astype('float32')
    kernel = np.ones((25,25),np.uint8)
    dilation = cv2.dilate(mask,kernel,iterations = 1)
    mask = dilation + cur_frame
    mask = mask.astype('float32')
    ret, thresh1 = cv2.threshold(mask, 180, 255, cv2.THRESH_BINARY)   
    median = cv2.medianBlur(thresh1, 5)
    median.astype(np.uint8)
    ar_tag = np.where(median == 0)
    a = np.array(ar_tag[0])
    b = np.array(ar_tag[1])
    corner_points = []
    corner_points.append([b[np.where(a == np.amin(a))][0],a[np.where(a == np.amin(a))][0]])
    corner_points.append([b[np.where(b == np.amax(b))][0],a[np.where(b == np.amax(b))][0]])
    corner_points.append([b[np.where(a == np.amax(a))][0],a[np.where(a == np.amax(a))][0]])
    corner_points.append([b[np.where(b == np.amin(b))][0],a[np.where(b == np.amin(b))][0]])
    return corner_points, median

def Homography( pts1, pts2):
    s = 1
    for i in range(4):
        x, y, xr, yr = pts1[i][0], pts1[i][1], pts2[i][0], pts2[i][1]
        if (s == 1) :
            A = np.array([[-x,-y,-1,0,0,0, x*xr, y*xr,xr], [0,0,0,-x,-y,-1, x*yr, y*yr, yr]])
        else:
            tmp = np.array([[-x,-y,-1,0,0,0, x*xr, y*xr,xr], [0,0,0,-x,-y,-1, x*yr, y*yr, yr]])
            A = np.vstack((A, tmp))

        s+=1
    U,S,Vt = np.linalg.svd(A.astype(np.float32))

    H = Vt[8,:]/Vt[8][8]
    H = H.reshape(3,3)
    
    return H

def Warp(image, H, size):
    u = size[0]
    v = size[1]
    indY, indX = np.indices((u,v))
    H_inv = np.linalg.inv(H)
    destination_pts = np.stack((indX.ravel(), indY.ravel(), np.ones(indX.size)))
    source_pts = H_inv.dot(destination_pts)
    source_pts /= source_pts[2,:]
    source_x, source_y = source_pts[:2,:].astype(int)
    image_transformed = np.zeros((u, v, 3))
    image_transformed[indY.ravel(), indX.ravel(), :] = image[source_y, source_x, :]
    tag = cv2.cvtColor(np.uint8(image_transformed), cv2.COLOR_BGR2GRAY)
    ret,tag = cv2.threshold(np.uint8(tag), 200 ,255,cv2.THRESH_BINARY)
    return tag

def decode_tag(ar_tag):
    global orientation_count
    global prev_orientation
    global prev_id
    ar_tag = ar_tag.astype(np.uint8)
    corner1 = np.median(ar_tag[32:48, 32:48])
    corner2 = np.median(ar_tag[32:48, 80:96])
    corner3 = np.median(ar_tag[80:96, 80:96])
    corner4 = np.median(ar_tag[80:96, 32:48])
    corner = [corner1, corner2, corner3, corner4]
    if(corner.count(255) >1 and orientation_count!=0):
        orientation = prev_orientation
        id = prev_id
    else:
    
        if (corner3 == 255 and (corner)):
            orientation = 0
        elif(corner2 == 255):
            orientation = -90
        elif(corner4 == 255):
            orientation = 90
        else:
            orientation = 180
        inner1 = np.median(ar_tag[48:64, 48:64])//255
        inner2 = np.median(ar_tag[48:64, 64:80])//255
        inner3 = np.median(ar_tag[64:80, 64:80])//255
        inner4 = np.median(ar_tag[64:80, 48:64])//255
        inner = np.array(([inner1],[inner2], [inner4], [inner3])).reshape(2,2)
        inner = rotate(inner, angle = orientation)
        id = inner[1][0]*8  + inner[1][1]*4 + inner[0][1]*2 + inner[0][0]
        prev_orientation = orientation
        prev_id = id
    return orientation, id

def warpTestudo(frame, template, H):
    dest = frame.copy()
    u = template.shape[0]
    v = template.shape[1]
    indY, indX = np.indices((u,v))
    H_inv = np.linalg.inv(H)
    destination_pts = np.stack((indX.ravel(), indY.ravel(), np.ones(indX.size)))
    source_pts = H_inv.dot(destination_pts)
    source_pts /= source_pts[2,:]
    source_x, source_y = np.round(source_pts[:2,:]).astype(int)
    dest[source_y, source_x, :] = template[indY.ravel(), indX.ravel(), :] 

    return dest

def main():
    video_file = '1tagvideo.mp4'
    testudo = cv2.imread('testudo.png')
    resized_testudo = cv2.resize(testudo, (320,320), interpolation = cv2.INTER_AREA)
    # pts = [[128,0],[128,128],[0,128],[0,0]] # points in (y,x) form of the destination image
    pts = [[320,0],[320,320],[0,320],[0,0]]
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video1 = cv2.VideoWriter('Testudo.mp4',fourcc, 25, (1920,1080))
    cap = cv2.VideoCapture(video_file)
    if (cap.isOpened()== False):
        print("Error opening video stream or file")
    while True:
        ret, frame = cap.read()
        if ret == True:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.blur(img,(21,21))
            c_points, m = generate_mask(blur)
            c_pts = np.array(c_points)
            sorted_c_points = order_points(c_pts)
            h_matrix = Homography(sorted_c_points, pts)
            tag_from_img = Warp(frame, h_matrix, (128,128))
            cv2.namedWindow("video_frame", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("video_frame", 700, 700)
            tag_orientation, tag_id = decode_tag(tag_from_img)
            if (tag_orientation == 90):
                testudo_points = [pts[1], pts[2], pts[3], pts[0]]
            elif(tag_orientation == -90):
                testudo_points = [pts[3], pts[0], pts[1], pts[2]]
            else:
                testudo_points = [pts[2], pts[3], pts[0], pts[1]]
            H_testudo = Homography(c_points, testudo_points)
            final = warpTestudo(frame, resized_testudo, H_testudo)
            video1.write(final)
            cv2.imshow('video_frame', final)
            if cv2.waitKey(1) & 0xFF ==ord('q'):
                break
        else:
            break
    
    video1.release()
    cap.release()

if __name__ == '__main__':
    main()


