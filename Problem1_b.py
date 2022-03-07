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


def generate_mask(frame):
    cur_frame = np.zeros(frame.shape)
    points = []
    mask = np.zeros(frame.shape)
    mask.fill(255)
    white_paper = np.where(frame > 185)
    cur_frame[white_paper] = 255
    x = np.array(white_paper[0])
    y = np.array(white_paper[1])
    #find the corners of the white paper
    points.append([y[np.where(x == np.amin(x))][0],x[np.where(x == np.amin(x))][0]])
    points.append([y[np.where(y == np.amax(y))][0],x[np.where(y == np.amax(y))][0]])
    points.append([y[np.where(x == np.amax(x))][0],x[np.where(x == np.amax(x))][0]])
    points.append([y[np.where(y == np.amin(y))][0],x[np.where(y == np.amin(y))][0]])
    points = np.array(points, np.int32)
    points = points.reshape((-1,1,2))
    # create a mask
    mask = cv2.fillPoly(mask, pts=[points], color=(0, 0, 0))
    mask = mask.astype('float32')
    kernel = np.ones((25,25),np.uint8)
    dilation = cv2.dilate(mask,kernel,iterations = 1)
    # adding the dilated mask to the image
    masked_img = dilation + cur_frame
    masked_img = masked_img.astype('float32')
    ret, thresh1 = cv2.threshold(masked_img, 180, 255, cv2.THRESH_BINARY)   
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
        id = int(np.round(id))
        prev_orientation = orientation
        prev_id = id
    
    return orientation, id

def main():
    video_file = '1tagvideo.mp4'
    ref_tag_image = 'refTag.png'
    ref_tag = cv2.imread(ref_tag_image)
    resized_tag = cv2.resize(ref_tag, (128,128), interpolation = cv2.INTER_AREA)
    rot, id = decode_tag(resized_tag)
    print('The orientation of the reference tag is ', rot)
    print('The ID of the reference tag is ', id)
    print('Now decoding tag in the video frames')
    count = 1
    pts = [[128,0],[128,128],[0,128],[0,0]] # points in (y,x) form of the destination image
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video1 = cv2.VideoWriter('AR_Tag_corner.mp4',fourcc, 20, (1920,1080))
    video2 = cv2.VideoWriter('AR_Tag_warped.mp4',fourcc, 25, (128,128), 0)
    cap = cv2.VideoCapture(video_file)
    if (cap.isOpened()== False):
        print("Error opening video stream or file")
    while True:
        ret, frame = cap.read()
        if ret == True:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.blur(img,(21,21))
            c_points, m = generate_mask(blur)
            h_matrix = Homography(c_points, pts)
            tag_from_img = Warp(frame, h_matrix, (128,128))
            cv2.namedWindow("video_frame", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("video_frame", 700, 700)
            angle, ID = decode_tag(tag_from_img)
            # print('********** For frame {} **********'.format(count))
            # print('The orientation of the tag is ', angle)
            str1 = 'The orientation of the tag is ' + str(angle)
            # print('The ID of the tag is ', ID)
            str2 = 'The ID of the tag is ' + str(ID)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, str1 , (50, 50), font, 2, (0, 255, 0), 2, cv2.LINE_4)
            cv2.putText(frame, str2 , (150, 150), font, 2, (0, 255, 0), 2, cv2.LINE_4)
            count = count+1
            for i in c_points:
                cv2.circle(frame, i, 10, [0,0,255],-1)
            video1.write(frame)
            video2.write(tag_from_img)
            cv2.imshow('video_frame', frame)
            if cv2.waitKey(1) & 0xFF ==ord('q'):
                break
        else:
            break
    print('The video of detected and decoded AR tag is saved in the folder')
    video1.release()
    video2.release()
    cap.release()

if __name__ == '__main__':
    main()

