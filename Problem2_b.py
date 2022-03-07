#!/usr/bin/env python
# coding: utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate

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


def projection_matrix(corners, ref_corners, intrinsic):
    h = Homography(ref_corners,corners)
    B_prime = np.dot(np.linalg.inv(intrinsic), h)
    determinant  = np.linalg.det(B_prime)
    if (determinant < 0):
        B_prime = B_prime*(-1)
    
    b1 = B_prime[:,0]
    b2 = B_prime[:,1]
    b3 = B_prime[:,2]
    norm = (np.linalg.norm(b1) + np.linalg.norm(b2))/2
    lambda_p = 1 / norm
    r1 = lambda_p*b1
    r2 = lambda_p*b2
    r3 = np.cross(r1,r2)
    t = lambda_p*b3
    P = np.dot(intrinsic, np.array([r1,r2, r3, t]).T)
    return P, h

def draw_cube(P, H):
    top_points_image = []
    top_points_world = np.array([[1,0,-1, 1], [1,1,-1,1],[0,1,-1,1], [0,0,-1,1]]).T
    top_points = np.dot(P, top_points_world)
    top_points /= top_points[2,:]
    top_y, top_x = top_points[:2,:].astype(int)
    for i in range(4):
        top_points_image.append([top_y[i], top_x[i]])
    return top_points_image



def main():
    video_file = '1tagvideo.mp4'
    pts = [[128,0],[128,128],[0,128],[0,0]] # points in (y,x) form of the destination image
    cube_bot_pt = [[1,0],[1,1],[0,1],[0,0]]
    K = np.array([[1346.100595, 0, 932.1633975], [0, 1355.933136, 654.8986796], [0, 0, 1]])
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video1 = cv2.VideoWriter('Cube.mp4',fourcc, 25, (1920,1080))
    cap = cv2.VideoCapture(video_file)
    if (cap.isOpened()== False):
        print("Error opening video stream or file")
    while True:
        ret, frame = cap.read()
        if ret == True:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.blur(img,(21,21))
            c_points, m = generate_mask(blur)
            cv2.namedWindow("video_frame", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("video_frame", 700, 700)
            c_points, m = generate_mask(blur)
            P_matrix, H_cube = projection_matrix(c_points, cube_bot_pt, K)
            cube_top_pts = draw_cube(P_matrix, H_cube)
            for i in range(4):
                cv2.line(frame, (c_points[i][0], c_points[i][1]), (cube_top_pts[i][0], cube_top_pts[i][1]), (0, 0, 255), thickness=3, lineType=8)

            points = np.array(c_points, np.int32)
            points = points.reshape((-1,4,2))
            t_points = np.array(cube_top_pts, np.int32)
            t_points = t_points.reshape((-1,4,2))
            frame = cv2.polylines(frame, points, True, (0,0,255),3)
            frame = cv2.polylines(frame, t_points, True, (0,0,255),3)
            video1.write(frame)
            cv2.imshow('video_frame', frame)
            if cv2.waitKey(1) & 0xFF ==ord('q'):
                break
        else:
            break
    print('The video of virtual cube on AR tag is saved in the folder')
    video1.release()
    cap.release()

if __name__ == '__main__':
    main()


