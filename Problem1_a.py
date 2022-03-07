import cv2
from scipy import fft
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

def generate_mask(save_path, frame):
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
    cv2.imwrite(save_path + '/mask.png', dilation)
    mask = dilation + cur_frame
    mask = mask.astype('float32')
    ret, thresh1 = cv2.threshold(mask, 180, 255, cv2.THRESH_BINARY)   
    median = cv2.medianBlur(thresh1, 5)
    cv2.imwrite(save_path + '/masked_image.png', median)
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

def high_pass_filter(shape,save_path):
    rows = shape[0] 
    cols = shape[1]
    crow, ccol = int(rows / 2), int(cols / 2)
    mask = np.ones((rows, cols), np.uint8)
    r = 80
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
    mask[mask_area] = 0
    print(mask.shape) 
    return mask

def find_edges(thresh, save_path):
    # j = smooth_img.copy()
    # ret,thresh = cv2.threshold(np.uint8(j), 200 ,255,cv2.THRESH_BINARY)
    fft_img = fft.fft2(thresh, axes = (0,1))
    fft_img_shift = fft.fftshift(fft_img)
    magnitude_spectrum_fft = 20*np.log(np.abs(fft_img_shift))
    cv2.imwrite(save_path + '/fft.png', magnitude_spectrum_fft)
    fft_hpf = fft_img_shift * high_pass_filter(fft_img_shift.shape, save_path) #convolution becomes multiplication in the frequency domain
    magnitude_fft_hpf = 20*np.log(np.abs(fft_hpf))
    plt.imshow(magnitude_fft_hpf, cmap='gray')
    plt.axis('off')
    plt.savefig(save_path + '/HPF_filtered.png') 

    # inverse fft operations to find the edges
    inv_shift = fft.ifftshift(fft_hpf)
    inv_fft = fft.ifft2(inv_shift)
    img_back_edge = np.abs(inv_fft)
    plt.imshow(img_back_edge, cmap='gray')
    plt.axis('off')
    plt.savefig(save_path + '/ARTag_edges.png') 
    plt.show()
    return img_back_edge


def main():
    count = 0
    base_path = os.path.dirname(os.path.abspath(__file__))
    save_path = base_path + '/Images'
    if(not (os.path.isdir(save_path))):
        os.makedirs(save_path)
    video_file = base_path + '/1tagvideo.mp4'
    vidcap = cv2.VideoCapture(video_file)
    count = 0
    while True:    
        success,image = vidcap.read()
        count += 1
        if(count == 359):
            print('Extracted frame to detect the ARTag', count)
            img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(save_path + '/ARTag_frame.png',img)
            break
        if cv2.waitKey(1) & 0xFF ==ord('q'):
            break
    
    blur = cv2.blur(img,(21,21))
    c, masked_img = generate_mask(save_path, blur)
    edge_img = find_edges(masked_img, save_path)

if __name__ == '__main__':
    main()