Part 1A
Detecting AR tag using FFT:
    1.Convert to grayscale and blur the image  
    2.Generating mask to remove the white paper.
    3.Adding the mask to the thresholded and blurred image.
    4.Applying FFT and inverse FFT to detect the tag
To run the code
$ python3 Problem1_a.py

Part 1B
Decoding the reference AR tag:
    The orientation is 0 oand the tag id is 15
Decoding the tag from video frames:-
    1. Detect the tag(Same process from A.1 to A.3) and then calculate the minimum and maximum of balck pixels to get the 4 corners.
    2.Warp the tag using Homography matrix and inverse warping
    3. Decode the tag
To run the code
$ python3 Problem1_b.py

Part 2A
Superimposing the testudo on AR tag
    1.Once the 4 corners of the AR tag are detected using the steps mentioned in the above section, we compute the homography between the AR tag corners and the testudo image corners (the image is resized to (320,320)).
    2.Once the homography matrix is computed, the testudo is warped on the AR tag by using inverse warping.
To run the code
$ python3 Problem2_a.py

Part 2B
Placing a virtual cube on the AR tag
    1.To place a virtual cube on the AR tag we first need to calculate the projection matrix from the camera intrinsic matix and the homography matrix.
    2.Now that we have the projection matrix we can convert any point in homogenous world coordinate to a coordinate in camera image plane. We have the bottom 4 corners of the cube. The top four corners of the cube in the homogenous world frame are [[1,0,-1, 1], [1,1,-1,1],[0,1,-1,1], [0,0,-1,1]]. Using the projection matrix we convert these coordinates to the camera image plane and then draw a cube based on the 8 points that we have.
To run the code
$ python3 Problem2_a.py