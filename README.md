# Augmented Reality with ArUco Markers - GitHub Repository

## Introduction
This repository contains the code for Task 4, which involves tag-based augmented reality using the pyAprilTag package. The code detects an ArUco marker in an image, utilizes the camera calibration parameters obtained in Task 3 to draw a 3D cube on top of the marker, and visualizes the augmented reality from different perspectives.

![View 1](Aruco-Cube/images/aruco cube 1.jpg)
![View 2](Aruco-Cube/images/aruco cube 2.jpg)
![View 3](Aruco-Cube/images/aruco cube 3.jpg)



## Task 4: Tag-based Augmented Reality (5pt)
### Code Summary
1. **Import Necessary Libraries:**
   - OpenCV (cv2): For computer vision tasks.
   - NumPy: For numerical operations.

2. **Define Euler to Rotation Matrix Function:**
   - Define a function to convert Euler angles into a rotation matrix.

3. **Initialize ArUco Dictionary and Parameters:**
   - Initialize the ArUco dictionary and parameters for marker detection using OpenCV’s ArUco module.

4. **Initialize Video Capture:**
   - Initialize video capture from the default camera (you can adjust the camera index if needed).

5. **Define Camera Matrix and Distortion Coefficients:**
   - Define the camera matrix (mtx) and distortion coefficients (dist) obtained from camera calibration (Task 3).

6. **Enter Infinite Loop for Augmented Reality:**
   - Enter an infinite loop that captures frames from the camera until the 'q' key is pressed.
   - Read a frame and convert it to grayscale.
   - Detect ArUco markers in the grayscale frame using `cv2.aruco.detectMarkers`.
   - If markers are detected, loop through each detected marker and perform the following steps:
      - Estimate the marker’s 3D pose using `cv2.aruco.estimatePoseSingleMarkers`.
      - Convert the rotation vector (rvec) to a rotation matrix using the euler to rotation matrix function.
      - Compute the transformation matrix H that combines rotation and translation.
      - Calculate the 3D coordinates of the marker’s corners in the camera frame.
      - Compute the normal vector of the marker’s plane.
      - Calculate 3D coordinates of additional points (P5, P6, P7, and P8) on the marker’s plane using the normal and the 3D points P1, P2, P3, and P4.
      - Transform these additional points back to image coordinates.
      - Draw lines and annotations on the frame to visualize the marker’s position and orientation.
   - Display the frame with ArUco markers and pose estimation.
   - Exit the loop if the 'q' key is pressed.

7. **Release Resources:**
   - Release the video capture object and close OpenCV windows.

### Code Location
The code for tag-based augmented reality with ArUco markers can be found in the file: `Task4_Augmented_Reality.py`

## Usage
1. Clone this repository.
2. Navigate to the root directory.
3. Run the Python script `aruco cube.py`.

## Author
- Name: Ashiq Rahman Anwar Batcha

## Acknowledgments
Thanks to the pyAprilTag package and OpenCV for enabling augmented reality with ArUco markers!
