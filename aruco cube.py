# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 15:20:32 2023

@author: techv
"""
import cv2
import numpy as np


def euler_to_rotation_matrix(roll, pitch, yaw):
    """
    Convert Euler angles to a rotation matrix.

    Parameters:
    - roll (float): Rotation angle around X-axis (in radians).
    - pitch (float): Rotation angle around Y-axis (in radians).
    - yaw (float): Rotation angle around Z-axis (in radians).

    Returns:
    - rotation_matrix (numpy.ndarray): 3x3 rotation matrix.
    """
    # Create individual rotation matrices for each axis
    Rx = np.array([[1,            0,             0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll), np.cos(roll)]])

    Ry = np.array([[np.cos(pitch),  0, np.sin(pitch)],
                   [            0,  1,             0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])

    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw),  np.cos(yaw), 0],
                   [0, 0, 1]])

    # Combine the individual rotation matrices to get the overall rotation matrix
    rotation_matrix = np.dot(Rz, np.dot(Ry, Rx))

    return rotation_matrix

# Define the ArUco dictionary and parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters_create()

# Initialize the video capture (you may need to adjust the camera index)
cap = cv2.VideoCapture(0)

mtx  = np.array([[630.94167118,   0.,         309.58687994],
                 [  0.,         639.39125771, 213.25071191],
                 [  0.,           0.,           1.        ]]) 

dist = np.array([[ 6.13543634e-02,  4.67541314e-01, -1.75439625e-02,  2.34532801e-03,  -3.02481784e+00]])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        for i in range(len(ids)):
            # Estimate pose of each marker and return the values rvec and tvec
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.02, mtx, dist)
            
            # Convert rotation vector to rotation matrix
            R = euler_to_rotation_matrix(rvec[0][0][0], rvec[0][0][1], rvec[0][0][2])
            
            # Compute transformation matrix H
            T = np.reshape(tvec[0][0],(3,1))
            H = np.dot(mtx,np.hstack((R,T)))
            
            # Calculate 3D coordinates of marker corners in camera frame
            p1 = np.dot(np.linalg.pinv(H),np.vstack(([[corners[0][0][0][0]],[corners[0][0][0][1]]],[[1]])))
            p1 = p1/p1[3][0]
            
            # Repeat for other marker corners (p2, p3, p4)
            p2 = np.dot(np.linalg.pinv(H),np.vstack(([[corners[0][0][1][0]],[corners[0][0][1][1]]],[[1]])))
            p2 = p2/p2[3][0]
            p12 = p1[0:3] - p2[0:3]
            
            p3 = np.dot(np.linalg.pinv(H),np.vstack(([[corners[0][0][2][0]],[corners[0][0][2][1]]],[[1]])))
            p3 = p3/p3[3][0]
            p13 = p1[0:3] - p3[0:3]
            
            p4 = np.dot(np.linalg.pinv(H),np.vstack(([[corners[0][0][3][0]],[corners[0][0][3][1]]],[[1]])))
            p4 = p4/p4[3][0]
            p14 = p1[0:3] - p4[0:3]
            
            # Calculate the normal vector of the marker plane
            normal = np.cross(p12.T,p13.T)
            norm = np.linalg.norm(normal)
            
            # Calculate 3D coordinates of points P5, P6, P7, and P8
            # Transform P5, P6, P7, P8 back to image coordinates
            print("normal :\n", normal/norm)
            P5 = p1[0:3] + norm.T
            print("P5 :\n", P5)
            p5 = np.dot(H,np.vstack((P5,[[1]])))
            p5 = p5/p5[2][0]
            print(p5)
            
            P6 = p2[0:3] + norm.T
            print("P6 :\n", P6)
            p6 = np.dot(H,np.vstack((P6,[[1]])))
            p6 = p6/p6[2][0]
            print(p6)
            
            P7 = p3[0:3] + norm.T
            print("P5 :\n", P5)
            p7 = np.dot(H,np.vstack((P7,[[1]])))
            p7 = p7/p7[2][0]
            print(p7)
            
            P8 = p4[0:3] + norm.T
            print("P8 :\n", P8)
            p8 = np.dot(H,np.vstack((P8,[[1]])))
            p8 = p8/p8[2][0]
            print(p8)

            # Draw lines and annotations on the frame
            frame = cv2.line(frame, (int(corners[0][0][0][0]),int(corners[0][0][0][1])), (int(corners[0][0][1][0]),int(corners[0][0][1][1])), (0,0,255), 3)
            frame = cv2.line(frame, (int(corners[0][0][1][0]),int(corners[0][0][1][1])), (int(corners[0][0][2][0]),int(corners[0][0][2][1])), (0,0,255), 3)
            frame = cv2.line(frame, (int(corners[0][0][2][0]),int(corners[0][0][2][1])), (int(corners[0][0][3][0]),int(corners[0][0][3][1])), (0,0,255), 3)
            frame = cv2.line(frame, (int(corners[0][0][3][0]),int(corners[0][0][3][1])), (int(corners[0][0][0][0]),int(corners[0][0][0][1])), (0,0,255), 3)
            
            frame = cv2.line(frame, (int(abs(p5[0][0])),int(abs(p5[1][0]))), (int(abs(p6[0][0])),int(abs(p6[1][0]))), (0,0,255), 3)
            frame = cv2.line(frame, (int(abs(p6[0][0])),int(abs(p6[1][0]))), (int(abs(p7[0][0])),int(abs(p7[1][0]))), (0,0,255), 3)
            frame = cv2.line(frame, (int(abs(p7[0][0])),int(abs(p7[1][0]))), (int(abs(p8[0][0])),int(abs(p8[1][0]))), (0,0,255), 3)
            frame = cv2.line(frame, (int(abs(p8[0][0])),int(abs(p8[1][0]))), (int(abs(p5[0][0])),int(abs(p5[1][0]))), (0,0,255), 3)
            
            frame = cv2.line(frame, (int(abs(p5[0][0])),int(abs(p5[1][0]))), (int(corners[0][0][0][0]),int(corners[0][0][0][1])), (0,0,255), 3)
            frame = cv2.line(frame, (int(abs(p6[0][0])),int(abs(p6[1][0]))), (int(corners[0][0][1][0]),int(corners[0][0][1][1])), (0,0,255), 3)
            frame = cv2.line(frame, (int(abs(p7[0][0])),int(abs(p7[1][0]))), (int(corners[0][0][2][0]),int(corners[0][0][2][1])), (0,0,255), 3)
            frame = cv2.line(frame, (int(abs(p8[0][0])),int(abs(p8[1][0]))), (int(corners[0][0][3][0]),int(corners[0][0][3][1])), (0,0,255), 3)
            
           
    # Display the frame with ArUco markers and pose estimation
    cv2.imshow('ArUco Marker Detection', frame)
    # cv2.imshow('ArUco Marker Detection_', frame_)

    # Exit the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close OpenCV windows
cap.release()
cv2.destroyAllWindows()