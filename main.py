import cv2
import numpy as np
import serial
import time

ser = serial.Serial('COM3',9600)

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
X=0
Y=0

kp = 2
ki = 0.3
kd = 5
sum_x = 0
sum_y = 0
last_x = 0
last_y = 0


def eval_PID(x,y):
    global kp
    global ki
    global kd
    global sum_x
    global sum_y
    global last_x
    global last_y


    error_x = x-240
    error_y = y-240
    sum_x+=error_x
    sum_y+=error_y
    pid_x = kp*error_x + ki*sum_x + kd*(error_x-last_x)
    pid_y = kp*error_y + ki*sum_y + kd*(error_y-last_y)

    last_x = error_x
    last_y = error_y
    return pid_x,pid_y

while True:

    ret,frame = cap.read()
    frame = frame[0:480, 0:480]
    frame = cv2.GaussianBlur(frame,(15,15),0)
    frame_hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    lower_orange = np.array([7, 97, 75])
    upper_orange = np.array([23, 255, 255])

    mask = cv2.inRange(frame_hsv,lower_orange,upper_orange)

    kernel = np.array((3,3),np.uint8)

    erosion = cv2.erode(mask,kernel)
    dilate = cv2.dilate(erosion,kernel)

    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    contours,ret = cv2.findContours(closing,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    if len(contours)>0:
        max_contour = max(contours,key = cv2.contourArea)
        # print(cv2.contourArea)

        # m = cv2.moments(max_contour)
        # Cx = int(m["m10"] / m["m00"])
        # Cy = int(m["m01"] / m["m00"])
        # center = (Cx, Cy)
        (x,y),radius = cv2.minEnclosingCircle(max_contour)

        center = (int(x), int(y))
        # print(center[0],center[1])
        X = center[0]
        Y = center[1]
        radius = int(radius)
        cv2.circle(frame, center, 2, (0, 0, 0), 2)
        cv2.circle(frame, center, radius, (0, 255, 0), 2)
        cv2.putText(frame,str(center),center,cv2.FONT_HERSHEY_DUPLEX,0.5,(0,255,0))



    cv2.circle(frame, (240,240), 2, (0, 0, 255), 3)
    cv2.rectangle(frame,[0,0],[480,480],(255,0,0),2)
    cv2.imshow('Frame',frame)

    x = int(X)
    y = int(Y)
    pid_x,pid_y = eval_PID(X,Y)
    # error_x = x-240
    # error_y = y-240
    # sum_x += error_x
    # sum_y += error_y
    # pid_x = kp*error_x + ki*sum_x + kd*(error_x-last_x)
    # pid_y = kp*error_y + ki*sum_y + kd*(error_y-last_y)

    # last_x = error_x
    # last_y = error_y

    # print(X,Y)
    ser.write(bytes(str(pid_x)+','+str(pid_y)+'\n','utf-8'))
    print(pid_x,pid_y)
    if cv2.waitKey(1) == ord('q') and 0xFF:
        break



cap.release()
cv2.destroyAllWindows()
