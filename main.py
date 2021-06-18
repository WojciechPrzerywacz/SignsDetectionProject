import cv2
import numpy as np
import tkinter as tk
from tkinter import *
from tkinter import filedialog as fd
from playsound import playsound
import time

def nothing(x):
    pass

def openFile():
    filepath = fd.askopenfilename(initialdir="C:\\Users\\twoto\\PycharmProjects\\SignsDetectionProject",
                                          title="Open file okay?",
                                          filetypes= (("images","*.jpg *.png"),
                                          ("all files","*.*")))
    return filepath


# 21.jpg big.png 20.jpeg sunny.jpeg
filename = "C:\\Users\\twoto\\PycharmProjects\\SignsDetectionProject\\3.jpg"

cv2.namedWindow("Color based detection", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow('Trackbar window', cv2.WINDOW_NORMAL)

cv2.resizeWindow('Trackbar window',1000,400)
cv2.createTrackbar("Min Sat", "Trackbar window", 105, 250, nothing)
cv2.createTrackbar("Max Hue", "Trackbar window", 36, 250, nothing)
cv2.createTrackbar("Min Val", "Trackbar window", 20, 100, nothing)
cv2.createTrackbar("Sign size", "Trackbar window", 40, 100, nothing)
cv2.createTrackbar("Image crop", "Trackbar window", 0, 100, nothing)

end_val = 0
# Stale dla kolorow czerwony i zolty
l_h = 10
l_v = 5
u_s = 255
u_v = 255

cap = cv2.imread(filename, 0)

img_height, img_width = cap.shape
font = cv2.FONT_HERSHEY_COMPLEX
while True:
    cv2.getTrackbarPos("", "Trackbar window")
    frame2 = cv2.imread(filename, 1)
    crop_size = cv2.getTrackbarPos("Image crop", "Trackbar window")
    frame = frame2[0:int(img_height-(crop_size*img_height/200)), 0:img_width]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #cv2.imshow("Color", hsv)
    l_s = cv2.getTrackbarPos("Min Sat", "Trackbar window")
    u_h = cv2.getTrackbarPos("Max Hue", "Trackbar window")
    size_modifier = cv2.getTrackbarPos("Sign size", "Trackbar window")
    l_v = cv2.getTrackbarPos("Min Val", "Trackbar window")
    lower_red = np.array([l_h, l_s, l_v])
    upper_red = np.array([u_h, u_s, u_v])

    mask = cv2.inRange(hsv, lower_red, upper_red)
    kernel = np.ones((1, 1), np.uint8) #określa rozmiar jednego 'pixela' na masce
    mask = cv2.erode(mask, kernel)
    #stały rozmiar okna (cv2.resize(mask, (960, 540))
    cv2.imshow("Color based detection", mask)
    key = cv2.waitKey(1)
    if key == 27:
        break
    if key == 32:
        filename = openFile();
        continue
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #NONE lub SIMPLE

    #print(len(contours))
    suspectsCounter = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        approx = cv2.approxPolyDP(cnt, 0.05 * cv2.arcLength(cnt, True), True)
        x = approx.ravel()[0]
        y = approx.ravel()[1]
        #print(x, " + ",y)
        # area depends on image size
        perfect_size = (img_height * img_width) * 0.004 * size_modifier / 100
        if area > perfect_size:
            cv2.drawContours(frame, [approx], 0, (0, 0, 0), 4)

            if len(approx) == 3:
                suspectsCounter = suspectsCounter + 1
                cv2.putText(frame, "Suspect", (x, y), font, 1, (0, 0, 0))
                image = frame
                out = np.zeros_like(image)  # Extract out the object and place into output image
                out[mask == 255] = image[mask == 255]
                #współrzędne gdzie maska biała
                (x, y) = np.where(mask == 255)
                (topx, topy) = (np.min(x), np.min(y))
                (bottomx, bottomy) = (np.max(x), np.max(y))

                x, y, w, h = cv2.boundingRect(cnt)
                #print(x, " + ", y, " + ",w, " + ", h) #pokazanie wykrytego znaku
                crop_img = out[y:y + h, x:x + w]
                #cv2.imshow('Suspect', crop_img)

    print(suspectsCounter)
    #if suspectsCounter == 1:
        #playsound("C:\\Users\\twoto\\PycharmProjects\\SignsDetectionProject\\beep.wav")

    cv2.namedWindow('Detection', cv2.WINDOW_AUTOSIZE)
    frame = cv2.putText(frame, "Click ESC to quit", org=(20, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2)
    #img_concate=np.concatenate(((cv2.inRange((cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)), lower_red, upper_red)),frame),axis=0)
    cv2.imshow('Detection', frame2)


cv2.destroyAllWindows()
