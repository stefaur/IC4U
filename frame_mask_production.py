import cv2
from time import time
import socket
from goprocam import GoProCamera, constants
import numpy as np

gpCam = GoProCamera.GoPro()
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
t=time()
gpCam.livestream("start")
gpCam.video_settings(res="1080", fps='30')
gpCam.gpControlSet(constants.Stream.WINDOW_SIZE, constants.Stream.WindowSize.R720)
cap = cv2.VideoCapture("udp://10.5.5.9:8554", cv2.CAP_FFMPEG)
nframe=0


while True:

    nmat, frame = cap.read()
    
    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lmin = np.array([139, 112, 195], np.uint8) 
    lmax = np.array([225, 255, 255], np.uint8)
    mask = cv2.inRange(hsvImage, lmin, lmax)
    
    cv2.imshow('mask', mask)
    cv2.imshow('frame', frame)
    
    if nframe == 0:
        cv2.imwrite("./images/"+"image.jpg", frame)
    if nframe % 1 == 0:
        cv2.imwrite("./images/mask/"+"m"+str(int(nframe/5))+".jpg", mask)
        cv2.imwrite("./images/frame/"+"f"+str(int(nframe/5))+".jpg", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if time() - t >= 2.5:
        sock.sendto("_GPHD_:0:0:2:0.000000\n".encode(), ("10.5.5.9", 8554))
        t=time()
    nframe+=1

cap.release()
cv2.destroyAllWindows()
