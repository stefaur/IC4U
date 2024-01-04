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


# creates the trackbars in a window
def nothing(x):
    pass
    
cv2.namedWindow('image')

cv2.createTrackbar('RMin', 'image', 0, 255, nothing)
cv2.createTrackbar('GMin', 'image', 0, 255, nothing)
cv2.createTrackbar('BMin', 'image', 0, 255, nothing)
cv2.createTrackbar('RMax', 'image', 0, 255, nothing)
cv2.createTrackbar('GMax', 'image', 0, 255, nothing)
cv2.createTrackbar('BMax', 'image', 0, 255, nothing)

cv2.setTrackbarPos('RMax', 'image', 255)
cv2.setTrackbarPos('GMax', 'image', 255)
cv2.setTrackbarPos('BMax', 'image', 255)


RMin = GMin = BMin = RMax = GMax = BMax = 0




while True:

    nmat, frame = cap.read()
    
    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if time() - t >= 2.5:
        sock.sendto("_GPHD_:0:0:2:0.000000\n".encode(), ("10.5.5.9", 8554))
        t=time()
    nframe+=1
    
    
    
    
    # Get current positions of all trackbars
    RMin = cv2.getTrackbarPos('RMin', 'image')
    GMin = cv2.getTrackbarPos('GMin', 'image')
    BMin = cv2.getTrackbarPos('BMin', 'image')
    RMax = cv2.getTrackbarPos('RMax', 'image')
    GMax = cv2.getTrackbarPos('GMax', 'image')
    BMax = cv2.getTrackbarPos('BMax', 'image')

    
    lower = np.array([BMin, GMin, RMin], np.uint8) 
    upper = np.array([BMax, GMax, RMax], np.uint8)
    
    mask = cv2.inRange(hsvImage, lower, upper)
    # Display result images
    
    cv2.imshow('frame', frame)
    cv2.imshow('image', mask)
    

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
