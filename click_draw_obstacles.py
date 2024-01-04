from PIL import Image
import cv2
import numpy as np
import os

WIDTH,HEIGHT = 848,480

dimSquare=40
column, row = 0,0

def findSquarePosition(cp,rp):
	c= cp//dimSquare
	r= rp //dimSquare
	return c,r


def createSquare(c,r, img):
  img = cv2.rectangle(img,(c*dimSquare,r*dimSquare),((c+1)*dimSquare-1,(r+1)*dimSquare-1), (0, 0, 255), -1)
  return img

img = cv2.imread('image.jpg')
img_dup = np.copy(img)
mouse_pressed = False
starting_x=starting_y= -1
def mousebutton(event,x,y,flags,param):
    global img_dup , starting_x,starting_y, mouse_pressed
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_pressed = True
        starting_x,starting_y=x,y
        
    elif event == cv2.EVENT_MOUSEMOVE:
      if mouse_pressed:
        starting_x,starting_y=x,y

    elif event == cv2.EVENT_LBUTTONUP:
        mouse_pressed=False
        ending_x,ending_y= x,y
cv2.namedWindow('image')
cv2.setMouseCallback('image',mousebutton)

while True:
  cv2.imshow("image", img_dup)
  column, row = findSquarePosition(starting_x, starting_y)
  createSquare(column, row, img_dup)
  if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
        
        
cv2.imwrite("obstacles.jpg", img_dup)
cv2.destroyAllWindows()


