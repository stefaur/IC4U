import numpy as np
import cv2
import math
from PIL import Image
from pygame import mixer
from heapq import *
# initialize audio module
mixer.init()

gap = 40
width = 840
height = 480
COLUMNS = (width//gap)
ROWS = height//gap

win = np.array(Image.open('obstacles.jpg'))
win = cv2.cvtColor(win, cv2.COLOR_BGR2RGB)

RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (128, 128, 128)
class Square:
	def __init__(self, row, col, width, total_columns, total_rows):
		self.row = row
		self.col = col
		self.x = row * width
		self.y = col * width
		self.color = WHITE
		self.neighbors = []
		self.width = width
		self.total_columns = total_columns
		self.total_rows = total_rows

	def get_location(self):
		return self.row, self.col

	def is_obstacle(self):
		return self.color == RED

	def is_dest(self):
		return self.color == BLACK

	def create_obstacle(self):
		self.color = RED

	def create_dest(self):
		self.color = BLACK

	def create_path(self):
		self.color = BLACK

	def draw(self, win):
		cv2.rectangle(win,(self.x, self.y, self.width, self.width), self.color, -1)
			
	def update_neighbors(self, grid):
		self.neighbors = []
		if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_obstacle(): # RIGHT
			self.neighbors.append(grid[self.row][self.col + 1])
		if self.col > 0 and not grid[self.row][self.col - 1].is_obstacle(): # LEFT
			self.neighbors.append(grid[self.row][self.col - 1])
		if self.row < self.total_columns - 1 and not grid[self.row + 1][self.col].is_obstacle(): # DOWN
			self.neighbors.append(grid[self.row + 1][self.col])
		if self.row > 0 and not grid[self.row - 1][self.col].is_obstacle(): # UP
			self.neighbors.append(grid[self.row - 1][self.col])

	def __lt__(self, other):
		return False

def h(a, b):
    return (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2

def reconstruct_path(came_from, current, draw):
	while current in came_from:
		current = came_from[current]
		current.create_path()
		draw()

def algo(draw, grid, start, end):
	count = 0
	oheap = []
	came_from = {}
	g_score = {square: float("inf") for row in grid for square in row}
	g_score[start] = 0
	f_score = {square: float("inf") for row in grid for square in row}
	f_score[start] = h(start.get_location(), end.get_location())
	open_set_hash = {start}
	heappush(oheap, (f_score[start], start))
	while oheap:
		current = heappop(oheap)[1]
		open_set_hash.remove(current)
		if current == end:
			reconstruct_path(came_from, end, draw)
			return True
		for neighbor in current.neighbors:
			temp_g_score = g_score[current] + 1
			if temp_g_score < g_score[neighbor]:
				came_from[neighbor] = current
				g_score[neighbor] = temp_g_score
				f_score[neighbor] = temp_g_score + h(neighbor.get_location(), end.get_location())
				if neighbor not in open_set_hash:
					count += 1
					open_set_hash.add(neighbor)
					heappush(oheap, (f_score[neighbor], neighbor))
			draw()
	return False
	
def make_grid(columns, rows):
	grid = []
	for i in range(columns):
		grid.append([])
		for j in range(rows):
			square = Square(i, j, gap, columns, rows)
			grid[i].append(square)
	return grid

def draw_grid(win, columns, rows, width, height):
	for i in range(rows):
		cv2.line(win, (0, i * gap), (width, i * gap), GREY, 1)
		for j in range(columns):
			cv2.line(win, (j * gap, 0), (j * gap, height), GREY, 1)

def draw(win, grid, columns, rows, width, height):
	win.fill(255)
	for row in grid:
		for square in row:
			square.draw(win)
	draw_grid(win, columns, rows, width, height)

def count_white_pixel(xmax,ymax,img):
	conta=0
	contaMax=0
	xtrovato,ytrovato=0,0
	x=0
	while(x<xmax):
		y=0
		while(y<ymax):
			if (img[y,x]== 255):
				conta+=1
			elif (contaMax<conta and xtrovato==0 and ytrovato==0):	   
				contaMax=conta
				xtrovato,ytrovato=x,y
				conta=0
			y+=1
		x+=1
	return conta, xtrovato,ytrovato
	
def make_obstacles(width, height, img):
	for x in range (width):
		for y in range (height):
			if img[y,x,2]>= 253 and img[y,x,1]<= 3 and img[y,x,0]<=3:	
				col, row =find_square(x,y,gap)
				square = grid[col][row]
				square.color=RED
				square.update_neighbors(grid)

def find_square(cp,rp, gap):
	c=int(cp/gap)
	r=int(rp/gap)
	return c,r

def go(c,r,imgPath,dimElem,xp,yp):
  ret=-1
  ymax,xmax=480, 840
  
  if xp<xmax/2:
    if r>0 and compare_square(c,r-1,imgPath,dimElem):
      ret=0
    else:
      if c<COLUMNS and compare_square(c+1,r,imgPath,dimElem):
        ret=2
      else:
        if c>0 and compare_square(c-1,r,imgPath,dimElem):
          ret=1
  else:
    if xp>xmax/2:
      if r<ROWS and compare_square(c,r-1,imgPath,dimElem):
        ret=0
      else:
       if c>0 and compare_square(c-1,r,imgPath,dimElem):
          ret=1
       else:
         if c<COLUMNS and compare_square(c+1,r,imgPath,dimElem):
           ret=2
  return ret

def compare_square(c,r,imgPath,grandpost):
   result=False
   arr = imgPath[r*grandpost:(r+1)*grandpost,c*grandpost:(c+1)*grandpost]
   # there are no numpy functions that work with RGB, to see if a pixel is completely red, so I selected the central pixel in the square. It's black if the number is less than 3
   if arr.shape==(40,40,3):
	   if arr[10,10,2]<= 3 and arr[10,10,1]<= 3 and arr[10,10,0]<=3:
		   result=True
   return result

def main():
	global start, grid
	grid = make_grid(COLUMNS, ROWS)

	k=60
	run = True
	while run:
		img1 = np.array(Image.open('./mask/m'+str(k)+'.jpg'))
		img2 = np.array(Image.open('./frame/f'+str(k)+'.jpg'))
		img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
		pixels,x,y=count_white_pixel(width, height,img1)
		if pixels !=0:
			c, r =find_square(x,y, gap)
		
			make_obstacles(width, height, win)
		
			start = grid[c][r]

			end = grid[14][0]
			end.create_dest()
		
			cv2.namedWindow('path')
			for row in grid:
				for square in row:
					square.update_neighbors(grid)
			algo(lambda: draw(win, grid, COLUMNS, ROWS, width, height), grid, start, end)
			
			ret=go(c,r,win,gap,x,y)
			if ret == 0:
				mixer.music.load('audio/foreword.mp3')
				mixer.music.play()

			elif ret == 1:
				mixer.music.load('audio/left.mp3')
				mixer.music.play()

			elif ret == 2:
				mixer.music.load('audio/right.mp3')
				mixer.music.play()

			cv2.imshow("frame", img2)
			cv2.imshow("mask", img1)		
			cv2.imshow("path", win)
		
			start = None
			end = None
			grid = make_grid(COLUMNS, ROWS)
		
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
			if k>=240:
				break
		k+=10
main()
