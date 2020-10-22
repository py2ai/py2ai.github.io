---
layout: post
title: How to make an interactive PSO algorithm in Python
categories: [GUI tutorial series]
mathjax: true
featured-img: pso2
summary: Using Matplotlib drag the target of PSO and let the particles optimize their position to find it.
---

[![GIF](https://github.com/py2ai/py2ai.github.io/blob/master/assets/img/posts/pso2.jpg?raw=true)](https://youtu.be/xEQv9YdvRiA "GIF")

Hello friends, today we will use Matplotlib in Python to make an interactive PSO environment, where you can change the target as well as 
the number of particles including their various parameters.

So we will use two python files: 

1) Draggable.py
2) main.py

Just make a new directory and place both these .py files. Then simply run the main.py. Lets see what's inside the Draggable.py

### Draggable.py

```python

# Welcome to PyShine

import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig = plt.figure(figsize=(4,4))
plt.xlim([-500, 500])
plt.ylim([-500, 500])
ax = fig.add_subplot(111)

class Draggable_Target:
	lock = None 
	def __init__(self, point):
		self.point = point
		self.press = None
		self.background = None
		self.ID = None

	def setID(self,ID):
		self.ID = ID
	def getID(self):
		return self.ID
		
	def connect(self):
		'connect to all the events we need'
		self.cidpress = self.point.figure.canvas.mpl_connect('button_press_event', self.on_press)
		self.cidrelease = self.point.figure.canvas.mpl_connect('button_release_event', self.on_release)
		self.cidmotion = self.point.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

	def on_press(self, event):
		if event.inaxes != self.point.axes: return
		if Draggable_Target.lock is not None: return
		contains, attrd = self.point.contains(event)
		if not contains: return
		self.press = (self.point.center), event.xdata, event.ydata
		Draggable_Target.lock = self
		canvas = self.point.figure.canvas
		axes = self.point.axes
		self.point.set_animated(True)
		canvas.draw()
		self.background = canvas.copy_from_bbox(self.point.axes.bbox)
		axes.draw_artist(self.point)
		canvas.blit(axes.bbox)

	def on_motion(self, event):
		if Draggable_Target.lock is not self:
			return
		if event.inaxes != self.point.axes: return
		self.point.center, xpress, ypress = self.press
		dx = event.xdata - xpress
		dy = event.ydata - ypress
		self.point.center = (self.point.center[0]+dx, self.point.center[1]+dy)
		print(str(self.point.center[0])+','+str( self.point.center[1]),file = open('target.csv','w'))
		canvas = self.point.figure.canvas
		axes = self.point.axes
		canvas.restore_region(self.background)
		axes.draw_artist(self.point)
		canvas.blit(axes.bbox)

	def on_release(self, event):
		'on release we reset the press data'
		if Draggable_Target.lock is not self:
			return

		self.press = None
		Draggable_Target.lock = None
		self.point.set_animated(False)
		self.background = None
		self.point.figure.canvas.draw()

	def disconnect(self):
		'disconnect all the stored connection ids'
		self.point.figure.canvas.mpl_disconnect(self.cidpress)
		self.point.figure.canvas.mpl_disconnect(self.cidrelease)
		self.point.figure.canvas.mpl_disconnect(self.cidmotion)

drs = []
circles =     [patches.Circle( (10, -10),    20,  label='Click to drag the Target', fc = 'k',color = 'k', alpha=1)]
cnt = 0
for circ in circles:
	ax.add_patch(circ)
	dr = Draggable_Target(circ)
	dr.setID(cnt)
	dr.connect()
	drs.append(dr)
	cnt+=1
plt.legend()
plt.show()

```

And here is the main.py

### main.py

```python

# Welcome to PyShine

import random
import math
import numpy as np
import csv, os
import _thread
import matplotlib.pyplot as plt


def start_drag():
	os.system('python Draggable.py')
_thread.start_new_thread(start_drag,())

initial = [5,5]
bounds = [(-800,800),(-800,800)]

colors = np.array([
    ( 31, 119, 180), (174, 199, 232), (255, 127,  14), (255, 187, 120),
    ( 44, 160,  44), (152, 223, 138), (214,  39,  40), (255, 152, 150),
    (148, 103, 189), (197, 176, 213), (140,  86,  75), (196, 156, 148),
    (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
    (188, 189,  34), (219, 219, 141), ( 23, 190, 207), (158, 218, 229),

    ( 31, 119, 180), (174, 199, 232), (255, 127,  14), (255, 187, 120),
    ( 44, 160,  44), (152, 223, 138), (214,  39,  40), (255, 152, 150),
    (148, 103, 189), (197, 176, 213), (140,  86,  75), (196, 156, 148),
    (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
    (188, 189,  34), (219, 219, 141), ( 23, 190, 207), (158, 218, 229),

    ( 31, 119, 180), (174, 199, 232), (255, 127,  14), (255, 187, 120),
    ( 44, 160,  44), (152, 223, 138), (214,  39,  40), (255, 152, 150),
    (148, 103, 189), (197, 176, 213), (140,  86,  75), (196, 156, 148),
    (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
    (188, 189,  34), (219, 219, 141), ( 23, 190, 207), (158, 218, 229),

    ( 31, 119, 180), (174, 199, 232), (255, 127,  14), (255, 187, 120),
    ( 44, 160,  44), (152, 223, 138), (214,  39,  40), (255, 152, 150),
    (148, 103, 189), (197, 176, 213), (140,  86,  75), (196, 156, 148),
    (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
    (188, 189,  34), (219, 219, 141), ( 23, 190, 207), (158, 218, 229)

]) / 255.


class Particle: 
	def __init__(self,initial):
		self.pos=[]
		self.vel=[] 
		self.best_pos=[] 
		self.best_error=-1 
		self.error=-1     
		for i in range(0,num_dimensions): 
			self.vel.append(random.uniform(-1,1))
			self.pos.append(initial[i])
	
	def update_velocity(self,global_best_position): 
		w = 0.5
		c1 = 1 
		c2 = 2 
		
		for i in range(0,num_dimensions): 
			r1=random.random()
			r2=random.random()
			
			cog_vel=c1*r1*(self.best_pos[i]-self.pos[i])
			social_vel=c2*r2*(global_best_position[i]-self.pos[i])
			self.vel[i]=w*self.vel[i]+cog_vel+social_vel 
		
	def update_position(self,bounds): 
		for i in range(0,num_dimensions):
			self.pos[i]=self.pos[i]+self.vel[i]
			
			
			if self.pos[i]>bounds[i][1]:
				self.pos[i]=bounds[i][1]

				
			if self.pos[i] < bounds[i][0]:
				self.pos[i]=bounds[i][0]
	
	
	def evaluate_fitness(self,fitness_function):
		self.error=fitness_function(self.pos) 
		print("ERROR------->",self.error)
		
		if self.error < self.best_error or self.best_error==-1:
			self.best_pos=self.pos 
			self.best_error=self.error

def fitness_function(x):
	x0,y0 = getXY('target.csv') 
	x0=float(x0)
	y0=float(y0)
	total=0 
	total+=(x0-x[0])**2 +(y0-x[1])**2
	return total


def getXY(filename):
	lat=0
	long=0
	with open(filename) as csvDataFile:
		csvReader = csv.reader(csvDataFile)
		for row in csvReader:
			lat = row[0]
			long= row[1]
	return lat,long
import time

class Interactive_PSO():
	def __init__(self,fitness_function,initial,bounds,num_particles):
		global num_dimensions 
		
		num_dimensions = len(initial) 
		global_best_error=-1             
		global_best_position=[] 
		self.gamma = 0.0001
		swarm=[]
		for i in range(0,num_particles):
			swarm.append(Particle(initial))

		i=0
		while True: 
			
			#print('x'+','+'y',file = open('pos.csv','w'))
			for j in range(0,num_particles):
				swarm[j].evaluate_fitness(fitness_function)
				print('global_best_position',swarm[j].error,global_best_error)

				
				if swarm[j].error < global_best_error or global_best_error == -1:
					global_best_position=list(swarm[j].pos) 
					global_best_error=float(swarm[j].error)
					plt.title("PyShine Interactive PSO, Particles:{}, Error:{}".format(num_particles,round(global_best_error,1)))
					
				if i%2==0:	
					global_best_error=-1
					global_best_position = list([swarm[j].pos[0]+self.gamma*(swarm[j].error)*random.random() ,swarm[j].pos[1]+self.gamma*(swarm[j].error)*random.random() ])
					
				
			pos_0 = {}
			pos_1 = {}
			for j in range(0,num_particles): 
				pos_0[j] = []
				pos_1[j] = []	
			
			for j in range(0,num_particles): 
				swarm[j].update_velocity(global_best_position)
				swarm[j].update_position(bounds) 
				
			
				pos_0[j].append(swarm[j].pos[0])
				pos_1[j].append(swarm[j].pos[1])
				#print(str(swarm[j].pos[0])+','+str(swarm[j].pos[1]),file = open('pos.csv','a'))
				plt.xlim([-500, 500])
				plt.ylim([-500, 500])
				
			for j in range(0,num_particles):
				plt.plot(pos_0[j], pos_1[j],  color = colors[j],marker = 'o'  )


			x,y = getXY('target.csv')	 
			plt.plot(float(x), float(y),  color = 'k',marker = 'o'  )
			plt.pause(0.01)
			
			plt.clf() 
			i+=1
		print ('Results')
		print ('Best Position:',global_best_position)
		print( 'Best Error:',global_best_error)

		
Interactive_PSO(fitness_function,initial,bounds,num_particles=1)# let say 2 particles and 50 iterations
if __name__ == "__Interactive_PSO__":
    main()

```




