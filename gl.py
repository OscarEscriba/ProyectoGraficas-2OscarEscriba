import struct

import pygame
from Camera import Camera
from math import tan, pi, atan2, acos
import numpy as np
#es el valor maximo de recursion	
MAX_RECIRSOPM_DEPTH = 3
def char(c):
	# 1 byte
	return struct.pack("=c", c.encode("ascii"))

def word(w):
	# 2 bytes
	return struct.pack("=h", w)

def dword(d):
	# 4 bytes
	return struct.pack("=l", d)



class RendererRT(object):
	def __init__(self, screen, enveriomentMap =None):
		
		self.screen = screen
		_, _, self.width, self.height = screen.get_rect()
		
		self.camera = Camera()
		self.glViewport(0,0, self.width, self.height)
		self.glProjection()
		
		self.glColor(1,1,1)
		self.glClearColor(0,0,0)
		self.glClear()
		
		self.scene = []
		self.lights = []
		self.enveriomentMap = enveriomentMap

	def glViewport(self, x, y, width, height):
		self.vpX = int(x)
		self.vpY = int(y)
		self.vpWidth = width
		self.vpHeight = height
		

	def glProjection(self, n = 0.1, f = 1000, fov = 60):
		self.nearPlane = n
		self.farPlane = f
		self.fov = fov * pi / 180
		
		aspectRatio = self.vpWidth / self.vpHeight
		
		self.topEdge = tan(self.fov / 2) * self.nearPlane
		self.rightEdge = self.topEdge * aspectRatio
		



	def glColor(self, r, g, b):
		r = min(1, max(0, r))
		g = min(1, max(0, g))		
		b = min(1, max(0, b))	
		
		self.currColor = [r,g,b]
		

	def glClearColor(self, r, g, b):
		r = min(1, max(0, r))
		g = min(1, max(0, g))		
		b = min(1, max(0, b))	
		
		self.clearColor = [r,g,b]
		

	def glClear(self):
		color = [int(i * 255) for i in self.clearColor]
		self.screen.fill(color)
		
		self.frameBuffer = [[self.clearColor for y in range(self.height)]
							for x in range(self.width)]
		

	def glEnviromentMapColor(self, orig, dir):
		if self.enveriomentMap:
			x = (atan2(dir[2], dir[0]) / (2*pi) + 0.5)
			y = acos(-dir[1]) / pi
			return self.enveriomentMap.getColor(x,y)
		return self.clearColor

	def glPoint(self, x, y, color = None):
		# Pygame empieza a renderizar desde la esquina
		# superior izquierda. Hay que voltear el valor y
		x = round(x)
		y = round(y)
		
		if (0<=x<self.width) and (0<=y<self.height):
			# Pygame recibe los colores en un rango de 0 a 255
			color = [int(i * 255) for i in (color or self.currColor)]
			self.screen.set_at((x, self.height - 1 - y), color)
			
			self.frameBuffer[x][y] = color
			

	def glGenerateFrameBuffer(self, filename):
		
		with open(filename, "wb") as file:
			# Header
			file.write(char("B"))
			file.write(char("M"))
			file.write(dword(14 + 40 + (self.width * self.height * 3)))
			file.write(dword(0))
			file.write(dword(14 + 40))
			
			# Info Header
			file.write(dword(40))
			file.write(dword(self.width))
			file.write(dword(self.height))
			file.write(word(1))
			file.write(word(24))
			file.write(dword(0))
			file.write(dword(self.width * self.height * 3))
			file.write(dword(0))
			file.write(dword(0))
			file.write(dword(0))
			file.write(dword(0))
			
			# Color table
			for y in range(self.height):
				for x in range(self.width):
					color = self.frameBuffer[x][y]
					color = bytes([color[2],
								   color[1],
								   color[0]])
					
					file.write(color)
				
					
	def glCastRay(self, orig, direction, sceneObject = None, recursion = 0):
		if recursion >= MAX_RECIRSOPM_DEPTH:
			return None
		intercept = None
		hit = None
		depth = float('inf') #profundidad
		for obj in self.scene:
			#esto es para ignorarse a si mismo y no crear sombra sobre uno
			if obj != sceneObject:
				intercept = obj.ray_intersect(orig, direction)
				if intercept != None:
					# si yo encuentro contacto con otro objeto si este punto que encontre 
					if intercept.distance < depth:
						hit = intercept
						depth = intercept.distance
		return hit


	def glRender(self):
		
		for x in range(self.vpX, self.vpX + self.vpWidth):
			for y in range(self.vpY, self.vpY + self.vpHeight):
				if 0<=x<self.width and 0<=y<self.height:
					# Coordenadas normalizadas
					# Que van de -1 a 1

					pX = ((x + 0.5 - self.vpX) / self.vpWidth) * 2 - 1
					pY = ((y + 0.5 - self.vpY) / self.vpHeight) * 2 - 1
					
					pX *= self.rightEdge
					pY *= self.topEdge
					
					dir = [pX, pY, -self.nearPlane]
					dir /= np.linalg.norm(dir)
					
					intercept = self.glCastRay(self.camera.translate, dir)
					color = [0,0,0]
					if intercept != None:
						color = intercept.obj.material.GetSurfaceColor(intercept, self)
					else:
						color = self.glEnviromentMapColor(self.camera.translate, dir)
					self.glPoint(x, y, color=color)
					pygame.display.flip()

					
					
					