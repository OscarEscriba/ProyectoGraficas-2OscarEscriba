import numpy as np
from MathLib import *
from math import cos, pi

class Light(object):
  def __init__(self, color = [1,1,1], intensity = 1, lightType = "None"):
    self.color = color
    self.intensity = intensity
    self.lightType = lightType

  def GetLightColor(self, intercept = None):
    return [(i* self.intensity) for i in self.color]

  def GetSpecularColor(self, intercept, viewPos):
    return [0,0,0]




class AmbientLight(Light):
  def __init__(self, color=[1, 1, 1], intensity=1):
    super().__init__(color, intensity, "Ambient")
  

class DirectionalLight(Light):
  def __init__(self, color=[1, 1, 1], intensity=1, direction = {0,-1,0}):
    super().__init__(color, intensity,"Directional")
    self.direction = direction / np.linalg.norm(direction)
  
  def GetLightColor(self, intercept = None):
    lightColor = super().GetLightColor()
    if intercept:
      dir = [-x for x in self.direction]
      intensity = np.dot(intercept.normal, dir)
      intensity = max(0, min(1, intensity))
      intensity *= (1-intercept.obj.material.ks)
      lightColor = [(i * intensity ) for i in lightColor]
    return lightColor
  
  def GetSpecularColor(self, intercept, viewPos):
    specColor = self.color
    if intercept:
      dir =  [-x for x in self.direction]
      reflect  = reflectVector(intercept.normal, dir)
      viewDIr = np.subtract(viewPos, intercept.point)
      viewDIr /= np.linalg.norm(viewDIr)
      specularity = max(0,np.dot(viewDIr, reflect)) ** intercept.obj.material.spec
      specularity *= intercept.obj.material.ks
      specularity *= self.intensity
      specColor = [(i*specularity) for i in specColor]
    return specColor

class PointLight(Light):
  def __init__(self, color=[1, 1, 1], intensity=1, position= [0,0,0], lightType = "Point"):
    super().__init__(color, intensity, lightType)
    self.lightType = lightType
    self.position = position
  
  def GetLightColor(self, intercept=None):
    lightColor = super().GetLightColor(intercept)
    if intercept:
      dir = np.subtract(self.position, intercept.point)
      R = np.linalg.norm(dir)
      dir /= R
      intensity = np.dot(intercept.normal, dir)
      intensity = max(0, min(1, intensity))
      intensity *= (1-intercept.obj.material.ks)
      #Ley de cuarados inversos
      # attenuation = intensity / R**2
      # R es la distancia del punto intercepto a las luz punto
      if R != 0:
        intensity /= R**2
      lightColor = [(i * intensity ) for i in lightColor]  
    return lightColor


  def GetSpecularColor(self, intercept, viewPos):
    specColor = self.color
    if intercept:
      dir = np.subtract(self.position, intercept.point)
      R = np.linalg.norm(dir)
      dir /= R
      reflect  = reflectVector(intercept.normal, dir)
      viewDIr = np.subtract(viewPos, intercept.point)
      viewDIr /= np.linalg.norm(viewDIr)
      specularity = max(0,np.dot(viewDIr, reflect)) ** intercept.obj.material.spec
      specularity *= intercept.obj.material.ks
      specularity *= self.intensity
      if R != 0:
        specularity /= R**2
      specColor = [(i*specularity) for i in specColor]
    return specColor
  
class SpotLight(PointLight):
  def __init__(self, color=[1, 1, 1], intensity=1, position=[0, 0, 0], lightType="Spot", direction = [0,-1,0], innerangle = 50, outterangle = 60):
    super().__init__(color, intensity, position, lightType)
    self.direction = direction /np.linalg.norm(direction)
    self.innerangle = innerangle
    self.outterangle = outterangle
  
  def GetLightColor(self, intercept=None):
    lightColor = super().GetLightColor(intercept)
    if intercept:
      lightColor = [i*self.SpotLightAttenuation(intercept=intercept) for i in lightColor] 
    return lightColor

  def GetSpecularColor(self, intercept, viewPos):
    specColor = super().GetSpecularColor(intercept, viewPos)
    if intercept:
      specColor =[i*self.SpotLightAttenuation(intercept=intercept) for i in specColor]     
    return specColor
  
  def SpotLightAttenuation(self, intercept):
    if intercept == None:
      return 0
    wi = np.subtract(self.position, intercept.point)
    wi /= np.linalg.norm(wi)
    innerAngleRads = self.innerangle * pi / 180
    ouuterAngleRads = self.outterangle * pi / 180
    attenuation = (-np.dot(self.direction, wi) - cos(ouuterAngleRads)) / (cos(innerAngleRads)-cos(ouuterAngleRads))
    attenuation = min(1, max(0,attenuation))
    return attenuation