#Determina cuales son las propiedades visuales de una superficie. 
#Como interactua la superficie con la luz
from MathLib import *
from refractionFunctions import *
OPAQUE = 0
REFLECTIVE = 1
TRANSPARENT = 2

class Material (object):
  #difuse se refiere al color de la superficie. Que cuando una superficie esta iluminaada de cierto luz que color refleja
  def __init__(self, difuse = [0.9,0.9,0.9], spec = 1.0 , ks = 0.0, matType = OPAQUE, texture = None, ior = 1.0):
    self.diffuse = difuse
    self.spec = spec
    self.ks = ks
    self.texture = texture
    self.matType = matType
    self.ior = ior

  def GetSurfaceColor(self, intercept, renderer, recursion = 0):

    # el modelo fond de iluminacion phong
    #PHNG REFLETION MODEL
    # LightColors = lightCOlor + Specular
    # el color final es 
    # FinalCOlor = DiffuseColor * LightCOlor

    lightColor = [0,0,0]
    finalColor = self.diffuse
    reflectColor = [0,0,0]
    refractColor = [0,0,0]
    if self.texture and intercept.textCoords:
      textureColor = self.texture.getColor(intercept.textCoords[0], intercept.textCoords[1])
      finalColor = [finalColor[i] * textureColor[i] for i in range(3)]

    for light in renderer.lights:
      shadowIntercept = None
      if light.lightType == "Directional":
        lightDir = [-x for x in light.direction]
        shadowIntercept = renderer.glCastRay(intercept.point, lightDir, intercept.obj)
      elif light.lightType == "Point":
        lightDir = np.subtract(light.position, intercept.point)
        R = np.linalg.norm(lightDir)
        lightDir /= R
        shadowIntercept = renderer.glCastRay(intercept.point, lightDir, intercept.obj)
        #si esta mas lejos el rayo lo ignoro
        if shadowIntercept:
          if shadowIntercept.distance >= R:
            shadowIntercept = None

      if shadowIntercept == None:
        lightColor = [(lightColor[i] + light.GetSpecularColor(intercept, renderer.camera.translate)[i]) for i in range (3)]

        if self.matType == OPAQUE:
          lightColor = [(lightColor[i] + light.GetLightColor(intercept)[i]) for i in range (3)]
    if self.matType == REFLECTIVE:
      raydir = [-x for x in intercept.rayDirection]
      reflect = reflectVector(intercept.normal, raydir)
      ##generamos el rayo que va a venir del nuevo punto y con el vecto de reflecto
      reflectintercept = renderer.glCastRay(intercept.point, reflect, intercept.obj, recursion+1)
      #en base a este intercepto tengo que ahora obtener un nuevo color de la superficie
      #si hice contacto con algun objeto
      if reflectintercept != None:
        reflectColor = reflectintercept.obj.material.GetSurfaceColor(reflectintercept, renderer, recursion+1)
      else:
        reflectColor = renderer.glEnviromentMapColor(intercept.point, reflect)
    
    elif self.matType == TRANSPARENT:
      #Revisamos si estamos afuera
      outside = np.dot(intercept.normal, intercept.rayDirection) < 0

      #Agregar ese margen de error o bias
      bias = [i *0.001 for i in intercept.normal]

      # Generamos los rayos de reflexion
      raydir = [-x for x in intercept.rayDirection]

      reflect = reflectVector(intercept.normal, raydir)

      #en el punto interepto le agregamos el margen de error
      reflectOrigin = np.add(intercept.point, bias) if outside else np.subtract(intercept.point, bias)

      #el objeto de la escena es None porque nos queremos revisar a nosotros
      reflectintercept = renderer.glCastRay(reflectOrigin, reflect, None, recursion+1)
      if reflectintercept != None:
        reflectColor = reflectintercept.obj.material.GetSurfaceColor(reflectintercept, renderer, recursion+1)
      else:
        reflectColor = renderer.glEnviromentMapColor(intercept.point, reflect)

      #Generamos los rayos de refraccion
      # 1 es del ior del aire
      if not totalInternalReflection(intercept.normal, intercept.rayDirection, 1.0, self.ior):
        refract = refractVector(intercept.normal, intercept.rayDirection, 1.0, self.ior)
        refractOrigin = np.subtract(intercept.point, bias) if outside else np.add(intercept.point, bias)
        refractIntercept = renderer.glCastRay(refractOrigin, refract, None, recursion+1)
        if refractIntercept != None:
          refractColor  = refractIntercept.obj.material.GetSurfaceColor(refractIntercept, renderer, recursion+1)
        else:
          refractColor = renderer.glEnviromentMapColor(intercept.point, refract)

        #con ecuaciones de frenel determinamos la refleccion y refraccion  agregamos

        Kr, Kt = fresnel(intercept.normal, intercept.rayDirection, 1.0, self.ior)
        reflectColor = [i * Kr for i in reflectColor]
        refractColor = [i * Kt for i in refractColor]

    finalColor = [(finalColor[i] * (lightColor[i] + reflectColor[i] + refractColor[i])) for i in range(3)]
    finalColor = [min(1, finalColor[i]) for i in range(3)]
    return finalColor
  
