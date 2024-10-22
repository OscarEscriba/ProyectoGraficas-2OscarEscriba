from gl import *
import pygame
from pygame.locals import *
from Figuras import *
from Material import *
from Lights import *
from texture import Texture

# Tamaño de pantalla
width = 520
height = 520
screen = pygame.display.set_mode((width, height), pygame.SCALED)
clock = pygame.time.Clock()

# Configuración del renderizador y entorno
rt = RendererRT(screen)
rt.enveriomentMap = Texture("Prueba2.bmp")  # Mapa de entorno
rt.glClearColor(0.5, 0.7, 0.9)  # Color de cielo suave
rt.glClear()

# Materiales
#fire_texture = Texture("fire_texture.bmp")  # Textura de fuego para los cilindros
reflective_material = Material(difuse=[0.5, 0.5, 0.5], spec=32, ks=0.9, matType=REFLECTIVE)  # Material reflectivo para la pirámide
brick = Material(difuse=[1,0.2,0.2], spec= 32, ks= 0.1)
grass = Material(difuse=[0.2,1,0.2], spec= 64, ks= 0.2)
mirror = Material(difuse=[0.9,0.9,0.9], spec= 128, ks= 0.4, matType=REFLECTIVE)
glass = Material(ior=1.5, spec=128, ks=0.2, matType=TRANSPARENT)
moon_material = Material(difuse=[0.9, 0.9, 0.9], spec=64, ks=0.3)  # Material para la luna
luna_material = Material(texture=Texture("Luna.bmp"))  # Textura para la luna
fire_material = Material(difuse=[1, 0.3, 0], spec=128, ks=0.8, matType=REFLECTIVE)  # Material para simular fuego

# Luces
rt.lights.append(DirectionalLight(direction=[-1, -1, -1], intensity=0.8))
rt.lights.append(AmbientLight(intensity=0.2))

# Pirámide en el centro de la escena (ajustada más pequeña)
pyramid = Pyramid(position=[0, -1, -5], base_size=1.75, height=2.5, material=mirror, rotation_angle=45)  # Pirámide en el centro
rt.scene.append(pyramid)

# Cilindros alrededor de la pirámide (tres a la derecha y tres a la izquierda, en orden descendente de altura)
# Para los cilindros de la derecha (positivo en X)
for i in range(2):  # 3 cilindros a la derecha de la pirámide
    height = 1 - i * 0.25  # Descendente en altura (de 2 a 1)
    x = 1.5 + i * 1.2  # Distancia creciente en X
    z = -5  # Los cilindros estarán alineados en Z
    cylinder = Cylinder(position=[x, -1, z], height=height, radius=0.1, material=fire_material)  # Cilindros con altura decreciente
    rt.scene.append(cylinder)

# Para los cilindros de la izquierda (negativo en X)
for i in range(2):  # 3 cilindros a la izquierda de la pirámide
    height = 1 - i * 0.25  # Descendente en altura (de 2 a 1)
    x = -1.5 - i * 1.2  # Distancia creciente en X (negativo)
    z = -5  # Los cilindros estarán alineados en Z
    cylinder = Cylinder(position=[x, -1, z], height=height, radius=0.1, material=fire_material)  # Cilindros con altura decreciente
    rt.scene.append(cylinder)

# Esfera para simular la luna arriba de la pirámide
moon = Sphere(position=[-1.3, 1.6, -6], radius=0.4, material=luna_material)  # Ajustar el tamaño de la luna
rt.scene.append(moon)

# Renderizar la escena
rt.glRender()

# Bucle principal para mantener la ventana de visualización abierta
isRunning = True
while isRunning:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            isRunning = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                isRunning = False
                
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
