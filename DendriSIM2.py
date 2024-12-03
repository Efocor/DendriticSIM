#-------------------------------------------------------------------------------------------
# Simulación de DLA avanzada
# Asignatura: Ciencias de la Computación
# Objetivo: Implementar una simulación de DLA (agregación limitada por difusión).
# Descripción: La idea es ver patrones de solidificación dendrítica en un sistema de partículas.
# Lenguaje: Python
# Autor : Felipe Alexander Correa Rodríguez
#-------------------------------------------------------------------------------------------

"""
El programa se divide en dos partes principales:
1. Interfaz de usuario moderna: Musestra una pantalla de inicio con animaciones simples y una interfaz de usuario moderna para configurar la simulación DLA.
2. Simulación DLA: Implementa la simulación DLA con opciones de configuración avanzadas y visualización en tiempo real.
3. Análisis de datos: Genera análisis detallado de la simulación DLA incluyendo distribución radial, dimensión fractal, anisotropía, densidad de ramificación y métricas adicionales.
(La dimensión fractal es un concepto matemático que se utiliza para describir la complejidad de una estructura fractal, como la simulación DLA.)
(La anisotropía es una medida de la variabilidad direccional de una estructura, en este caso, la distribución de partículas en la simulación DLA.)
(La densidad de ramificación es una medida de la cantidad de ramas o estructuras secundarias en la simulación DLA.)
(Las métricas adicionales incluyen estadísticas generales sobre la simulación DLA, como el número total de partículas, el tiempo total de simulación, la distancia media, la tasa de crecimiento y la dimensión fractal.)
(¡La simulación DLA es un modelo computacional que simula el crecimiento de estructuras dendríticas mediante la agregación limitada por difusión de partículas en un espacio bidimensional!)

La simulación DLA se puede configurar con los siguientes parámetros:
- Número de partículas: Cantidad total de partículas a simular. (3000 por defecto, más partículas hacen más lenta la simulación)
- Tamaño de la cuadrícula: Dimensiones del espacio de simulación. (100 por defecto, más tamaño hace más lenta la simulación y a su vez depende de la resolución del PC)
- Pasos máximos: Límite de movimientos por partícula. (500 por defecto, los pasos son los movimientos que puede hacer una partícula, su cantidad de movimientos)
- Probabilidad de adhesión: Chance de que una partícula se adhiera (%). (50% por defecto, no funciona con valores float o negativos)
- Dirección del viento: 0: Deshabilitado, 1: Norte, 2: Sur, 3: Este, 4: Oeste. (0 por defecto, esto es para la dirección del viento)
- Fuerza del viento: Intensidad del efecto del viento (%). (Hay que escoger una dirección para que se active lo de la la fuerza)
- Tipo de vecinos: 4 para von Neumann, 8 para Moore. (Esto es básicamente si se consideran 4 o 8 vecinos, o sea si se mueve en diagonal o no)
- Forma del grid: 1 para cuadrado, 2 para círculo. (Esto es para la forma del grid, si es cuadrado o círculo)
- Forma de la semilla: 1 para normal, 2 para triangular, 3 para circular. (Esto es para la forma de la semilla, si es normal, triangular o circular)

La simulación se puede guardar como imagen en cualquier momento. La simulación se guarda en la carpeta "simulaciones_dla" en el directorio actual.

Controles:
- Haga clic en un campo de entrada para seleccionarlo.
- Escriba números para cambiar el valor del campo.
- Una vez se termina, cliquear en "Iniciar Simulación" para comenzar la simulación.
- Cliquear en "Ayuda" para ver la guía de configuración.
- Cliquear en "Guardar" en la simulación para guardarla como imagen, puede guardarla en cualquier momento.
- Si cierra la pantalla de simulación, se devuelve a la pantalla de configuración.

Nota: La simulación puede ser lenta con muchos parámetros o partículas, se recomienda ajustar los valores para obtener una simulación más rápida.

*El código está en mitad español y mitad inglés según fue cómodo al programar.

Por Feri.
"""

#--------apertura--------

#librerías
import pygame
import numpy as np
import random
import colorsys
import  math
import sys
import os
from datetime import datetime
import matplotlib.pyplot as plt
import time
import pandas as pd
from scipy import stats
from scipy.stats import gaussian_kde

#inicialización de pygame
pygame.init()
plt.style.use('dark_background')  #para que los gráficos coincidan con el tema oscuro

#configuración inicial
ANCHO_DEFAULT, ALTURA_DEFAULT = 1200, 800
screen = pygame.display.set_mode((ANCHO_DEFAULT, ALTURA_DEFAULT))
pygame.display.set_caption("DendriSIM")
clock = pygame.time.Clock()

#colores
COLOR_BACKGROUND = (30, 30, 30)
COLOR_TEXTO = (255, 255, 255)
CAJAINPUT_COLOR = (50, 50, 50)
COLOR_HIGHLIGHT = (100, 100, 255)
COLOR_BOTON = (70, 130, 180)
COLOR_BOTONHOVER = (100, 149, 237)

#fuentes (de letra)
FONT = pygame.font.SysFont("Arial", 24)
FUENTE_TITULO = pygame.font.SysFont("Arial", 32, bold=True)

#parámetros iniciales
config = {
    "num_particles": 3000, #número de partículas
    "grid_size": 100, #tamaño de la cuadrícula
    "max_steps": 500, #pasos máximos
    "sticking_prob": 50,  #en porcentaje (50%)
    "wind_direction": 0,  #0: Sin viento, 1: Norte, 2: Oeste, 3: Sur, 4: Este
    "wind_strength": 20,  #en porcentaje (20%)
    "neighbor_type": 4,   #4 o 8. Esto se refiere si se consideran 4 o 8 vecinos
    "grid_shape": 1,      #1:cuadrado, 2: círculo
    "seed_shape": 1,      #1:normal, 2: triangular, 3: circular
}

#los campos que permiten cambios
input_fields = {
    "num_particles": {"label": "Número de partículas", "value": "3000", "pos": (100, 150)},
    "grid_size": {"label": "Tamaño de la cuadrícula", "value": "100", "pos": (100, 200)},
    "max_steps": {"label": "Pasos máximos", "value": "500", "pos": (100, 250)},
    "sticking_prob": {"label": "Probabilidad de adhesión (%)", "value": "50", "pos": (100, 300)},
    "wind_direction": {"label": "Dirección del viento (0-4)", "value": "0", "pos": (100, 350)},
    "wind_strength": {"label": "Fuerza del viento (%)", "value": "20", "pos": (100, 400)},
    "neighbor_type": {"label": "Tipo de vecinos (4 u 8)", "value": "4", "pos": (100, 450)},
    "grid_shape": {"label": "Forma del grid", "value": "1", "pos": (100, 500)},
    "seed_shape": {"label": "Forma semilla", "value": "1", "pos": (100, 550)},
}

selected_field = None

class Button:
    def __init__(self, x, y, width, height, text, color, hover_color):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.is_hovered = False

    def draw(self, surface):
        color = self.hover_color if self.is_hovered else self.color
        pygame.draw.rect(surface, color, self.rect)
        pygame.draw.rect(surface, COLOR_TEXTO, self.rect, 2)
        
        text_surface = FONT.render(self.text, True, COLOR_TEXTO)
        text_rect = text_surface.get_rect(center=self.rect.center)
        surface.blit(text_surface, text_rect)

    def eventoaccion(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        return self.is_hovered and event.type == pygame.MOUSEBUTTONDOWN

def gradientecolores(distance, max_distance):
    """Genera un color basado en la distancia al centro."""
    hue = (1 - distance/max_distance) * 0.7
    saturation = 1.0
    value = 1.0
    rgb = colorsys.hsv_to_rgb(hue, saturation, value)
    return tuple(int(x * 255) for x in rgb)

#funciones
def distancalcu(grid, seed_pos):
    """Calcula las distancias desde cada partícula al centro."""
    distancias = np.zeros_like(grid, dtype=float)
    height, width = grid.shape
    
    for i in range(height):
        for j in range(width):
            if grid[i, j] == 1:
                distance = np.sqrt((i - seed_pos[0])**2 + (j - seed_pos[1])**2)
                distancias[i, j] = distance
    
    return distancias

def guardasimu(surface):
    """Guarda la simulación como imagen."""
    if not os.path.exists("simulaciones_dla"):
        os.makedirs("simulaciones_dla", exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"simulaciones_dla/dla_simulation_{timestamp}.png"
    pygame.image.save(surface, filename)
    return filename

def textosombra(surface, text, position, font=FONT, color=COLOR_TEXTO, shadow_color=(0, 0, 0)):
    """Dibuja texto con sombra para mejor visibilidad."""
    shadow = font.render(text, True, shadow_color)
    text_surface = font.render(text, True, color)
    surface.blit(shadow, (position[0] + 2, position[1] + 2))
    surface.blit(text_surface, position)

def pantallaconfig():
    """Dibuja la pantalla de configuración."""
    screen.fill(COLOR_BACKGROUND)
    
    #título de la app
    title = "Configuración de Simulación DLA"
    textosombra(screen, title, (ANCHO_DEFAULT//2 - FUENTE_TITULO.size(title)[0]//2, 50), FUENTE_TITULO)

    #campos o input de entrada
    for field_name, field in input_fields.items():
        label = field["label"]
        value = field["value"]
        pos = field["pos"]
        
        textosombra(screen, label, pos)
        
        input_rect = pygame.Rect(pos[0] + 300, pos[1], 200, 40)
        color = COLOR_HIGHLIGHT if selected_field == field_name else CAJAINPUT_COLOR
        pygame.draw.rect(screen, color, input_rect)
        pygame.draw.rect(screen, COLOR_TEXTO, input_rect, 2)
        
        value_surface = FONT.render(value, True, COLOR_TEXTO)
        screen.blit(value_surface, (input_rect.x + 10, input_rect.y + 10))

    #botoncitos
    boton_start = Button(ANCHO_DEFAULT//2 - 100, ALTURA_DEFAULT - 150, 200, 50, 
                         "Iniciar Simulación", COLOR_BOTON, COLOR_BOTONHOVER)
    quit_button = Button(ANCHO_DEFAULT//2 - 100, ALTURA_DEFAULT - 80, 200, 50,
                        "Salir", COLOR_BOTON, COLOR_BOTONHOVER)

    boton_start.draw(screen)
    quit_button.draw(screen)

    pygame.display.flip()
    return boton_start, quit_button

#----------------------------------------------------------------------------

def genera_graficos_analisis(particle_positions, growth_times, grid_size):
    """
    Genera análisis detallado de la simulación DLA incluyendo:
    - Distribución radial de partículas
    - Análisis de crecimiento temporal
    - Dimensión fractal
    - Análisis de anisotropía
    - Densidad de ramificación
    """
    #guardar el backend actual y cambiar a 'Agg' (no interactivo)
    current_backend = plt.get_backend()
    plt.switch_backend('Agg')
    
    #configuración general de estilo
    plt.style.use('dark_background')
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'figure.titlesize': 16
    })

    #crea figura sin mostrarla, solo para hacer el .png
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('Análisis detallado del crecimiento dendrítico', fontsize=16, y=0.95)

    #calcular datos básicos
    center = grid_size // 2
    distances = [np.sqrt((x-center)**2 + (y-center)**2) for x, y in particle_positions]
    angles = [np.arctan2(y-center, x-center) for x, y in particle_positions]
    
    #1. distribución Radial Mejorada
    ax1 = plt.subplot(231)
    hist, bins, _ = ax1.hist(distances, bins=40, density=True, alpha=0.7, 
                            color='skyblue', edgecolor='white')
    #ajuste de curva
    kde = gaussian_kde(distances)
    x_range = np.linspace(min(distances), max(distances), 100)
    ax1.plot(x_range, kde(x_range), 'r-', lw=2, label='KDE')
    ax1.set_title('Distribución Radial')
    ax1.set_xlabel('Distancia desde el centro')
    ax1.set_ylabel('Densidad')
    ax1.legend()

    #2. análisis temporal de crecimiento
    ax2 = plt.subplot(232)
    growth_rate = np.diff([0] + growth_times)
    ax2.plot(range(len(growth_rate)), growth_rate, 'g-', alpha=0.5, label='Tasa instantánea')
    ax2.plot(range(len(growth_rate)), 
             pd.Series(growth_rate).rolling(window=20).mean(), 
             'w-', lw=2, label='Media móvil')
    ax2.set_title('Tasa de Crecimiento Temporal')
    ax2.set_xlabel('Número de partícula')
    ax2.set_ylabel('Tiempo entre adhesiones (s)')
    ax2.legend()

    #3. dimensión fractal mejorada
    ax3 = plt.subplot(233)
    radii = np.logspace(0, np.log10(grid_size//2), 50)
    particles_in_radius = [sum(1 for d in distances if d <= r) for r in radii]
    
    #calcular dimensión fractal
    from scipy import stats
    slope, intercept, r_value, _, _ = stats.linregress(np.log(radii), 
                                                      np.log(particles_in_radius))
    
    ax3.loglog(radii, particles_in_radius, 'bo-', label='Datos', alpha=0.6)
    ax3.loglog(radii, np.exp(intercept + slope * np.log(radii)), 'r--', 
               label=f'D = {abs(slope):.2f}')
    ax3.set_title(f'Dimensión Fractal\nD = {abs(slope):.2f}')
    ax3.set_xlabel('Radio (log)')
    ax3.set_ylabel('Partículas acumuladas (log)')
    ax3.legend()

    #4. análisis de anisotropía
    ax4 = plt.subplot(234, projection='polar')
    hist, bins = np.histogram(angles, bins=36)
    center_bins = (bins[:-1] + bins[1:]) / 2
    ax4.plot(center_bins, hist, '-', color='cyan')
    ax4.fill_between(center_bins, 0, hist, alpha=0.3, color='cyan')
    ax4.set_title('Análisis de Anisotropía')
    
    #5. densidad local vs distancia
    ax5 = plt.subplot(235)
    bin_edges = np.linspace(0, max(distances), 30)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    hist, _ = np.histogram(distances, bins=bin_edges)
    density = hist / (2 * np.pi * bin_centers)
    ax5.plot(bin_centers, density, 'y-', lw=2)
    ax5.set_title('Densidad Local vs Distancia')
    ax5.set_xlabel('Distancia desde el centro')
    ax5.set_ylabel('Densidad local')

    #6. estadísticas Generales
    ax6 = plt.subplot(236)
    ax6.axis('off')
    stats_text = [
        f"Total de partículas: {len(particle_positions)}",
        f"Tiempo total: {growth_times[-1]:.2f} s",
        f"Radio máximo: {max(distances):.2f}",
        f"Dimensión fractal: {abs(slope):.2f}",
        f"Tasa promedio: {len(particle_positions)/growth_times[-1]:.2f} part/s",
        f"Distancia media: {np.mean(distances):.2f}",
        f"Desviación estándar: {np.std(distances):.2f}"
    ]
    ax6.text(0.05, 0.95, '\n'.join(stats_text), 
             transform=ax6.transAxes, 
             verticalalignment='top',
             fontsize=10)
    ax6.set_title('Estadísticas Generales')

    #ajustes finales
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    #guardar
    if not os.path.exists("simulaciones_dla"):
        os.makedirs("simulaciones_dla", exist_ok=True)
    plt.savefig(f'simulaciones_dla/analisis_dla_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png', dpi=300, bbox_inches='tight')

    #cerrar el coso
    plt.close(fig)
    
    # Restaurar el backend original
    plt.switch_backend(current_backend)
    
    #generar informe adicional de métricas
    genera_informe_metricas(particle_positions, growth_times, distances, slope)

def genera_informe_metricas(particle_positions, growth_times, distances, dim_fractal):
    """Genera un archivo de texto con métricas adicionales."""
    if not os.path.exists("simulaciones_dla"):
        os.makedirs("simulaciones_dla", exist_ok=True)
    with open(f"simulaciones_dla/metricas_dla_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", "w") as f:
        f.write("ANÁLISIS DETALLADO DE LA SIMULACIÓN DLA\n")
        f.write("=====================================\n\n")
        
        #métricas básicas
        f.write(f"Número total de partículas: {len(particle_positions)}\n")
        f.write(f"Tiempo total de simulación: {growth_times[-1]:.2f} segundos\n")
        f.write(f"Dimensión fractal: {abs(dim_fractal):.3f}\n\n")
        
        #métricas de distancia
        f.write("MÉTRICAS DE DISTANCIA\n")
        f.write(f"Distancia media: {np.mean(distances):.2f}\n")
        f.write(f"Distancia máxima: {max(distances):.2f}\n")
        f.write(f"Desviación estándar: {np.std(distances):.2f}\n\n")
        
        #métricas de crecimiento
        growth_rates = np.diff([0] + growth_times)
        f.write("MÉTRICAS DE CRECIMIENTO\n")
        f.write(f"Tasa promedio: {len(particle_positions)/growth_times[-1]:.2f} partículas/s\n")
        f.write(f"Tiempo promedio entre adhesiones: {np.mean(growth_rates):.3f} s\n")
        f.write(f"Variación en tasa de crecimiento: {np.std(growth_rates):.3f} s\n")
        
        f.write('Explicación de las métricas\n')
        f.write('=========================\n\n')
        f.write('Número total de partículas: Cantidad total de partículas en la simulación.\n')
        f.write('Tiempo total de simulación: Duración total de la simulación en segundos.\n')
        f.write('Dimensión fractal: La dimensión fractal es sobre la estructura de la simulación, es decir cuán fractal es.\n\n')
        f.write('Distancia media: Distancia media de las partículas al centro.\n')
        f.write('Distancia máxima: Distancia máxima de las partículas al centro.\n')
        f.write('Desviación estándar: Desviación estándar de las distancias al centro.\n\n')
        f.write('Tasa promedio: Tasa promedio de adhesión de partículas por segundo.\n')
        f.write('Tiempo promedio entre adhesiones: Tiempo promedio entre adhesiones sucesivas.\n')
        f.write('Variación en tasa de crecimiento: Variación en el tiempo entre adhesiones.\n')
        pass
        
        print(f"Informe de métricas guardado en: {f.name}")
    
#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------

#simulación DLA
def comienzasimu(config):
    """Configura e inicia la simulación DLA con coloreado por tiempo y análisis de datos."""
    grid_size = config["grid_size"]
    num_particles = config["num_particles"]
    max_steps = config["max_steps"]
    sticking_prob = config["sticking_prob"] / 100
    wind_direction = config["wind_direction"]
    wind_strength = config["wind_strength"] / 100
    neighbor_type = config["neighbor_type"]
    grid_shape = config["grid_shape"]
    seed_shape = config["seed_shape"]
    
    wind_factor = wind_strength * (1 + wind_strength)
    screen_size = min(1200, grid_size * 4)
    simulation_screen = pygame.display.set_mode((screen_size, screen_size))
    pygame.display.set_caption("Simulación DLA en Proceso")
    clock = pygame.time.Clock()
    
    #inicialización de grids
    grid = [[0] * grid_size for _ in range(grid_size)]
    time_grid = [[0] * grid_size for _ in range(grid_size)]
    center = grid_size // 2
    
    #datos para análisis
    particle_positions = []
    growth_times = []
    start_time = time.time()
    
    #crear máscara para grid circular
    if grid_shape == 2:
        grid_radius = grid_size // 2
        for i in range(grid_size):
            for j in range(grid_size):
                if ((i - center)**2 + (j - center)**2) > grid_radius**2:
                    grid[i][j] = -1

    #configurar semilla según forma elegida
    if seed_shape == 1:
        grid[center][center] = 1
        time_grid[center][center] = 1
        particle_positions.append((center, center))
        growth_times.append(0)
    elif seed_shape == 2:
        for x, y in [(0,0), (1,0), (0,-1), (0,1)]:
            grid[center+x][center+y] = 1
            time_grid[center+x][center+y] = 1
            particle_positions.append((center+x, center+y))
            growth_times.append(0)
    elif seed_shape == 3:
        for i in range(-1, 2):
            for j in range(-1, 2):
                if abs(i) + abs(j) <= 1:
                    grid[center+i][center+j] = 1
                    time_grid[center+i][center+j] = 1
                    particle_positions.append((center+i, center+j))
                    growth_times.append(0)

    #botones
    save_button = Button(10, 10, 120, 40, "Guardar", COLOR_BOTON, COLOR_BOTONHOVER)
    stats_button = Button(140, 10, 120, 40, "Analizar", COLOR_BOTON, COLOR_BOTONHOVER)
    particles_added = len(particle_positions)
    running = True

    #configura colormap
    color_map = plt.cm.viridis

    while running:
        clock.tick(60)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if save_button.eventoaccion(event):
                filename = guardasimu(simulation_screen)
                print(f"Simulación guardada como: {filename}")
            if stats_button.eventoaccion(event):
                genera_graficos_analisis(particle_positions, growth_times, grid_size)
                
        if particles_added < num_particles:
            for _ in range(min(5, num_particles - particles_added)):
                #genera posición inicial válida
                while True:
                    x, y = random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)
                    if grid_shape == 1 or (grid_shape == 2 and 
                        ((x - center)**2 + (y - center)**2) <= (grid_size//2)**2):
                        break
                
                for _ in range(max_steps):
                    dx, dy = random.choice([-1, 0, 1]), random.choice([-1, 0, 1])
                    
                    #aplicar efecto del viento
                    if wind_direction == 1:
                        dx += wind_factor
                    elif wind_direction == 2:
                        dy += wind_factor
                    elif wind_direction == 3:
                        dx -= wind_factor
                    elif wind_direction == 4:
                        dy -= wind_factor
                    
                    new_x = max(0, min(grid_size - 1, int(x + dx)))
                    new_y = max(0, min(grid_size - 1, int(y + dy)))
                    
                    if grid_shape == 2 and grid[new_x][new_y] == -1:
                        continue
                    
                    x, y = new_x, new_y
                    
                    #verificar vecinos según tipo
                    if neighbor_type == 4:
                        neighbors = [
                            grid[x-1][y] if x > 0 else 0,
                            grid[x+1][y] if x < grid_size-1 else 0,
                            grid[x][y-1] if y > 0 else 0,
                            grid[x][y+1] if y < grid_size-1 else 0,
                        ]
                    else:
                        neighbors = [
                            grid[nx][ny]
                            for nx in range(max(0, x-1), min(grid_size, x+2))
                            for ny in range(max(0, y-1), min(grid_size, y+2))
                            if (nx != x or ny != y) and grid[nx][ny] != -1
                        ]
                    
                    if any(n == 1 for n in neighbors) and random.random() < sticking_prob:
                        grid[x][y] = 1
                        current_time = time.time() - start_time
                        time_grid[x][y] = particles_added + 1
                        particle_positions.append((x, y))
                        growth_times.append(current_time)
                        particles_added += 1
                        break

        #renderizado
        simulation_screen.fill((0, 0, 0))
        pixel_size = screen_size // grid_size
        
        for i in range(grid_size):
            for j in range(grid_size):
                if grid[i][j] == 1:
                    #color basado en orden de adhesión
                    normalized_time = time_grid[i][j] / num_particles
                    color_rgba = color_map(normalized_time)
                    color = tuple(int(c * 255) for c in color_rgba[:3])
                    pygame.draw.rect(simulation_screen, color,
                                   (j * pixel_size, i * pixel_size, pixel_size, pixel_size))
                elif grid[i][j] == -1 and grid_shape == 2:
                    pygame.draw.rect(simulation_screen, (35, 35, 35),
                                   (j * pixel_size, i * pixel_size, pixel_size, pixel_size))

        save_button.draw(simulation_screen)
        stats_button.draw(simulation_screen)
        progress = f"Progreso: {particles_added}/{num_particles} partículas"
        textosombra(simulation_screen, progress, (10, 60))
        
        pygame.display.flip()
    
    pygame.display.set_mode((ANCHO_DEFAULT, ALTURA_DEFAULT))
    return particle_positions, growth_times

#-------------------------------------------------------------------------------------------

def handle_input(event):
    """Maneja la entrada del usuario."""
    global selected_field

    if event.type == pygame.MOUSEBUTTONDOWN:
        mouse_pos = pygame.mouse.get_pos()
        for field_name, field in input_fields.items():
            #define el área clicable del campo
            field_rect = pygame.Rect(field["pos"][0], field["pos"][1], 300, 40)
            if field_rect.collidepoint(mouse_pos):
                selected_field = field_name
                return
        selected_field = None

    if event.type == pygame.KEYDOWN and selected_field:
        if event.key == pygame.K_BACKSPACE:
            #elimina el último carácter
            input_fields[selected_field]["value"] = input_fields[selected_field]["value"][:-1]
        elif event.unicode.isdigit():
            #agregar un dígito al valor actual
            input_fields[selected_field]["value"] += event.unicode

#presentación
def welcomefe(screen, COLORS, FONTS):
    """Muestra una pantalla de inicio profesional con animaciones."""
    running = True
    alpha = 0  #para fade in
    logo_scale = 0  #para animación del logo
    posicion_particulas = [(random.randint(0, 1200), random.randint(0, 800)) for _ in range(50)]
    speed_particulas = [(random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)) for _ in range(50)]
    
    start_time = pygame.time.get_ticks()
    
    def dibujaparticulas():
        for i, (pos, speed) in enumerate(zip(posicion_particulas, speed_particulas)):
            x, y = pos
            posicion_particulas[i] = ((x + speed[0]) % 1200, (y + speed[1]) % 800)
            pygame.draw.circle(screen, (65, 105, 225, min(alpha, 128)), 
                             (int(x), int(y)), 2)
    
    while running:
        current_time = pygame.time.get_ticks()
        elapsed_time = current_time - start_time
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and elapsed_time > 1500:
                    return
        
        #fondo con gradiente
        for y in range(800):
            darkness = 1 - (y / 800) * 0.3
            color = tuple(int(c * darkness) for c in COLORS['background'])
            pygame.draw.line(screen, color, (0, y), (1200, y))
        
        #actualizar animaciones
        if alpha < 255:
            alpha += 5
        if logo_scale < 1:
            logo_scale += 0.05
        
        #dibuja partículas de fondo
        dibujaparticulas()
        
        #dibuja logo y textos con efectos
        title_surf = FONTS['title'].render("DendriSIM", True, COLORS['text'])
        subtitle_surf = FONTS['subtitle'].render("Simulador de Agregación Limitada por Difusión", 
                                               True, COLORS['accent'])
        version_surf = FONTS['text'].render("v2.1", True, COLORS['text'])
        
        #aplicar escala al logo
        scaled_title = pygame.transform.scale(title_surf, 
                                            (int(title_surf.get_width() * logo_scale),
                                             int(title_surf.get_height() * logo_scale)))
        
        #centra los elementos
        title_rect = scaled_title.get_rect(center=(600, 300))
        subtitle_rect = subtitle_surf.get_rect(center=(600, 380))
        version_rect = version_surf.get_rect(bottomright=(1150, 750))
        
        #aplica transparencia
        scaled_title.set_alpha(alpha)
        subtitle_surf.set_alpha(alpha)
        version_surf.set_alpha(alpha)
        
        #bibuja elementos
        screen.blit(scaled_title, title_rect)
        screen.blit(subtitle_surf, subtitle_rect)
        screen.blit(version_surf, version_rect)
        
        #muestra mensaje de continuar después de la animación inicial
        if elapsed_time > 1500:
            continue_surf = FONTS['text'].render("Presiona ESPACIO para continuar", 
                                           True, COLORS['text'])
            continue_rect = continue_surf.get_rect(center=(600, 500))
            continue_surf.set_alpha(int(127 + 127 * math.sin(elapsed_time * 0.003)))
            screen.blit(continue_surf, continue_rect)
        
        pygame.display.flip()
        pygame.time.Clock().tick(60)

#-------------------------------------------------------------------------------------------
#lógica del programa
#-------------------------------------------------------------------------------------------

def main():
    """Bucle principal del programa con GUI moderna."""
    global selected_field

    #inicialización base
    pygame.init()
    screen = pygame.display.set_mode((1200, 800))
    pygame.display.set_caption("DendriSIM")
    clock = pygame.time.Clock()

    #colores modernos para mostrar cuando aparece GUI
    COLORS = {
        'background': (15, 15, 25),
        'panel': (25, 28, 40),
        'panel2': (50, 56, 80),
        'accent': (65, 105, 225),
        'text': (220, 220, 240),
        'success': (46, 204, 113),
        'warning': (231, 76, 60),
        'input_bg': (35, 38, 50),
        'input_active': (45, 48, 60),
        'button': (65, 105, 225),
        'button_hover': (85, 125, 245)
    }

    #fuentes, o sea la letra
    FONTS = {
        'title': pygame.font.SysFont("Segoe UI", 36, bold=True),
        'subtitle': pygame.font.SysFont("Segoe UI", 24),
        'text': pygame.font.SysFont("Segoe UI", 20),
        'input': pygame.font.SysFont("Segoe UI", 18)
    }

    #clase para botones modernos
    class ModernButton:
        def __init__(self, x, y, width, height, text, color, hover_color):
            self.rect = pygame.Rect(x, y, width, height)
            self.text = text
            self.color = color
            self.hover_color = hover_color
            self.is_hovered = False

        def draw(self, surface):
            color = self.hover_color if self.is_hovered else self.color
            
            #dibujar sombra
            shadow_rect = self.rect.copy()
            shadow_rect.y += 2
            pygame.draw.rect(surface, (0, 0, 0, 128), shadow_rect, border_radius=10)
            
            #dibujar botón
            pygame.draw.rect(surface, color, self.rect, border_radius=10)
            
            #efectito de brillo en hover
            if self.is_hovered:
                pygame.draw.rect(surface, (255, 255, 255, 30), self.rect, 
                               border_radius=10, width=2)

            #texto del botón
            text_surf = FONTS['text'].render(self.text, True, COLORS['text'])
            text_rect = text_surf.get_rect(center=self.rect.center)
            surface.blit(text_surf, text_rect)

        def eventoaccion(self, event):
            if event.type == pygame.MOUSEMOTION:
                self.is_hovered = self.rect.collidepoint(event.pos)
            return self.is_hovered and event.type == pygame.MOUSEBUTTONDOWN

    #crear botones
    boton_start = ModernButton(900, 700, 250, 50, "INICIAR SIMULACIÓN", 
                               COLORS['button'], COLORS['button_hover'])
    boton_ayuda = ModernButton(50, 700, 250, 50, "AYUDA", 
                              COLORS['panel'], COLORS['input_active'])

    #variables de estado
    running = True
    help_visible = False
    error_message = None
    error_timer = 0

    #posicionamiento de campos
    margin_top = 150
    margin_left = 50
    vertical_spacing = 80
    x, y = margin_left, margin_top

    #organizar campos en dos columnas
    for key, field in input_fields.items():
        if y + vertical_spacing > 650:
            y = margin_top
            x += 400
        field["pos"] = (x, y)
        y += vertical_spacing

    def backgroundengradiente():
        """Dibuja un fondo con gradiente"""
        for y in range(800):
            darkness = 1 - (y / 800) * 0.3
            color = tuple(int(c * darkness) for c in COLORS['background'])
            pygame.draw.line(screen, color, (0, y), (1200, y))

    def panelesprint(rect, color):
        """Dibuja un panel con sombra y bordes redondeados"""
        shadow_rect = rect.copy()
        shadow_rect.y += 3
        pygame.draw.rect(screen, (0, 0, 0, 128), shadow_rect, border_radius=15)
        pygame.draw.rect(screen, color, rect, border_radius=15)

    def mensajitodeerror():
        """Dibuja mensaje de error con animación"""
        if error_message and error_timer > 0:
            surf = FONTS['text'].render(error_message, True, COLORS['warning'])
            rect = surf.get_rect(center=(600, 650))
            screen.blit(surf, rect)

    welcomefe(screen, COLORS, FONTS)
        
    #bucle principal
    while running:
        # Dibujar fondo
        backgroundengradiente()

        #título y subtítulo
        title_surf = FONTS['title'].render("DendriSIM", True, COLORS['text'])
        subtitle_surf = FONTS['subtitle'].render("Diffusion Limited Aggregation", 
                                               True, COLORS['accent'])
        screen.blit(title_surf, (600 - title_surf.get_width()//2, 30))
        screen.blit(subtitle_surf, (600 - subtitle_surf.get_width()//2, 80))

        #panel principal
        main_panel = pygame.Rect(30, 120, 1140, 560)
        panelesprint(main_panel, COLORS['panel'])

        #printea campos de entrada
        for key, field in input_fields.items():
            #título del campo
            title_surf = FONTS['text'].render(field["label"], True, COLORS['text'])
            screen.blit(title_surf, (field["pos"][0], field["pos"][1] - 29))

            #cuadro de entrada
            input_rect = pygame.Rect(field["pos"][0], field["pos"][1], 300, 40)
            color = COLORS['input_active'] if selected_field == key else COLORS['input_bg']
            pygame.draw.rect(screen, color, input_rect, border_radius=8)
            pygame.draw.rect(screen, COLORS['accent'], input_rect, 2, border_radius=8)

            #valor
            value_surf = FONTS['input'].render(field["value"], True, COLORS['text'])
            screen.blit(value_surf, (input_rect.x + 10, input_rect.y + 10))

        #dibujar botones
        boton_start.draw(screen)
        boton_ayuda.draw(screen)

        #panel de ayuda
        if help_visible:
            help_panel = pygame.Rect(200, 150, 800, 500)
            panelesprint(help_panel, COLORS['panel2'])
            
            help_title = FONTS['subtitle'].render("Guía de Configuración", True, COLORS['accent'])
            screen.blit(help_title, (help_panel.centerx - help_title.get_width()//2, 170))

            help_texts = [
                "• Número de partículas: Cantidad total de partículas a simular",
                "• Tamaño de cuadrícula: Dimensiones del espacio de simulación",
                "• Pasos máximos: Límite de movimientos por partícula",
                "• Probabilidad de adhesión: Chance de que una partícula se adhiera (%)",
                "• Dirección del viento: 0: Deshabilitado 1: Norte, 2: Sur, 3: Este, 4: Oeste",
                "• Fuerza del viento: Intensidad del efecto del viento (%)",
                "• Tipo de vecinos: 4 para von Neumann, 8 para Moore",
                "Especiales:",
                "• Forma del grid: 1 para cuadrado, 2 para círculo",
                "• Forma de la semilla: 1 para normal, 2 para triangular, 3 para circular",
                "",
                "Los valores fuera de rango se ajustarán automáticamente.",
            ]

            for i, text in enumerate(help_texts):
                text_surf = FONTS['text'].render(text, True, COLORS['text'])
                screen.blit(text_surf, (220, 220 + i * 30))

        #mensaje de error
        mensajitodeerror()
        if error_timer > 0:
            error_timer -= 1

        #manejo de eventos
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            #entrada de texto
            if event.type == pygame.MOUSEBUTTONDOWN:
                selected_field = None
                pos = event.pos
                for key, field in input_fields.items():
                    input_rect = pygame.Rect(field["pos"][0], field["pos"][1], 300, 40)
                    if input_rect.collidepoint(pos):
                        selected_field = key

            if event.type == pygame.KEYDOWN and selected_field:
                if event.key == pygame.K_BACKSPACE:
                    input_fields[selected_field]["value"] = input_fields[selected_field]["value"][:-1]
                elif event.key == pygame.K_RETURN:
                    selected_field = None
                elif event.unicode.isnumeric() or event.unicode == '.':
                    input_fields[selected_field]["value"] += event.unicode

            #botones
            if boton_start.eventoaccion(event):
                try:
                    for key in config:
                        value = int(input_fields[key]["value"])
                        if key in ["sticking_prob", "wind_strength"]:
                            if not (0 <= value <= 100):
                                raise ValueError(f"{key} debe estar entre 0 y 100")
                        config[key] = value
                    comienzasimu(config)
                except ValueError as e:
                    error_message = f"Error: {str(e)}"
                    error_timer = 180  #3 segundos a 60 FPS

            if boton_ayuda.eventoaccion(event):
                help_visible = not help_visible

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
