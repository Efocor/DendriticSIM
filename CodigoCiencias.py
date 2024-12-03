#---------------------------------------------------------
# Simulación de DLA avanzada
# Ramo: Ciencias de la Computación
# Objetivo: Implementar una simulación de DLA (agregación limitada por difusión).
# Descripción: La idea es ver patrones de solidificación dendrítica en un sistema de partículas.
#---------------------------------------------------------

#librerías
import pygame
import numpy as np
import random
import colorsys
import time
import os
from datetime import datetime

#inicialización de pygame
pygame.init()

#configuración inicial
DEFAULT_WIDTH, DEFAULT_HEIGHT = 1200, 800
screen = pygame.display.set_mode((DEFAULT_WIDTH, DEFAULT_HEIGHT))
pygame.display.set_caption("Simulación DLA Avanzada")
clock = pygame.time.Clock()

#colores
BACKGROUND_COLOR = (30, 30, 30)
TEXT_COLOR = (255, 255, 255)
INPUT_BOX_COLOR = (50, 50, 50)
HIGHLIGHT_COLOR = (100, 100, 255)
BUTTON_COLOR = (70, 130, 180)
BUTTON_HOVER_COLOR = (100, 149, 237)

#fuentes (de letra)
FONT = pygame.font.SysFont("Arial", 24)
TITLE_FONT = pygame.font.SysFont("Arial", 32, bold=True)

#parámetros iniciales
config = {
    "num_particles": 3000, #número de partículas
    "grid_size": 100, #tamaño de la cuadrícula
    "max_steps": 500, #pasos máximos
    "sticking_prob": 50,  #en porcentaje (50%)
    "wind_direction": 0,  #0: Sin viento, 1: Norte, 2: Oeste, 3: Sur, 4: Este
    "wind_strength": 20,  #en porcentaje (20%)
    "neighbor_type": 4,   #4 o 8. Esto se refiere si se consideran 4 o 8 vecinos
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
    "grid_shape": {"label": "Forma del grid (1:Cuad, 2:Circ)", "value": "1", "pos": (100, 500)},
    "seed_shape": {"label": "Forma semilla (1:Norm, 2:Triang, 3:Circ)", "value": "1", "pos": (100, 550)},
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
        pygame.draw.rect(surface, TEXT_COLOR, self.rect, 2)
        
        text_surface = FONT.render(self.text, True, TEXT_COLOR)
        text_rect = text_surface.get_rect(center=self.rect.center)
        surface.blit(text_surface, text_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        return self.is_hovered and event.type == pygame.MOUSEBUTTONDOWN

def get_color_gradient(distance, max_distance):
    """Genera un color basado en la distancia al centro."""
    hue = (1 - distance/max_distance) * 0.7
    saturation = 1.0
    value = 1.0
    rgb = colorsys.hsv_to_rgb(hue, saturation, value)
    return tuple(int(x * 255) for x in rgb)

#funciones
def calculate_distances(grid, seed_pos):
    """Calcula las distancias desde cada partícula al centro."""
    distances = np.zeros_like(grid, dtype=float)
    height, width = grid.shape
    
    for i in range(height):
        for j in range(width):
            if grid[i, j] == 1:
                distance = np.sqrt((i - seed_pos[0])**2 + (j - seed_pos[1])**2)
                distances[i, j] = distance
    
    return distances

def save_simulation(surface):
    """Guarda la simulación como imagen."""
    if not os.path.exists("simulaciones_dla"):
        os.makedirs("simulaciones_dla", exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"simulaciones_dla/dla_simulation_{timestamp}.png"
    pygame.image.save(surface, filename)
    return filename

def draw_text_with_shadow(surface, text, position, font=FONT, color=TEXT_COLOR, shadow_color=(0, 0, 0)):
    """Dibuja texto con sombra para mejor visibilidad."""
    shadow = font.render(text, True, shadow_color)
    text_surface = font.render(text, True, color)
    surface.blit(shadow, (position[0] + 2, position[1] + 2))
    surface.blit(text_surface, position)

def draw_config_screen():
    """Dibuja la pantalla de configuración."""
    screen.fill(BACKGROUND_COLOR)
    
    #título de la app
    title = "Configuración de Simulación DLA"
    draw_text_with_shadow(screen, title, (DEFAULT_WIDTH//2 - TITLE_FONT.size(title)[0]//2, 50), TITLE_FONT)

    #campos o input de entrada
    for field_name, field in input_fields.items():
        label = field["label"]
        value = field["value"]
        pos = field["pos"]
        
        draw_text_with_shadow(screen, label, pos)
        
        input_rect = pygame.Rect(pos[0] + 300, pos[1], 200, 40)
        color = HIGHLIGHT_COLOR if selected_field == field_name else INPUT_BOX_COLOR
        pygame.draw.rect(screen, color, input_rect)
        pygame.draw.rect(screen, TEXT_COLOR, input_rect, 2)
        
        value_surface = FONT.render(value, True, TEXT_COLOR)
        screen.blit(value_surface, (input_rect.x + 10, input_rect.y + 10))

    #botoncitos
    start_button = Button(DEFAULT_WIDTH//2 - 100, DEFAULT_HEIGHT - 150, 200, 50, 
                         "Iniciar Simulación", BUTTON_COLOR, BUTTON_HOVER_COLOR)
    quit_button = Button(DEFAULT_WIDTH//2 - 100, DEFAULT_HEIGHT - 80, 200, 50,
                        "Salir", BUTTON_COLOR, BUTTON_HOVER_COLOR)

    start_button.draw(screen)
    quit_button.draw(screen)

    pygame.display.flip()
    return start_button, quit_button

def start_simulation(config): 
    """Configura e inicia la simulación DLA."""
    grid_size = config["grid_size"]
    num_particles = config["num_particles"]
    max_steps = config["max_steps"]
    sticking_prob = config["sticking_prob"] / 100  #convertir a rango 0-1
    wind_direction = config["wind_direction"]
    wind_strength = config["wind_strength"] / 100  #idem arriba
    neighbor_type = config["neighbor_type"]

    #asegurar que la fuerza del viento es más eficaz, pero gradual
    wind_factor = wind_strength * (1 + wind_strength)  #escalado para hacerlo más notorio y no solo que en el 100% funcione
    
    screen_size = min(1200, grid_size * 4)
    simulation_screen = pygame.display.set_mode((screen_size, screen_size))
    pygame.display.set_caption("Simulación DLA en Proceso")
    clock = pygame.time.Clock()
    
    #inicialización de la cuadrícula
    grid = [[0] * grid_size for _ in range(grid_size)]
    seed_pos = (grid_size // 2, grid_size // 2)
    grid[seed_pos[0]][seed_pos[1]] = 1

    save_button = Button(10, 10, 120, 40, "Guardar", BUTTON_COLOR, BUTTON_HOVER_COLOR)
    
    particles_added = 0
    running = True

    #paleta de colores pastel
    pastel_colors = [
        (255, 182, 193),  # Rosa Pastel
        (173, 216, 230),  # Azul Pastel
        (144, 238, 144),  # Verde Pastel
        (255, 255, 224),  # Amarillo Pastel
        (219, 112, 147),  # Rosa fuerte
        (216, 191, 216),  # Lavanda Pastel
        (255, 228, 181),  # Durazno claro
        (152, 251, 152),  # Verde claro
        (240, 230, 140),  # Amarillo suave
        (176, 224, 230)   # Azul suave
    ]
    
    while running:
        clock.tick(60)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if save_button.handle_event(event):
                filename = save_simulation(simulation_screen)
                print(f"Simulación guardada como: {filename}")

        #agregar partículas gradualmente
        if particles_added < num_particles:
            for _ in range(min(5, num_particles - particles_added)):  #procesar múltiples partículas por frame
                x, y = random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)
                
                for _ in range(max_steps):
                    #movimiento aleatorio modificado por viento
                    dx, dy = random.choice([-1, 0, 1]), random.choice([-1, 0, 1])

                    #ajuste del viento, ahora con mayor impacto si el viento es fuerte
                    if wind_direction == 1:  #viento hacia la derecha
                        dx += wind_factor
                    elif wind_direction == 2:  #viento hacia abajo
                        dy += wind_factor
                    elif wind_direction == 3:  #viento hacia la izquierda
                        dx -= wind_factor
                    elif wind_direction == 4:  #viento hacia arriba
                        dy -= wind_factor

                    #limitar movimiento por los bordes de la cuadrícula y redondear a enteros
                    x = max(0, min(grid_size - 1, int(x + dx)))  # Asegurar que x sea un entero
                    y = max(0, min(grid_size - 1, int(y + dy)))  # Asegurar que y sea un entero

                    #vecinos (4 u 8) y probabilidad de adhesión
                    if neighbor_type == 4:
                        neighbors = [
                            grid[x-1][y] if x > 0 else 0,
                            grid[x+1][y] if x < grid_size-1 else 0,
                            grid[x][y-1] if y > 0 else 0,
                            grid[x][y+1] if y < grid_size-1 else 0,
                        ]
                    else:  #vecindad de 8
                        neighbors = [
                            grid[nx][ny]
                            for nx in range(max(0, x-1), min(grid_size, x+2))
                            for ny in range(max(0, y-1), min(grid_size, y+2))
                            if (nx != x or ny != y)
                        ]

                    #si la partícula se adhiere
                    if any(neighbors) and random.random() < sticking_prob:
                        grid[x][y] = 1
                        particles_added += 1
                        break

        #visualización
        simulation_screen.fill((0, 0, 0))
        
        #fibujar partículas con colores pastel (por capas)
        pixel_size = screen_size // grid_size
        for i in range(grid_size):
            for j in range(grid_size):
                if grid[i][j] > 0:
                    #calcular la distancia al centro (semilla)
                    dist = ((i - seed_pos[0]) ** 2 + (j - seed_pos[1]) ** 2) ** 0.5
                    max_dist = ((grid_size // 2) ** 2 + (grid_size // 2) ** 2) ** 0.5
                    
                    #definir el número de capas (por ejemplo, 10 capas)
                    num_layers = len(pastel_colors)
                    layer = int((dist / max_dist) * num_layers)
                    color = pastel_colors[layer % num_layers]  #cambiar el color dependiendo de la capa

                    pygame.draw.rect(simulation_screen, color, 
                                     (j * pixel_size, i * pixel_size, pixel_size, pixel_size))
        #interfaz
        save_button.draw(simulation_screen)
        progress = f"Progreso: {particles_added}/{num_particles} partículas"
        draw_text_with_shadow(simulation_screen, progress, (10, 60))

        pygame.display.flip()

    pygame.display.set_mode((DEFAULT_WIDTH, DEFAULT_HEIGHT))

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

def main():
    """Bucle principal del programa con GUI moderna."""
    global selected_field

    # Inicialización base
    pygame.init()
    screen = pygame.display.set_mode((1200, 800))
    pygame.display.set_caption("DLA Simulation Suite")
    clock = pygame.time.Clock()

    # Colores modernos
    COLORS = {
        'background': (15, 15, 25),
        'panel': (25, 28, 40),
        'accent': (65, 105, 225),
        'text': (220, 220, 240),
        'success': (46, 204, 113),
        'warning': (231, 76, 60),
        'input_bg': (35, 38, 50),
        'input_active': (45, 48, 60),
        'button': (65, 105, 225),
        'button_hover': (85, 125, 245)
    }

    # Fuentes
    FONTS = {
        'title': pygame.font.SysFont("Segoe UI", 36, bold=True),
        'subtitle': pygame.font.SysFont("Segoe UI", 24),
        'text': pygame.font.SysFont("Segoe UI", 20),
        'input': pygame.font.SysFont("Segoe UI", 18)
    }

    # Clase para botones modernos
    class ModernButton:
        def __init__(self, x, y, width, height, text, color, hover_color):
            self.rect = pygame.Rect(x, y, width, height)
            self.text = text
            self.color = color
            self.hover_color = hover_color
            self.is_hovered = False

        def draw(self, surface):
            color = self.hover_color if self.is_hovered else self.color
            
            # Dibujar sombra
            shadow_rect = self.rect.copy()
            shadow_rect.y += 2
            pygame.draw.rect(surface, (0, 0, 0, 128), shadow_rect, border_radius=10)
            
            # Dibujar botón
            pygame.draw.rect(surface, color, self.rect, border_radius=10)
            
            # Efecto de brillo en hover
            if self.is_hovered:
                pygame.draw.rect(surface, (255, 255, 255, 30), self.rect, 
                               border_radius=10, width=2)

            # Texto del botón
            text_surf = FONTS['text'].render(self.text, True, COLORS['text'])
            text_rect = text_surf.get_rect(center=self.rect.center)
            surface.blit(text_surf, text_rect)

        def handle_event(self, event):
            if event.type == pygame.MOUSEMOTION:
                self.is_hovered = self.rect.collidepoint(event.pos)
            return self.is_hovered and event.type == pygame.MOUSEBUTTONDOWN

    # Crear botones
    start_button = ModernButton(900, 700, 250, 50, "INICIAR SIMULACIÓN", 
                               COLORS['button'], COLORS['button_hover'])
    help_button = ModernButton(50, 700, 250, 50, "AYUDA", 
                              COLORS['panel'], COLORS['input_active'])

    # Variables de estado
    running = True
    help_visible = False
    error_message = None
    error_timer = 0

    # Posicionamiento de campos
    margin_top = 150
    margin_left = 50
    vertical_spacing = 80
    x, y = margin_left, margin_top

    # Organizar campos en dos columnas
    for key, field in input_fields.items():
        if y + vertical_spacing > 650:
            y = margin_top
            x += 400
        field["pos"] = (x, y)
        y += vertical_spacing

    def draw_gradient_background():
        """Dibuja un fondo con gradiente"""
        for y in range(800):
            darkness = 1 - (y / 800) * 0.3
            color = tuple(int(c * darkness) for c in COLORS['background'])
            pygame.draw.line(screen, color, (0, y), (1200, y))

    def draw_panel(rect, color):
        """Dibuja un panel con sombra y bordes redondeados"""
        shadow_rect = rect.copy()
        shadow_rect.y += 3
        pygame.draw.rect(screen, (0, 0, 0, 128), shadow_rect, border_radius=15)
        pygame.draw.rect(screen, color, rect, border_radius=15)

    def draw_error_message():
        """Dibuja mensaje de error con animación"""
        if error_message and error_timer > 0:
            surf = FONTS['text'].render(error_message, True, COLORS['warning'])
            rect = surf.get_rect(center=(600, 650))
            screen.blit(surf, rect)

    # Bucle principal
    while running:
        # Dibujar fondo
        draw_gradient_background()

        # Título y subtítulo
        title_surf = FONTS['title'].render("Suite Simulador DLA", True, COLORS['text'])
        subtitle_surf = FONTS['subtitle'].render("Diffusion Limited Aggregation", 
                                               True, COLORS['accent'])
        screen.blit(title_surf, (600 - title_surf.get_width()//2, 30))
        screen.blit(subtitle_surf, (600 - subtitle_surf.get_width()//2, 80))

        # Panel principal
        main_panel = pygame.Rect(30, 120, 1140, 560)
        draw_panel(main_panel, COLORS['panel'])

        # Dibujar campos de entrada
        for key, field in input_fields.items():
            # Título del campo
            title_surf = FONTS['text'].render(field["label"], True, COLORS['text'])
            screen.blit(title_surf, (field["pos"][0], field["pos"][1] - 29))

            # Cuadro de entrada
            input_rect = pygame.Rect(field["pos"][0], field["pos"][1], 300, 40)
            color = COLORS['input_active'] if selected_field == key else COLORS['input_bg']
            pygame.draw.rect(screen, color, input_rect, border_radius=8)
            pygame.draw.rect(screen, COLORS['accent'], input_rect, 2, border_radius=8)

            # Valor
            value_surf = FONTS['input'].render(field["value"], True, COLORS['text'])
            screen.blit(value_surf, (input_rect.x + 10, input_rect.y + 10))

        # Dibujar botones
        start_button.draw(screen)
        help_button.draw(screen)

        # Panel de ayuda
        if help_visible:
            help_panel = pygame.Rect(200, 150, 800, 500)
            draw_panel(help_panel, COLORS['panel'])
            
            help_title = FONTS['subtitle'].render("Guía de Configuración", True, COLORS['accent'])
            screen.blit(help_title, (help_panel.centerx - help_title.get_width()//2, 170))

            help_texts = [
                "• Número de partículas: Cantidad total de partículas a simular",
                "• Tamaño de cuadrícula: Dimensiones del espacio de simulación",
                "• Pasos máximos: Límite de movimientos por partícula",
                "• Probabilidad de adhesión: Chance de que una partícula se adhiera (%)",
                "• Dirección del viento: 1: Norte, 2: Sur, 3: Este, 4: Oeste",
                "• Fuerza del viento: Intensidad del efecto del viento (%)",
                "• Tipo de vecinos: 4 para von Neumann, 8 para Moore",
                "No implementadas:",
                "• Forma del grid: 1 para cuadrado, 2 para círculo",
                "• Forma de la semilla: 1 para normal, 2 para triangular, 3 para circular",
                "",
                "Los valores fuera de rango se ajustarán automáticamente.",
            ]

            for i, text in enumerate(help_texts):
                text_surf = FONTS['text'].render(text, True, COLORS['text'])
                screen.blit(text_surf, (220, 220 + i * 30))

        # Mensaje de error
        draw_error_message()
        if error_timer > 0:
            error_timer -= 1

        # Manejo de eventos
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Entrada de texto
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

            # Botones
            if start_button.handle_event(event):
                try:
                    for key in config:
                        value = int(input_fields[key]["value"])
                        if key in ["sticking_prob", "wind_strength"]:
                            if not (0 <= value <= 100):
                                raise ValueError(f"{key} debe estar entre 0 y 100")
                        config[key] = value
                    start_simulation(config)
                except ValueError as e:
                    error_message = f"Error: {str(e)}"
                    error_timer = 180  # 3 segundos a 60 FPS

            if help_button.handle_event(event):
                help_visible = not help_visible

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
