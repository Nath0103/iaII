import cv2
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# TALLER DE LABORATORIO 1: TRANSFORMACIÓN
# ==========================================

# 1. Crear el píxel BGR completamente amarillo (Azul=0, Verde=255, Rojo=255)
pixel = np.array([0, 255, 255], dtype=np.float32)

# 2. Pesos ponderados para el orden BGR (0.114*B + 0.587*G + 0.299*R)
pesos = np.array([0.114, 0.587, 0.299])

# Producto punto entre el píxel y los pesos
gris_calculado = np.dot(pixel, pesos)

# 3. Imprimir el resultado manual
print(f"1. Valor de intensidad en gris para amarillo puro: {gris_calculado:.2f}")

# 4. Verificación con una imagen real descargada usando OpenCV
ruta_imagen = 'imagenPrueba/leopardoNieve.jpg'
imagen = cv2.imread(ruta_imagen)

if imagen is not None:
    # Conversión mediante la función optimizada de OpenCV
    img_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    print("\n2. Conversión a grises en OpenCV realizada con éxito.")
    print("   - Dimensiones de imagen original (Alto, Ancho, Canales):", imagen.shape)
    print("   - Dimensiones de imagen en gris (Alto, Ancho):", img_gris.shape)
    
    # En Codespaces guardamos la imagen convertida para verla en el explorador
    cv2.imwrite('imgTaller2/imagen_gris_resultado.png', img_gris)
    print("   - Se guardó 'imagen_gris_resultado.png' en tus archivos.")
else:
    print("\nError: No se encontró la imagen en la ruta especificada.")

# ==========================================
# TALLER DE LABORATORIO 2: ANÁLISIS ESTADÍSTICO
# ==========================================

if imagen is not None:
    # Configurar colores y etiquetas para cada canal BGR
    colores = ('b', 'g', 'r')
    etiquetas = ('Canal Azul', 'Canal Verde', 'Canal Rojo')
    
    plt.figure(figsize=(10, 5))
    
    # Calcular y graficar el histograma de cada canal
    for i, col in enumerate(colores):
        hist = cv2.calcHist([imagen], [i], None, [256], [0, 256])
        plt.plot(hist, color=col, label=etiquetas[i])
        plt.xlim([0, 256])

    plt.title("Histograma Comparativo de Canales RGB")
    plt.xlabel("Valor del Píxel (0 - 255)")
    plt.ylabel("Frecuencia (Cantidad de Píxeles)")
    plt.legend()
    plt.grid(True)
    
    # Guardar la gráfica como imagen para poder abrirla desde VS Code
    plt.savefig('imgTaller2/histograma_resultado.png')
    print("\n3. ¡Proceso completado!")
    print("   - La gráfica del histograma se guardó como 'histograma_resultado.png'.")