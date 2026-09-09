import numpy as np

# ==========================================
# LABORATORIO 1: TRANSFORMACIONES AFINES
# (Brillo y Contraste)
# ==========================================

print("=== TALLER DE LABORATORIO 1 ===")

# 1. Crear matriz de prueba 5x5 simulando radiografía sobreexpuesta (valores altos)
matriz_original = np.random.randint(200, 256, (5, 5))

# 2. Aplicar reducción de contraste al 50% (alpha = 0.5) y reducción de brillo (-50)
alpha = 0.5
beta = -50.0
matriz_procesada = (alpha * matriz_original) + beta

# 3. Aplicar np.clip para acotar al rango [0, 255] y convertir a uint8
matriz_procesada = np.clip(matriz_procesada, 0, 255).astype(np.uint8)

# 4. Imprimir ambas matrices para comparar
print("\nMatriz Original (Sobreexpuesta):")
print(matriz_original)

print("\nMatriz Procesada (Ajustada):")
print(matriz_procesada)

# ==========================================
# LABORATORIO FINAL: PROGRAMANDO UN KERNEL
# (Convolución / Filtro 3x3)
# ==========================================

print("\n" + "="*40)
print("=== TALLER DE LABORATORIO FINAL ===")

# 1. Crear en NumPy las matrices "Sección de Imagen (I)" y "Kernel (K)"
I = np.array([
    [100, 100, 100],
    [100, 200, 100],
    [100, 100, 100]
])

K = np.array([
    [ 0, -1,  0],
    [-1,  5, -1],
    [ 0, -1,  0]
])

# 2. Calcular el valor central (Producto Hadamard e/ elemento y suma total)
producto_hadamard = I * K
pixel_central = np.sum(producto_hadamard)

# 3. Imprimir el resultado del píxel central
print(f"\nValor del píxel central calculado: {pixel_central}")