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
