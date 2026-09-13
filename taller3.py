import cv2
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. CARGA DE IMAGEN EN ESCALA DE GRISES
# ==========================================
ruta_imagen = 'imagenPrueba/leopardoNieve.jpg'
imagen = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)

if imagen is None:
    print("Error: No se encontró la imagen en la ruta especificada.")
else:
    print("Imagen cargada exitosamente.")

    # ==========================================
    # 2. UMBRALIZACIÓN (THRESHOLDING)
    # ==========================================
    # Binarización Estática (T = 127)
    _, imagen_binaria = cv2.threshold(imagen, 127, 255, cv2.THRESH_BINARY)
    
    # Binarización de Otsu (Cálculo automático del umbral óptimo)
    T_otsu, imagen_otsu = cv2.threshold(imagen, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(f"El umbral óptimo calculado automáticamente por Otsu fue: {T_otsu:.2f}")

    # Guardar binarización
    cv2.imwrite('imgTaller3/1_imagen_binaria.png', imagen_binaria)
    cv2.imwrite('imgTaller3/1_imagen_otsu.png', imagen_otsu)

    # ==========================================
    # 3. OPERACIONES MORFOLÓGICAS
    # ==========================================
    # Construcción del Elemento Estructurante (Kernel 3x3)
    kernel = np.ones((3, 3), np.uint8)

    # APERTURA (Erosión seguida de Dilatación)
    # Ideal para eliminar ruido blanco pequeño en el fondo (ruido de sal)
    apertura = cv2.morphologyEx(imagen_binaria, cv2.MORPH_OPEN, kernel)

    # CIERRE (Dilatación seguida de Erosión)
    # Ideal para rellenar huecos negros dentro del objeto (ruido de pimienta)
    cierre = cv2.morphologyEx(imagen_binaria, cv2.MORPH_CLOSE, kernel)

    # Guardar resultados
    cv2.imwrite('imgTaller3/2_resultado_apertura.png', apertura)
    cv2.imwrite('imgTaller3/3_resultado_cierre.png', cierre)

    # ==========================================
    # 4. GRAFICAR Y COMPARAR (Matplotlib)
    # ==========================================
    plt.figure(figsize=(15, 5))

    # Imagen Binarizada Original
    plt.subplot(1, 3, 1)
    plt.imshow(imagen_binaria, cmap='gray')
    plt.title("1. Binarizada (T=127)")
    plt.axis('off')

    # Resultado Apertura
    plt.subplot(1, 3, 2)
    plt.imshow(apertura, cmap='gray')
    plt.title("2. Apertura (Erosión + Dilatación)")
    plt.axis('off')

    # Resultado Cierre
    plt.subplot(1, 3, 3)
    plt.imshow(cierre, cmap='gray')
    plt.title("3. Cierre (Dilatación + Erosión)")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('imgTaller3/comparativa_morfologia.png')
    
    print("\n¡Proceso completado con éxito!")
    print("Archivos generados en el explorador:")
    print(" - '1_imagen_binaria.png'")
    print(" - '2_resultado_apertura.png'")
    print(" - '3_resultado_cierre.png'")
    print(" - 'comparativa_morfologia.png' (Visualización comparativa completa)")