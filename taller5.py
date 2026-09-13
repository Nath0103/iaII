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
    print("Imagen cargada exitosamente para la detección de bordes.")

    # ==========================================
    # 2. OPERADORES DE SOBEL (Derivadas 1er Orden)
    # ==========================================
    # Gradiente Horizontal (Sobel X)
    sobel_x = cv2.Sobel(imagen, cv2.CV_64F, 1, 0, ksize=3)
    sobel_x_abs = cv2.convertScaleAbs(sobel_x)

    # Gradiente Vertical (Sobel Y)
    sobel_y = cv2.Sobel(imagen, cv2.CV_64F, 0, 1, ksize=3)
    sobel_y_abs = cv2.convertScaleAbs(sobel_y)

    # Combinación de Sobel Magnitude: sqrt(Gx^2 + Gy^2)
    sobel_combinado = cv2.addWeighted(sobel_x_abs, 0.5, sobel_y_abs, 0.5, 0)

    # Guardar imágenes de Sobel en la carpeta imgTaller5
    cv2.imwrite('imgTaller5/1_sobel_x.png', sobel_x_abs)
    cv2.imwrite('imgTaller5/2_sobel_y.png', sobel_y_abs)
    cv2.imwrite('imgTaller5/3_sobel_combinado.png', sobel_combinado)

    # ==========================================
    # 3. ALGORITMO CANNY Y EXPERIMENTACIÓN
    # ==========================================
    # Canny Estándar (Umbrales 50 y 150)
    canny_estandar = cv2.Canny(imagen, 50, 150)

    # Canny Permisivo / Sensible (Umbrales bajos: 10 y 50)
    canny_bajo = cv2.Canny(imagen, 10, 50)

    # Canny Estricto (Umbrales altos: 200 y 250)
    canny_alto = cv2.Canny(imagen, 200, 250)

    # Guardar imágenes de Canny en la carpeta imgTaller5
    cv2.imwrite('imgTaller5/4_canny_estandar.png', canny_estandar)
    cv2.imwrite('imgTaller5/5_canny_sensible.png', canny_bajo)
    cv2.imwrite('imgTaller5/6_canny_estricto.png', canny_alto)

    # ==========================================
    # 4. GRAFICAR Y COMPARAR (Matplotlib)
    # ==========================================
    plt.figure(figsize=(15, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(imagen, cmap='gray')
    plt.title("1. Original Grises")
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(sobel_x_abs, cmap='gray')
    plt.title("2. Sobel X (Bordes Verticales)")
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(sobel_y_abs, cmap='gray')
    plt.title("3. Sobel Y (Bordes Horizontales)")
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(canny_estandar, cmap='gray')
    plt.title("4. Algoritmo Canny Estándar (50, 150)")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('imgTaller5/comparativa_sobel_canny.png')

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(canny_bajo, cmap='gray')
    plt.title("Canny Sensible (10, 50)")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(canny_estandar, cmap='gray')
    plt.title("Canny Óptimo/Estándar (50, 150)")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(canny_alto, cmap='gray')
    plt.title("Canny Estricto (200, 250)")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('imgTaller5/comparativa_umbrales_canny.png')

    print("\n¡Proceso del Taller 5 completado exitosamente!")
    print("Archivos de visualización guardados en imgTaller5.")