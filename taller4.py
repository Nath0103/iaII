import cv2
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. CARGA DE IMAGEN Y GENERACIÓN DE RUIDO
# ==========================================
ruta_imagen = 'imagenPrueba/leopardoNieve.jpg'
imagen_original = cv2.imread(ruta_imagen)

if imagen_original is None:
    print("Error: No se encontró la imagen en la ruta especificada.")
else:
    print("Imagen original cargada correctamente.")

    # Generar ruido sintético de "Sal y Pimienta" sobre la imagen
    imagen_ruidosa = imagen_original.copy()
    probabilidad_ruido = 0.05  # 5% de píxeles alterados
    
    # Ruido de Sal (píxeles blancos)
    num_sal = np.ceil(probabilidad_ruido * imagen_original.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_sal)) for i in imagen_original.shape[:2]]
    imagen_ruidosa[tuple(coords)] = 255

    # Ruido de Pimienta (píxeles negros)
    num_pimienta = np.ceil(probabilidad_ruido * imagen_original.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_pimienta)) for i in imagen_original.shape[:2]]
    imagen_ruidosa[tuple(coords)] = 0

    # Guardar la imagen ruidosa generada en la carpeta imgTaller4
    cv2.imwrite('imgTaller4/1_imagen_ruidosa.png', imagen_ruidosa)

    # ==========================================
    # 2. APLICACIÓN DE FILTROS (Kernel 7x7)
    # ==========================================
    tamano_kernel = 7

    # Filtro 1: Media (Promedio)
    filtro_media = cv2.blur(imagen_ruidosa, (tamano_kernel, tamano_kernel))

    # Filtro 2: Gaussiano
    filtro_gaussiano = cv2.GaussianBlur(imagen_ruidosa, (tamano_kernel, tamano_kernel), 0)

    # Filtro 3: Mediana
    filtro_mediana = cv2.medianBlur(imagen_ruidosa, tamano_kernel)

    # ==========================================
    # 3. GUARDAR RESULTADOS INDIVIDUALES
    # ==========================================
    cv2.imwrite('imgTaller4/2_filtro_media.png', filtro_media)
    cv2.imwrite('imgTaller4/3_filtro_gaussiano.png', filtro_gaussiano)
    cv2.imwrite('imgTaller4/4_filtro_mediana.png', filtro_mediana)

    # ==========================================
    # 4. MATPLOTLIB: VISUALIZACIÓN COMPARATIVA
    # ==========================================
    img_ruido_rgb = cv2.cvtColor(imagen_ruidosa, cv2.COLOR_BGR2RGB)
    f_media_rgb = cv2.cvtColor(filtro_media, cv2.COLOR_BGR2RGB)
    f_gauss_rgb = cv2.cvtColor(filtro_gaussiano, cv2.COLOR_BGR2RGB)
    f_mediana_rgb = cv2.cvtColor(filtro_mediana, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(16, 10))

    plt.subplot(2, 2, 1)
    plt.imshow(img_ruido_rgb)
    plt.title("1. Con Ruido (Sal y Pimienta)")
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(f_media_rgb)
    plt.title(f"2. Filtro de Media ({tamano_kernel}x{tamano_kernel})")
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(f_gauss_rgb)
    plt.title(f"3. Filtro Gaussiano ({tamano_kernel}x{tamano_kernel})")
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(f_mediana_rgb)
    plt.title(f"4. Filtro de Mediana ({tamano_kernel}x{tamano_kernel})")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('imgTaller4/comparativa_filtros.png')

    print("\n¡Proceso del Taller 4 completado exitosamente!")
    print("Archivos generados en imgTaller4:")
    print(" - 'imgTaller4/1_imagen_ruidosa.png'")
    print(" - 'imgTaller4/2_filtro_media.png'")
    print(" - 'imgTaller4/3_filtro_gaussiano.png'")
    print(" - 'imgTaller4/4_filtro_mediana.png'")
    print(" - 'imgTaller4/comparativa_filtros.png'")