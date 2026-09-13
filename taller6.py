import cv2
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. PIPELINE DE PREPROCESAMIENTO
# ==========================================
ruta_imagen = 'imagenPrueba/prueba.jpg'

# Cargar imagen en color y en grises
imagen_color = cv2.imread(ruta_imagen)

if imagen_color is None:
    print("Error: No se pudo cargar la imagen en la ruta especificada.")
else:
    imagen_gris = cv2.cvtColor(imagen_color, cv2.COLOR_BGR2GRAY)

    # 1. Umbralización de Otsu para separar objeto del fondo
    _, imagen_binaria = cv2.threshold(imagen_gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 2. Limpieza Morfológica (Apertura) para eliminar ruido
    kernel = np.ones((3, 3), np.uint8)
    imagen_limpia = cv2.morphologyEx(imagen_binaria, cv2.MORPH_OPEN, kernel)

    # ==========================================
    # 2. EXTRACCIÓN DE CONTORNOS Y CARACTERÍSTICAS
    # ==========================================
    # Encontrar contornos externos
    contornos, _ = cv2.findContours(imagen_limpia, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Copia para dibujar los resultados
    resultado = imagen_color.copy()

    # Umbral empresarial de área (en píxeles) para clasificar objetos
    UMBRAL_AREA_GRANDE = 2000

    print("--- ANÁLISIS DE OBJETOS ENCONTRADOS ---")
    
    contador_objetos = 0
    for i, cnt in enumerate(contornos):
        # 1. Calcular el área del contorno
        area = cv2.contourArea(cnt)

        # Filtrar ruido menor (objetos diminutos no deseados)
        if area > 100:
            contador_objetos += 1
            
            # 2. Calcular Bounding Box (Rectángulo envolvente)
            x, y, w, h = cv2.boundingRect(cnt)

            # 3. Calcular Centroide mediante Momentos
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = x + w // 2, y + h // 2

            # 4. Lógica de clasificación por área
            if area > UMBRAL_AREA_GRANDE:
                color_box = (255, 0, 0)  # Azul en BGR (Objeto Grande)
                categoria = "Grande"
            else:
                color_box = (0, 0, 255)  # Rojo en BGR (Objeto Pequeño)
                categoria = "Pequeno"

            # Dibujar el Bounding Box sobre la imagen
            cv2.rectangle(resultado, (x, y), (x + w, y + h), color_box, 2)

            # Dibujar un punto central (centroide)
            cv2.circle(resultado, (cx, cy), 4, (0, 255, 0), -1)

            # Etiquetar texto sobre el objeto
            texto = f"#{contador_objetos} ({categoria})"
            cv2.putText(resultado, texto, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_box, 1)

            print(f"Objeto #{contador_objetos}: Área = {area:.1f}px | Categoría = {categoria} | Centroide = ({cx}, {cy})")

    # ==========================================
    # 3. GUARDAR RESULTADOS (En la carpeta imgTaller6)
    # ==========================================
    cv2.imwrite('imgTaller6/1_imagen_binaria_limpia.png', imagen_limpia)
    cv2.imwrite('imgTaller6/2_clasificacion_objetos.png', resultado)

    # Convertir a RGB para mostrar con Matplotlib
    resultado_rgb = cv2.cvtColor(resultado, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(imagen_limpia, cmap='gray')
    plt.title("1. Máscara Binaria Limpia")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(resultado_rgb)
    plt.title("2. Clasificador (Azul: Grande | Rojo: Pequeño)")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('imgTaller6/panel_clasificacion.png')

    print("\n¡Proceso del Taller 6 completado exitosamente!")
    print("Archivos generados en imgTaller6:")
    print(" - 'imgTaller6/1_imagen_binaria_limpia.png'")
    print(" - 'imgTaller6/2_clasificacion_objetos.png'")
    print(" - 'imgTaller6/panel_clasificacion.png'")