import cv2
import numpy as np

# Colores base
VERDE = (0, 200, 0)
AMARILLO = (0, 200, 200)
NEGRO = (0, 0, 0)
GRIS = (220, 220, 220)

# Tamaño ventana
ventana_ancho, ventana_alto = 800, 600
croquis = np.zeros((ventana_alto, ventana_ancho, 3), dtype=np.uint8)
croquis[:] = GRIS

# Tamaño plazas
plaza_ancho, plaza_alto = 100, 80
filas, columnas = 3, 4

# Centrar el estacionamiento
total_ancho = columnas * plaza_ancho + (columnas - 1) * 20
total_alto = filas * plaza_alto + (filas - 1) * 20
offset_x = (ventana_ancho - total_ancho) // 2
offset_y = (ventana_alto - total_alto) // 2

# Posiciones plazas
plazas = []
for i in range(filas):
    for j in range(columnas):
        x = offset_x + j * (plaza_ancho + 20)
        y = offset_y + i * (plaza_alto + 20)
        plazas.append((x, y))

# Cargar imágenes de los carritos con transparencia
img_carrito_rojo = cv2.imread("imagenes/Carrito.png", cv2.IMREAD_UNCHANGED)
img_carrito_azul = cv2.imread("imagenes/CARRITO2.png", cv2.IMREAD_UNCHANGED)
img_carrito_rojo = cv2.resize(img_carrito_rojo, (plaza_ancho, plaza_alto))
img_carrito_azul = cv2.resize(img_carrito_azul, (plaza_ancho, plaza_alto))

# Estado inicial de plazas: 1-9 verdes, 10-12 amarillas
estado_plazas = ["libre"] * 9 + ["apartado"] * 3

cap = cv2.VideoCapture(0)


def detectar_colores(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Rojo
    rojo_bajo1 = np.array([0, 150, 50])
    rojo_alto1 = np.array([10, 255, 255])
    rojo_bajo2 = np.array([170, 150, 50])
    rojo_alto2 = np.array([180, 255, 255])
    mask_rojo = cv2.inRange(hsv, rojo_bajo1, rojo_alto1) | cv2.inRange(hsv, rojo_bajo2, rojo_alto2)

    # Azul
    azul_bajo = np.array([100, 150, 50])
    azul_alto = np.array([140, 255, 255])
    mask_azul = cv2.inRange(hsv, azul_bajo, azul_alto)

    rojo_detectado = cv2.countNonZero(mask_rojo) > 2000
    azul_detectado = cv2.countNonZero(mask_azul) > 2000

    return rojo_detectado, azul_detectado


def pegar_imagen_transparente(fondo, imagen, x, y):
    """
    Pega una imagen PNG sobre otra imagen (fondo) respetando transparencia.
    Si la imagen no tiene canal alfa, se genera uno eliminando fondo blanco/gris.
    """
    h, w = imagen.shape[:2]

    # Crear canal alfa si no existe
    if imagen.shape[2] == 3:
        # Consideramos blanco/gris claro como fondo
        gris_fondo = (imagen[:,:,0] > 200) & (imagen[:,:,1] > 200) & (imagen[:,:,2] > 200)
        alpha = np.ones((h, w), dtype=float)
        alpha[gris_fondo] = 0
        img_rgb = imagen.astype(float)
    else:
        # Ya tiene alfa
        b, g, r, a = cv2.split(imagen)
        img_rgb = cv2.merge((b, g, r)).astype(float)
        alpha = a.astype(float)/255.0

    alpha_inv = 1.0 - alpha
    roi = fondo[y:y+h, x:x+w].astype(float)

    for c in range(3):
        roi[:, :, c] = img_rgb[:, :, c] * alpha + roi[:, :, c] * alpha_inv

    fondo[y:y+h, x:x+w] = roi.astype(np.uint8)



while True:
    ret, frame = cap.read()
    if not ret:
        break

    rojo, azul = detectar_colores(frame)

    croquis_visual = croquis.copy()

    for idx, (x, y) in enumerate(plazas):
        # Plaza 1 → carrito rojo si rojo detectado
        if idx == 0 and rojo:
            pegar_imagen_transparente(croquis_visual, img_carrito_rojo, x, y)
        # Plaza 12 → carrito azul si azul detectado
        elif idx == 11 and azul:
            pegar_imagen_transparente(croquis_visual, img_carrito_azul, x, y)
        else:
            # Plazas normales con color base
            color = VERDE if idx < 9 else AMARILLO
            cv2.rectangle(croquis_visual, (x, y), (x + plaza_ancho, y + plaza_alto), color, -1)
            cv2.rectangle(croquis_visual, (x, y), (x + plaza_ancho, y + plaza_alto), NEGRO, 2)

            # Número centrado
            text_size = cv2.getTextSize(str(idx + 1), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = x + (plaza_ancho - text_size[0]) // 2
            text_y = y + (plaza_alto + text_size[1]) // 2
            cv2.putText(croquis_visual, str(idx + 1), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, NEGRO, 2)

    cv2.imshow("Estacionamiento con Carritos", croquis_visual)
    cv2.imshow("Camara", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Esc para salir
        break

cap.release()
cv2.destroyAllWindows()
