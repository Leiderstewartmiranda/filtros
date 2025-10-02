import cv2
import numpy as np

# === Cargar el clasificador de rostros ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# === Cargar imágenes de filtros (PNG transparente) ===
def load_filter(path, scale=1.0):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img

# Cargar filtros (asegúrate de tener estos archivos en la carpeta "filters")
try:
    gafas = load_filter("filters/gafas.png")
    bigote = load_filter("filters/bigote.png")
    sombrero = load_filter("filters/sombrero.png")
    corona = load_filter("filters/corona.png")
    payaso = load_filter("filters/payaso.png")
except:
    print("Algunos filtros no se pudieron cargar. Usando filtros por defecto.")
    # Crear filtros simples si no existen los archivos
    gafas = np.zeros((100, 100, 4), dtype=np.uint8)
    cv2.rectangle(gafas, (10, 40), (90, 60), (255, 255, 255, 255), -1)
    
    bigote = np.zeros((50, 100, 4), dtype=np.uint8)
    cv2.ellipse(bigote, (50, 25), (40, 20), 0, 0, 360, (255, 255, 255, 255), -1)
    
    sombrero = np.zeros((80, 150, 4), dtype=np.uint8)
    cv2.ellipse(sombrero, (75, 20), (70, 15), 0, 0, 360, (0, 0, 255, 255), -1)
    cv2.rectangle(sombrero, (30, 20), (120, 60), (0, 0, 255, 255), -1)
    
    # Crear filtro de payaso por defecto
    payaso = np.zeros((150, 150, 4), dtype=np.uint8)
    # Peluca colorida
    cv2.circle(payaso, (75, 75), 60, (255, 0, 255, 255), -1)
    # Nariz roja
    cv2.circle(payaso, (75, 85), 15, (0, 0, 255, 255), -1)
    # Ojos
    cv2.circle(payaso, (55, 65), 8, (255, 255, 255, 255), -1)
    cv2.circle(payaso, (95, 65), 8, (255, 255, 255, 255), -1)
    cv2.circle(payaso, (55, 65), 4, (0, 0, 0, 255), -1)
    cv2.circle(payaso, (95, 65), 4, (0, 0, 0, 255), -1)
    # Boca sonriente
    cv2.ellipse(payaso, (75, 105), (25, 15), 0, 0, 180, (0, 0, 0, 255), 3)

# === Función para superponer PNG transparente ===
def overlay_image(bg, fg, x, y):
    h, w = fg.shape[:2]
    rows, cols = bg.shape[:2]
    
    # Asegurarse de que las coordenadas estén dentro del frame
    x = max(0, min(x, cols - w))
    y = max(0, min(y, rows - h))
    
    for i in range(h):
        for j in range(w):
            if y + i >= rows or x + j >= cols:
                continue
            alpha = fg[i, j, 3] / 255.0
            if alpha > 0:
                bg[y+i, x+j] = (1-alpha) * bg[y+i, x+j] + alpha * fg[i, j, :3]
    return bg

# === Función para dibujar filtro de payaso completo ===
def draw_clown_face(frame, x, y, w, h):
    # Dibujar elementos de payaso directamente en el frame
    center_x = x + w // 2
    center_y = y + h // 2
    
    # Peluca colorida (círculo alrededor del rostro)
    cv2.circle(frame, (center_x, center_y), w//2 + 10, (255, 0, 255), -1)
    
    # Nariz roja
    cv2.circle(frame, (center_x, center_y + h//10), w//8, (0, 0, 255), -1)
    
    # Ojos exagerados
    eye_radius = w // 10
    cv2.circle(frame, (center_x - w//4, center_y - h//10), eye_radius, (255, 255, 255), -1)
    cv2.circle(frame, (center_x + w//4, center_y - h//10), eye_radius, (255, 255, 255), -1)
    cv2.circle(frame, (center_x - w//4, center_y - h//10), eye_radius//2, (0, 0, 0), -1)
    cv2.circle(frame, (center_x + w//4, center_y - h//10), eye_radius//2, (0, 0, 0), -1)
    
    # Boca sonriente grande
    cv2.ellipse(frame, (center_x, center_y + h//4), (w//3, h//6), 0, 0, 180, (0, 0, 0), 3)
    
    # Mejillas coloridas
    cv2.circle(frame, (center_x - w//3, center_y + h//10), w//8, (255, 150, 150), -1)
    cv2.circle(frame, (center_x + w//3, center_y + h//10), w//8, (255, 150, 150), -1)
    
    # Sombrero de payaso (opcional)
    hat_points = np.array([
        [center_x - w//2, center_y - h//2],
        [center_x + w//2, center_y - h//2],
        [center_x + w//3, center_y - h],
        [center_x - w//3, center_y - h]
    ], np.int32)
    cv2.fillPoly(frame, [hat_points], (255, 0, 0))

# === Iniciar cámara ===
cap = cv2.VideoCapture(0)

# Variable para controlar el filtro actual
current_filter = "gafas"  # Filtro por defecto

print("Controles:")
print("1: Gafas")
print("2: Bigote") 
print("3: Sombrero")
print("4: Corona")
print("5: Payaso")
print("0: Sin filtro")
print("ESC: Salir")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Voltear frame horizontalmente para efecto espejo
    frame = cv2.flip(frame, 1)
    
    # Detectar rostros
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Aplicar solo el filtro seleccionado
        if current_filter == "gafas":
            gafas_resized = cv2.resize(gafas, (w, int(h/3)))
            frame = overlay_image(frame, gafas_resized, x, y + int(h/4))
            
        elif current_filter == "bigote":
            bigote_resized = cv2.resize(bigote, (int(w/1.5), int(h/6)))
            frame = overlay_image(frame, bigote_resized, x + int(w/6), y + int(2*h/3))
            
        elif current_filter == "sombrero":
            sombrero_resized = cv2.resize(sombrero, (int(w*1.5), int(h/2)))
            frame = overlay_image(frame, sombrero_resized, x - int(w/4), y - int(h/2))
            
        elif current_filter == "corona":
            corona_resized = cv2.resize(corona, (w, int(h/3)))
            frame = overlay_image(frame, corona_resized, x, y - int(h/4))
            
        elif current_filter == "payaso":
            # Intentar usar la imagen PNG primero
            try:
                payaso_resized = cv2.resize(payaso, (w, h))
                frame = overlay_image(frame, payaso_resized, x, y)
            except:
                # Si falla, dibujar el payaso
                draw_clown_face(frame, x, y, w, h)

    # Mostrar información del filtro actual
    cv2.putText(frame, f"Filtro: {current_filter}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Mostrar controles en pantalla
    cv2.putText(frame, "1:Gafas 2:Bigote 3:Sombrero", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "4:Corona 5:Payaso 0:Limpiar", (10, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Filtros Snapchat Style", frame)

    key = cv2.waitKey(1) & 0xFF
    
    # Controles del teclado
    if key == 27:  # ESC para salir
        break
    elif key == ord('1'):
        current_filter = "gafas"
    elif key == ord('2'):
        current_filter = "bigote"
    elif key == ord('3'):
        current_filter = "sombrero"
    elif key == ord('4'):
        current_filter = "corona"
    elif key == ord('5'):
        current_filter = "payaso"
    elif key == ord('0'):
        current_filter = "none"

cap.release()
cv2.destroyAllWindows()