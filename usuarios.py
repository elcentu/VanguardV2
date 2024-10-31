import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from tensorflow.keras.layers import TFSMLayer
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import cv2
import psycopg2
from psycopg2 import sql
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Configuración de la base de datos
PGUSER = 'usuario_uva'
PGHOST = 'localhost'
PGDATABASE = 'proyect_uva'
PGPASSWORD = 'uva12345'
PGPORT = 5432

# Configuración de la API de Gemini
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Configuración del modelo de Gemini
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    system_instruction="Necesito que hagas una manera de contrarrestar enfermedades de la vid o uva, proporcionando un listado de acciones seguras y no dubitativas que el usuario debe tomar.",
)

# Cargar el modelo usando TFSMLayer
modelo = TFSMLayer('C:\\Users\\JEFERSON\\Desktop\\app web con efficennet\\mi_webapp\\grapeproy_savedmodel', call_endpoint='serving_default')

# Diccionario de nombres de clases
class_names = {
    0: 'Botrytis cinerea',
    1: 'Esca',
    2: 'Mildiú',
    3: 'Oídio',
    4: 'Podredumbre negra',
    5: 'Saludable',
    6: 'Tizón de la hoja'
}

# Función para verificar si una imagen contiene suficiente verde (vid o uva)
def is_image_of_grape(img_path):
    img = cv2.imread(img_path)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv_img, lower_green, upper_green)
    green_ratio = cv2.countNonZero(green_mask) / (img.size / 3)
    return green_ratio > 0.05

# Función para procesar imágenes
def process_image(img_path):
    img = Image.open(img_path)
    img = img.resize((224, 224))
    img_array = np.array(img)
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
    img_array = img_array.astype('float32')
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions_dict = modelo(img_array)
    predictions = predictions_dict['dense_1']
    
    class_idx = np.argmax(predictions)
    return class_names[class_idx], predictions[0][class_idx]

# Función para procesar videos
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_interval = 10
    frame_count = 0
    processed_frames = []
    detected_classes = set()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        try:
            if frame_count % frame_interval == 0:
                frame_resized = cv2.resize(frame, (224, 224))
                img_array = np.array(frame_resized).astype('float32')
                img_array = preprocess_input(img_array)
                img_array = np.expand_dims(img_array, axis=0)
                
                predictions_dict = modelo(img_array)
                predictions = predictions_dict['dense_1']
                
                class_idx = np.argmax(predictions)
                class_name = class_names.get(class_idx, "Desconocida")
                detected_classes.add(class_name)
                
                frame_filename = f'frame_{frame_count}.png'
                frame_path = os.path.join('static', 'uploads', frame_filename)
                cv2.imwrite(frame_path, frame)
                processed_frames.append({
                    'frame_path': f'uploads/{frame_filename}',
                    'class_name': class_name
                })
        except Exception as e:
            print(f"Error al procesar el frame {frame_count}: {e}")
        
        frame_count += 1
    
    cap.release()
    return processed_frames, list(detected_classes)

# Conectar a la base de datos
def get_db_connection():
    conn = psycopg2.connect(
        dbname=PGDATABASE,
        user=PGUSER,
        password=PGPASSWORD,
        host=PGHOST,
        port=PGPORT
    )
    return conn

@app.route('/')
def index():
    return render_template('index.html')

# Registro de usuario
@app.route('/registro', methods=['GET', 'POST'])
def registro():
    if request.method == 'POST':
        nombre = request.form['nombre']
        correo = request.form['correo']
        contraseña = request.form['contraseña']
        
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT * FROM usuarios WHERE correo = %s", (correo,))
            if cursor.fetchone():
                flash('El correo electrónico ya está registrado.', 'danger')
                return redirect(url_for('registro'))
            cursor.execute(sql.SQL("INSERT INTO usuarios (nombre, correo, contraseña) VALUES (%s, %s, %s)"),
                           [nombre, correo, contraseña])
            conn.commit()
            flash('Registro exitoso', 'success')
            return redirect(url_for('index'))
        except Exception as e:
            print("Error al registrar el usuario:", e)
            flash('Error al registrar el usuario', 'danger')
        finally:
            cursor.close()
            conn.close()
        
    return render_template('registro.html')

# Función para el inicio de sesión
@app.route('/ingresar', methods=['GET', 'POST'])
def ingresar():
    if request.method == 'POST':
        correo = request.form['correo']
        contraseña = request.form['contraseña']
        
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, contraseña FROM usuarios WHERE correo = %s AND contraseña = %s", (correo, contraseña))
        user = cursor.fetchone()
        cursor.close()
        conn.close()

        if user:
            session['user_id'] = user[0]
            flash('Inicio de sesión exitoso', 'success')
            return redirect(url_for('imagenes'))
        else:
            flash('Correo o contraseña incorrectos', 'danger')
            return redirect(url_for('ingresar'))

    return render_template('ingresar.html')

# Carga y procesamiento de imágenes/videos
@app.route('/imagenes', methods=["GET", "POST"])
def imagenes():
    if 'user_id' not in session:
        flash("Por favor, inicia sesión para acceder a esta página", "danger")
        return redirect(url_for('ingresar'))
    
    if request.method == "POST":
        file = request.files.get('media')
        if file:
            file_type = file.content_type
            upload_folder = os.path.join('static', 'uploads')
            os.makedirs(upload_folder, exist_ok=True)
            file_path = os.path.join(upload_folder, file.filename)
            file.save(file_path)

            # Filtrado previo para verificar si es una imagen de uva
            if "image" in file_type and not is_image_of_grape(file_path):
                flash("La imagen no contiene una vid o uva.", 'danger')
                return redirect(url_for("imagenes"))

            class_name = None
            if "image" in file_type:
                class_name, _ = process_image(file_path)
                media_type = 'image'
                frames_result = None
                media_path = f'uploads/{file.filename}'
            elif "video" in file_type:
                frames_result, detected_classes = process_video(file_path)
                class_name = ", ".join(detected_classes)
                media_type = 'video'
                media_path = None
            else:
                flash("Tipo de archivo no soportado", 'danger')
                return redirect(url_for("imagenes"))
            
            try:
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO consultas (user_id, class_name, consulta_fecha) VALUES (%s, %s, %s)",
                    (session['user_id'], class_name, datetime.now())
                )
                conn.commit()
            except Exception as e:
                print(f"Error guardando la consulta: {e}")
            finally:
                cursor.close()
                conn.close()
            
            return render_template("imagenes.html", 
                                   media_type=media_type, 
                                   class_name=class_name if media_type == 'image' else None, 
                                   frames_result=frames_result if media_type == 'video' else None,
                                   media_path=media_path,
                                   detected_classes=detected_classes if media_type == 'video' else [class_name])
    return render_template("imagenes.html")

# Ruta para obtener recomendaciones de Gemini
@app.route('/get_recommendations', methods=['POST'])
def get_recommendations():
    data = request.get_json()
    enfermedades = data.get("disease_names", [])
    
    chat_session = model.start_chat(history=[])
    consulta = f"Dame recomendaciones para tratar las siguientes enfermedades de la vid: {', '.join(enfermedades)}"
    response = chat_session.send_message(consulta)
    
    return jsonify({"recommendations": response.text})

@app.route('/mis-consultas')
def mis_consultas():
    if 'user_id' not in session:
        flash("Por favor, inicia sesión para ver tus consultas", "danger")
        return redirect(url_for('ingresar'))
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT class_name, consulta_fecha FROM consultas WHERE user_id = %s ORDER BY consulta_fecha DESC", (session['user_id'],))
    consultas = cursor.fetchall()
    cursor.close()
    conn.close()
    return render_template('mis_consultas.html', consultas=consultas)

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash("Has cerrado sesión exitosamente", "success")
    return redirect(url_for('index'))

# Ejecutar la aplicación Flask
if __name__ == '__main__':
    app.run(debug=True, port=5001)
