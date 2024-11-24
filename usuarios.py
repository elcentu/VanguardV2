import os
import re  # Para validaciones de nombre
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
import markdown2  # Para procesar el contenido de Gemini
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.patches as patches
import matplotlib
from flask import make_response
import pandas as pd
matplotlib.use('Agg')

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
modelo = TFSMLayer('grapeproy2', call_endpoint='serving_default')

# Diccionario de nombres de clases
class_names = {
    0: 'Yesca',
    1: 'Oídio',
    2: 'Podredumbre negra',
    3: 'Botrytis cinerea',
    4: 'Saludable',
    5: 'Mildiú',
    6: 'Tizon de la hoja',
    7: 'Otros'
}

# Función para procesar imágenes
def process_image(img_path):
    try:
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
        confidence = float(predictions[0][class_idx])

        return class_names[class_idx], confidence
    except Exception as e:
        print(f"Error en process_image: {e}")
        return None, None

# Función para procesar videos
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_interval = 10
    frame_count = 0
    processed_frames = []
    detected_classes = set()
    fps = cap.get(cv2.CAP_PROP_FPS)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        try:
            if frame_count % frame_interval == 0:
                frame_time = frame_count / fps
                hours = int(frame_time // 3600)
                minutes = int((frame_time % 3600) // 60)
                seconds = int(frame_time % 60)
                time_str = f"{hours:02}:{minutes:02}:{seconds:02}"

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
                    'class_name': class_name,
                    'frame_time': time_str
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

# Registro de usuario con validación
@app.route('/registro', methods=['GET', 'POST'])
def registro():
    if request.method == 'POST':
        nombre = request.form['nombre']
        correo = request.form['correo']
        contraseña = request.form['contraseña']

        if not re.match(r"[^@]+@[^@]+\.[^@]+", correo):
            flash('Por favor, introduce un correo electrónico válido.', 'danger')
            return redirect(url_for('registro'))

        if len(contraseña) < 8 or not re.search(r"[A-Z]", contraseña) or not re.search(r"[0-9]", contraseña):
            flash('La contraseña debe tener al menos 8 caracteres, una letra mayúscula y un número.', 'danger')
            return redirect(url_for('registro'))

        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT * FROM usuarios WHERE correo = %s", (correo,))
            if cursor.fetchone():
                flash('El correo electrónico ya está registrado.', 'danger')
                return redirect(url_for('registro'))
            
            # Registrar usuario con rol 'usuario' por defecto
            cursor.execute(
                sql.SQL("INSERT INTO usuarios (nombre, correo, contraseña, rol) VALUES (%s, %s, %s, %s)"),
                [nombre, correo, contraseña, 'usuario']
            )
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
        cursor.execute("SELECT id, contraseña, rol FROM usuarios WHERE correo = %s AND contraseña = %s", (correo, contraseña))
        user = cursor.fetchone()
        cursor.close()
        conn.close()

        if user:
            session['user_id'] = user[0]
            session['user_rol'] = user[2]  # Guardar rol en la sesión
            flash('Inicio de sesión exitoso', 'success')

            # Redirigir según el rol del usuario
            if user[2] == 'admin':
                return redirect(url_for('admin_panel'))  # Ruta para el panel del administrador
            else:
                return redirect(url_for('imagenes'))  # Ruta del panel de usuario normal
        else:
            flash('Correo o contraseña incorrectos', 'danger')
            return redirect(url_for('ingresar'))

    return render_template('ingresar.html')

# Carga y procesamiento de imágenes
@app.route('/imagenes', methods=["GET", "POST"])
def imagenes():
    if 'user_id' not in session:
        flash("Por favor, inicia sesión para acceder a esta página", "danger")
        return redirect(url_for('ingresar'))
    
    if request.method == "POST":
        # Limpiar datos anteriores de imágenes en la sesión
        session.pop('frames_for_validation_images', None)

        files = request.files.getlist('media')
        upload_folder = os.path.join('static', 'uploads')
        os.makedirs(upload_folder, exist_ok=True)
        results = []

        for file in files:
            if file:
                file_path = os.path.join(upload_folder, file.filename)
                file.save(file_path)
                class_name, confidence = process_image(file_path)
                results.append({
                    'frame_path': f'uploads/{file.filename}',
                    'class_name': class_name,
                    'confidence': confidence,
                    'is_video': False,  # Especifica que es una imagen
                    'frame_time': None  # No aplica tiempo para imágenes
                })
        
        # Guardar imágenes procesadas en la sesión
        session['frames_for_validation_images'] = results

        return render_template("imagenes.html", results=results)
    return render_template("imagenes.html")


# Carga y procesamiento de videos
@app.route('/videos', methods=["GET", "POST"])
def videos():
    if 'user_id' not in session:
        flash("Por favor, inicia sesión para acceder a esta página", "danger")
        return redirect(url_for('ingresar'))
    
    if request.method == "POST":
        # Limpiar datos anteriores de videos en la sesión
        session.pop('frames_for_validation_videos', None)

        file = request.files.get('media')
        if file:
            upload_folder = os.path.join('static', 'uploads')
            os.makedirs(upload_folder, exist_ok=True)
            file_path = os.path.join(upload_folder, file.filename)
            file.save(file_path)

            frames_result, detected_classes = process_video(file_path)
            
            # Guardar frames procesados en la sesión
            session['frames_for_validation_videos'] = frames_result

            return render_template("videos.html", frames_result=frames_result, detected_classes=detected_classes)
    return render_template("videos.html")



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

# Ruta para obtener recomendaciones de Gemini
@app.route('/get_recommendations', methods=['POST'])
def get_recommendations():
    data = request.get_json()
    enfermedades = data.get("disease_names", [])
    
    if not enfermedades:
        return jsonify({"recommendations": "No se proporcionaron enfermedades para obtener recomendaciones."})

    chat_session = model.start_chat(history=[])
    response = chat_session.send_message(", ".join(enfermedades))
    
    formatted_response = markdown2.markdown(response.text)
    
    return jsonify({"recommendations": formatted_response})

@app.route('/enviar_a_validacion', methods=['POST'])
def enviar_a_validacion():
    if 'user_id' not in session:
        flash("Por favor, inicia sesión para acceder a esta página", "danger")
        return redirect(url_for('ingresar'))
    
    data = request.get_json()
    selected_frames = data.get('frames', [])

    if not selected_frames:
        return jsonify({
            "message": "No se seleccionaron frames para enviar a validación.",
            "redirect_url": url_for('imagenes')
        }), 400

    # Determinar si los datos provienen de imágenes o videos
    is_video = any('frame_time' in frame for frame in selected_frames)

    # Limpiar datos previos y almacenar lo nuevo
    if is_video:
        session.pop('frames_for_validation_images', None)  # Limpiar imágenes
        session['frames_for_validation_videos'] = selected_frames  # Guardar videos
    else:
        session.pop('frames_for_validation_videos', None)  # Limpiar videos
        session['frames_for_validation_images'] = selected_frames  # Guardar imágenes

    return jsonify({
        "message": "Frames enviados a validación exitosamente.",
        "redirect_url": url_for('validacion')
    })

@app.route('/validacion', methods=['GET', 'POST'])
def validacion():
    class_legend = {
        'A': 'Botrytis cinerea',
        'B': 'Esca',
        'C': 'Mildiú',
        'D': 'Oídio',
        'E': 'Podredumbre negra',
        'F': 'Tizón de la hoja',
        'G': 'Saludable',
        'H': 'Otros'
    }

    if request.method == 'POST':
        try:
            # Procesar validaciones enviadas desde el frontend
            data = request.get_json()
            validations = data.get('validations', [])
            user_id = session.get('user_id')

            # Validar que el usuario esté autenticado
            if not user_id:
                return jsonify({"message": "Usuario no autenticado."}), 401

            # Obtener el nombre del usuario desde la sesión
            user_name = session.get('user_name')
            if not user_name:
                # Recuperar el nombre del usuario desde la base de datos si no está en la sesión
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute("SELECT nombre FROM usuarios WHERE id = %s", (user_id,))
                result = cursor.fetchone()
                user_name = result[0] if result else "Usuario desconocido"
                session['user_name'] = user_name  # Almacenar el nombre en la sesión para futuros usos
                cursor.close()
                conn.close()

            # Validar que se hayan enviado datos
            if not validations:
                return jsonify({"message": "No se enviaron validaciones."}), 400

            conn = get_db_connection()
            cursor = conn.cursor()

            # Procesar cada validación
            for validation in validations:
                frame_path = validation.get('frame_path')  # Ruta del frame
                user_class_letter = validation.get('user_class')  # Clase seleccionada

                # Verificar que la clase seleccionada sea válida
                if user_class_letter not in class_legend:
                    return jsonify({"message": f"Clase inválida: {user_class_letter}"}), 400

                # Obtener el nombre completo de la clase seleccionada
                user_class_name = class_legend[user_class_letter]

                # Obtener la clase predicha por el modelo
                model_class = next(
                    (frame['class_name'] for frame in session.get('frames_for_validation_images', []) + session.get('frames_for_validation_videos', [])
                     if frame['frame_path'] == frame_path),
                    'Desconocida'
                )

                # Insertar la validación en la base de datos con el query proporcionado
                cursor.execute(
                    """
                    INSERT INTO validaciones (user_id, nombre_usuario, frame_path, user_class, model_class, fecha_validacion)
                    VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                    """,
                    (user_id, user_name, frame_path, user_class_name, model_class)
                )

            # Confirmar los cambios en la base de datos
            conn.commit()
            cursor.close()
            conn.close()

            # Responder al frontend
            return jsonify({"message": "Validaciones guardadas correctamente."})

        except Exception as e:
            print(f"Error al guardar validaciones: {e}")
            return jsonify({"message": "Error interno al guardar validaciones."}), 500

    # Mostrar las imágenes o videos seleccionados para validación
    frames_images = session.get('frames_for_validation_images', [])
    frames_videos = session.get('frames_for_validation_videos', [])

    # Priorizar mostrar solo un conjunto de datos en la validación
    if frames_images and frames_videos:
        flash("Se detectaron datos de imágenes y videos; solo se mostrarán imágenes seleccionadas.", "info")
        frames_videos = []

    # Seleccionar datos para validación
    frames_for_validation = frames_images or frames_videos

    # Clasificar los datos como videos o imágenes
    for frame in frames_for_validation:
        frame['is_video'] = 'frame_time' in frame

    # Renderizar la plantilla con los datos para validación
    return render_template('validacion.html', frames=frames_for_validation, class_legend=class_legend)

@app.route('/admin_panel', methods=['GET', 'POST'])
def admin_panel():
    if 'user_rol' not in session or session['user_rol'] != 'admin':
        flash("Acceso denegado. Solo administradores pueden ingresar.", "danger")
        return redirect(url_for('ingresar'))

    conn = get_db_connection()
    cursor = conn.cursor()

    # Obtener filtros desde el formulario
    usuario = request.args.get('usuario')
    fecha_desde = request.args.get('fecha_desde')
    fecha_hasta = request.args.get('fecha_hasta')
    clase = request.args.get('clase')

    # Construir consulta con filtros
    query = """
        SELECT u.nombre, u.correo, v.id, v.frame_path, v.user_class, v.model_class, v.fecha_validacion
        FROM validaciones v
        JOIN usuarios u ON v.user_id = u.id
        WHERE 1=1
    """
    params = []
    if usuario:
        query += " AND u.nombre ILIKE %s"
        params.append(f"%{usuario}%")
    if fecha_desde:
        query += " AND v.fecha_validacion >= %s"
        params.append(fecha_desde)
    if fecha_hasta:
        query += " AND v.fecha_validacion <= %s"
        params.append(fecha_hasta)
    if clase:
        query += " AND v.user_class = %s"
        params.append(clase)

    query += " ORDER BY u.nombre, v.fecha_validacion DESC"

    cursor.execute(query, tuple(params))
    raw_validaciones = cursor.fetchall()
    cursor.close()
    conn.close()

    # Agrupar validaciones por usuario y luego por fecha
    validaciones_por_usuario = {}
    for row in raw_validaciones:
        usuario_nombre = row[0]
        fecha_validacion = row[6].strftime('%Y-%m-%d %H:%M:%S') if isinstance(row[6], datetime) else row[6]

        if usuario_nombre not in validaciones_por_usuario:
            validaciones_por_usuario[usuario_nombre] = {}

        if fecha_validacion not in validaciones_por_usuario[usuario_nombre]:
            validaciones_por_usuario[usuario_nombre][fecha_validacion] = []

        validaciones_por_usuario[usuario_nombre][fecha_validacion].append({
            "correo": row[1],
            "id": row[2],
            "frame_path": row[3],
            "user_class": row[4],
            "model_class": row[5],
            "fecha_validacion": fecha_validacion
        })

    return render_template('admin_panel.html', validaciones_por_usuario=validaciones_por_usuario)

@app.route('/exportar_excel', methods=['POST'])
def exportar_excel():
    if 'user_rol' not in session or session['user_rol'] != 'admin':
        flash("Acceso denegado. Solo administradores pueden exportar datos.", "danger")
        return redirect(url_for('admin_panel'))

    # Obtener los IDs seleccionados del formulario
    selected_ids = request.form.getlist('validaciones_seleccionadas')
    if not selected_ids:
        flash("No se seleccionaron validaciones para exportar.", "warning")
        return redirect(url_for('admin_panel'))

    try:
        # Convertir los IDs seleccionados a enteros
        selected_ids = [int(id) for id in selected_ids]
    except ValueError:
        flash("Error al procesar las validaciones seleccionadas. Por favor, inténtalo de nuevo.", "danger")
        return redirect(url_for('admin_panel'))

    conn = get_db_connection()
    cursor = conn.cursor()

    # Construir consulta para obtener los datos seleccionados
    query = """
        SELECT u.nombre AS usuario, u.correo, v.id, v.frame_path, v.user_class, v.model_class, v.fecha_validacion
        FROM validaciones v
        JOIN usuarios u ON v.user_id = u.id
        WHERE v.id = ANY(%s)
    """
    cursor.execute(query, (selected_ids,))
    datos = cursor.fetchall()
    cursor.close()
    conn.close()

    # Verificar si hay datos para exportar
    if not datos:
        flash("No se encontraron validaciones para los IDs seleccionados.", "warning")
        return redirect(url_for('admin_panel'))

    # Convertir los datos a un DataFrame de pandas para exportar
    columnas = ['Usuario', 'Correo', 'ID Validación', 'Ruta Imagen', 'Clase Validada', 'Clase Predicha', 'Fecha Validación']
    df = pd.DataFrame(datos, columns=columnas)

    # Crear el archivo Excel en memoria
    import io
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Validaciones')

        # Ajustar el ancho de las columnas
        workbook = writer.book
        worksheet = writer.sheets['Validaciones']
        for i, column in enumerate(df.columns):
            max_length = max(
                df[column].astype(str).map(len).max(),  # Longitud máxima de los valores
                len(column)  # Longitud del nombre de la columna
            )
            worksheet.set_column(i, i, max_length + 2)  # Ajuste con un margen de 2 espacios

    # Crear la respuesta de descarga
    output.seek(0)
    response = make_response(output.read())
    response.headers['Content-Disposition'] = 'attachment; filename=validaciones_seleccionadas.xlsx'
    response.headers['Content-type'] = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'

    return response

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash("Has cerrado sesión exitosamente", "success")
    return redirect(url_for('index'))

# Ejecutar la aplicación Flask
if __name__ == '__main__':
    app.run(debug=True, port=5001)
