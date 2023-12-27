from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
from tensorflow import keras
import sqlite3

app = Flask(__name__)

model = keras.models.load_model('C:\\Users\\AAHILAHMED\\PycharmProjects\\pythonProject\\trained_models\\nebulae_classification_model.h5')  # Replace 'trained_model.h5' with the actual path to your trained model

def get_db_connection():
    conn = sqlite3.connect('astronomy_data.db')
    cursor = conn.cursor()
    return conn, cursor

def init_database():
    conn, cursor = get_db_connection()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS image_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT NOT NULL,
            classification TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_database()

# Function to classify an image
def classify_image(image_path):
    try:
        img = Image.open(image_path)
        img = img.resize((128, 128))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)
        prediction = model.predict(img)
        class_labels = ['Nebulae', 'Planet', 'Star Cluster']
        predicted_class = class_labels[np.argmax(prediction)]
        return predicted_class
    except Exception as e:
        return str(e)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/source_code.html')
def source_code():
    return render_template('source_code.html')

@app.route('/presentation.html')
def presentation():
    return render_template('presentation.html')

@app.route('/classify_image', methods=['POST'])
def classify_image_endpoint():
    image_path = request.form.get('imagePath')
    print(f'Received image path: {image_path}')

    try:
        # Perform image classification here
        result = classify_image(image_path)
        print(f'Classification:{result}')
        conn, cursor = get_db_connection()
        cursor.execute("INSERT INTO image_data (image_path, classification) VALUES (?,?)", (image_path, result))
        conn.commit()
        conn.close()
        return jsonify(result=result)
    except Exception as e:
        print(f"Error during classification: {str(e)}")
        return jsonify(result="Error during classification")

# Route to retrieve image data from the database
@app.route('/Database.html')
def get_image_data():
    conn, cursor = get_db_connection()
    cursor.execute("SELECT image_path, classification FROM image_data")
    data = cursor.fetchall()
    conn.close()

    return render_template('Database.html', image_data=data)

if __name__ == '__main__':
    app.run(debug=True)
    data= get_image_data()
    for row in data:
        print(f"Image Path: {row[0]}, Classification: {row[1]}")