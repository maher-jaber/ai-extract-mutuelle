from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import uuid
from main import GeneralizedOCRProcessor

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)  # Active CORS pour toutes les routes

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'tiff', 'bmp'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialisation du processeur OCR
processor = GeneralizedOCRProcessor(lang='fr')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/process', methods=['POST'])
def process_document():
    # Vérifier si un fichier a été envoyé
    if 'file' not in request.files:
        return jsonify({'error': 'Aucun fichier fourni'}), 400
    
    file = request.files['file']
    
    # Vérifier si le fichier a un nom
    if file.filename == '':
        return jsonify({'error': 'Aucun fichier sélectionné'}), 400
    
    # Vérifier l'extension du fichier
    if file and allowed_file(file.filename):
        # Générer un nom de fichier unique
        file_id = str(uuid.uuid4())
        file_extension = file.filename.rsplit('.', 1)[1].lower()
        filename = f"{file_id}.{file_extension}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Sauvegarder le fichier
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(filepath)
        
        try:
            # Traiter le fichier avec votre OCR
            result = processor.process_file(filepath)
            
            # Nettoyer le fichier uploadé
            if os.path.exists(filepath):
                os.remove(filepath)
                
            return jsonify(result)
            
        except Exception as e:
            # Nettoyer en cas d'erreur
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': f'Erreur de traitement: {str(e)}'}), 500
    
    return jsonify({'error': 'Type de fichier non autorisé'}), 400

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'OK', 'message': 'Serveur OCR opérationnel'})

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)