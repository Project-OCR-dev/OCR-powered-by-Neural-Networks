from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, get_flashed_messages
import os
import sys
from werkzeug.utils import secure_filename

# Ajouter le chemin pour importer preprocessing
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from preprocessing.image_processing import ocrLettreIsolee, ocrTexteComplet
from PIL import Image

app = Flask(__name__)
# dossier ou sont sauvegardés les fichiers uploadé
app.config['UPLOAD_FOLDER'] = 'static/uploads'
# liste des formats de fichiers autorisés
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
# il faudra générer une vraie clé aléatoire avant mise en production sur serveur
app.config['SECRET_KEY'] = 'une-cle-secrete'

# Créer le dossier uploads s'il n'existe pas
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def allowed_file(filename):
    """
    Vérifie si l'extension du fichier est autorisée.
    
    Args:
        filename (str): Nom du fichier à vérifier
    
    Returns:
        bool: True si extension autorisée, False sinon
    """
    if '.' in filename:
        ext = filename.rsplit('.', 1)[1].lower()
        return ext in app.config['ALLOWED_EXTENSIONS']
    return False


@app.route('/')
def index():
    """Affiche la page d'accueil avec le formulaire"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Traite l'upload d'un fichier image.
    
    Vérifie la présence du fichier, valide son extension,
    sauvegarde dans le dossier uploads/ et lance l'analyse OCR
    selon le mode choisi (lettre isolée ou texte complet).
    
    Returns:
        redirection vers la page de résultats
    """
    # Vérifier qu'un fichier a été envoyé
    if 'file' not in request.files:
        flash("Erreur : aucun fichier sélectionné", 'error')
        return redirect(url_for('index'))

    file = request.files['file']

    if file.filename == '':
        flash("Erreur : fichier inexistant", 'error')
        return redirect(url_for('index'))

    # Vérifier l'extension
    if not allowed_file(file.filename):
        flash("Erreur : type de fichier non autorisé !", 'error')
        return redirect(url_for('index'))
    
    # Sauvegarder le fichier
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # Récupérer le mode choisi
    mode = request.form.get('mode', 'lettre')  # Par défaut 'lettre'
    
    # Charger l'image
    try:
        image = Image.open(filepath)
        
        # Traiter selon le mode
        if mode == 'lettre':
            resultat = ocrLettreIsolee(image)
            type_ocr = "Lettre isolée"
        else:  # mode == 'texte'
            resultat = ocrTexteComplet(image)
            type_ocr = "Texte complet"
        
        flash(f"{filename} analysé avec succès !", 'success')
        
        # Rediriger vers la page de résultats
        return redirect(url_for('results', 
                               filename=filename,
                               prediction=resultat,
                               type_ocr=type_ocr))
    
    except Exception as e:
        flash(f"Erreur lors du traitement : {str(e)}", 'error')
        return redirect(url_for('index'))


@app.route('/processing/<filename>')
def process(filename):
    """Affiche la page de traitement (optionnel - peut être supprimé)"""
    return render_template('processing.html', filename=filename)


@app.route('/analyze/<filename>', methods=['POST'])
def analyze(filename):
    """
    Route optionnelle pour analyse asynchrone
    (peut être supprimée si on fait tout dans /upload)
    """
    import random
    
    # données simulées - sera remplacé par le vrai modèle ML plus tard
    prediction = random.choice(['A', 'B', 'C', 'D', 'E', '1', '2', '3', '7', '9'])
    confidence = round(random.uniform(85, 98), 1)
    
    # Rediriger vers results avec les données en query parameters
    return redirect(url_for('results', filename=filename, 
                           prediction=prediction, 
                           confidence=confidence))


@app.route('/results/<filename>')
def results(filename):
    """Affiche la page de résultats"""
    # Récupérer les paramètres de l'URL
    prediction = request.args.get('prediction', 'N/A')
    type_ocr = request.args.get('type_ocr', 'Lettre isolée')
    confidence = request.args.get('confidence', 0)
    
    return render_template('results.html',
                         filename=filename,
                         prediction=prediction,
                         type_ocr=type_ocr,
                         confidence=confidence)


if __name__ == '__main__':
    app.run(debug=True)