from flask import Flask, render_template, request,jsonify,redirect,url_for,flash,get_flashed_messages
import os
import random

app = Flask(__name__)
# dossier ou sont sauvegardés les fichiers uploadé
app.config['UPLOAD_FOLDER'] = 'static/uploads'
# liste des formats de fichiers autorisés
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
# il faudra générer une vraie clé aléatoire avant mise en production sur serveur
app.config['SECRET_KEY'] = 'une-cle-secrete'


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

# Route qui permet de charger index.html
@app.route('/')
def index():
    """Affiche la page d'accueil avec le formulaire"""
    return render_template('index.html')

# Route qui reçoit le fichier uploadé via POST
@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Traite l'upload d'un fichier image.
    
    Vérifie la présence du fichier, valide son extension,
    et le sauvegarde dans le dossier uploads/ puis redirige 
    vers la page de traitement 
    
    Returns:
        redirection vers fonction process
    """
    if 'file' not in request.files:
        flash("Erreur : fichier non selectionné", 'error')
        return redirect(url_for('index'))

    file = request.files['file']

    if file.filename == '':
        flash("Erreur :  aucun fichier sélectionné", 'error')
        return redirect(url_for('index'))

    if allowed_file(file.filename):
        flash(f"{file.filename} uploadé avec succès !", 'success')
        file.save(os.path.join('static/uploads', file.filename))
        return redirect(url_for('process',filename=file.filename))
    else:
        flash("Erreur :  type de fichier non autorisé !", 'error')
        return redirect(url_for('index'))
    

@app.route('/processing/<filename>')
def process(filename):
    """Affiche la page de traitement"""
    return render_template('processing.html', filename=filename)


if __name__ == '__main__':
    app.run(debug=True)