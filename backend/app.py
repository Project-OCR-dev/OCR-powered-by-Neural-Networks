from flask import Flask, render_template, request
import os

app = Flask(__name__)
# dossier ou sont sauvegardés les fichiers uploadé
app.config['UPLOAD_FOLDER'] = 'uploads'
# liste des formats de fichiers autorisés
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}


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
    et le sauvegarde dans le dossier uploads/.
    
    Returns:
        str: Message de succès ou d'erreur
    """
    if 'file' not in request.files:
        return "Erreur : aucun fichier envoyé"

    file = request.files['file']

    if file.filename == '':
        return "Erreur : aucun fichier sélectionné"

    if allowed_file(file.filename):
        file.save(os.path.join('uploads', file.filename))
        return "Fichier uploadé avec succès !"
    else:
        return "Erreur : type de fichier non autorisé !"

if __name__ == '__main__':
    app.run(debug=True)