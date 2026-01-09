# 👤 Système de Reconnaissance Faciale

Un système complet de reconnaissance faciale en Python utilisant OpenCV et face_recognition. Ce projet permet de détecter, enregistrer et reconnaître des visages en temps réel via webcam ou depuis des images.

## 🌟 Fonctionnalités

- ✅ **Reconnaissance en temps réel** via webcam
- ✅ **Enregistrement de nouveaux visages** avec capture automatique
- ✅ **Analyse d'images** pour reconnaître les visages
- ✅ **Encodage optimisé** des visages avec mise en cache
- ✅ **Interface intuitive** en ligne de commande
- ✅ **Détection multi-visages** dans une même image
- ✅ **Score de confiance** pour chaque reconnaissance

## 📋 Prérequis

- Python 3.7 ou supérieur
- Webcam (pour la capture en temps réel)
- Windows, macOS ou Linux

## 🚀 Installation

### 1. Cloner le dépôt

```bash
git clone <url-du-repo>
cd FacialRecognition
```

### 2. Créer un environnement virtuel (recommandé)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

**Note**: L'installation de `dlib` peut nécessiter CMake et des outils de compilation:
- **Windows**: Installez Visual Studio Build Tools
- **Linux**: `sudo apt-get install cmake build-essential`
- **Mac**: `brew install cmake`

## 📁 Structure du projet

```
FacialRecognition/
├── src/
│   ├── face_recognition_system.py  # Système principal de reconnaissance
│   ├── register_face.py            # Script d'enregistrement de visages
│   └── utils.py                    # Fonctions utilitaires
├── known_faces/                    # Dossier pour les visages connus
│   ├── Jean/                       # Un dossier par personne
│   │   ├── photo1.jpg
│   │   └── photo2.jpg
│   └── Marie/
│       └── photo1.jpg
├── requirements.txt                # Dépendances Python
├── face_encodings.pkl             # Encodages des visages (généré auto)
├── LICENSE
└── README.md
```

## 📖 Guide d'utilisation

### 1. Enregistrer de nouveaux visages

Avant d'utiliser le système de reconnaissance, vous devez enregistrer des visages connus:

```bash
cd src
python register_face.py
```

**Options disponibles:**
- **Option 1**: Capture automatique de plusieurs photos via webcam
- **Option 2**: Capture d'une seule photo
- **Option 3**: Ajouter une image existante depuis un fichier

**Conseils pour de meilleurs résultats:**
- Prenez 5-10 photos par personne
- Variez les angles et expressions
- Assurez une bonne luminosité
- Évitez les ombres sur le visage

### 2. Lancer la reconnaissance faciale

```bash
cd src
python face_recognition_system.py
```

**Options disponibles:**
1. **Reconnaissance en temps réel**: Utilise la webcam
2. **Analyser une image**: Reconnaît les visages dans une image
3. **Réencoder les visages**: Reconstruit la base de données d'encodages
4. **Quitter**

### 3. Utilisation en temps réel

Une fois la reconnaissance lancée:
- **Q**: Quitter l'application
- **S**: Prendre une capture d'écran

## 💻 Exemples de code

### Reconnaissance dans une image

```python
from face_recognition_system import FaceRecognitionSystem

# Créer le système
system = FaceRecognitionSystem(known_faces_dir="known_faces")

# Analyser une image
system.recognize_in_image("photo_groupe.jpg")
```

### Utilisation des fonctions utilitaires

```python
from utils import get_face_encoding, compare_faces

# Obtenir l'encodage d'un visage
encodings = get_face_encoding("photo.jpg")

# Comparer avec des visages connus
matches, distances = compare_faces(known_encodings, encodings[0])
```

## ⚙️ Configuration

### Ajuster la sensibilité

Dans `face_recognition_system.py`, vous pouvez modifier le paramètre `tolerance`:

```python
system = FaceRecognitionSystem(
    known_faces_dir="known_faces",
    tolerance=0.6  # Plus bas = plus strict (défaut: 0.6)
)
```

Valeurs recommandées:
- `0.4`: Très strict (peu de faux positifs)
- `0.6`: Équilibré (défaut)
- `0.7`: Permissif (plus de faux positifs)

## 🛠️ Dépannage

### La webcam ne fonctionne pas
```python
# Essayez un autre index de caméra
video_capture = cv2.VideoCapture(1)  # Au lieu de 0
```

### Erreur d'installation de dlib
```bash
# Windows: Installez depuis un wheel précompilé
pip install https://github.com/jloh02/dlib/releases/download/v19.24.1/dlib-19.24.1-cp39-cp39-win_amd64.whl
```

### Reconnaissance lente
- Réduisez la résolution de la webcam
- Augmentez l'intervalle entre les frames (modifier `frame_count % 2`)

## 📊 Performance

- **Détection**: ~30 FPS sur webcam 720p
- **Reconnaissance**: ~10-15 FPS avec 10 visages connus
- **Précision**: ~99% avec de bonnes conditions d'éclairage

## 🔒 Confidentialité

- Les encodages de visages sont stockés localement
- Aucune donnée n'est envoyée sur Internet
- Les photos sont stockées uniquement sur votre machine

## 🤝 Contribution

Les contributions sont les bienvenues! N'hésitez pas à:
1. Fork le projet
2. Créer une branche (`git checkout -b feature/AmazingFeature`)
3. Commit vos changements (`git commit -m 'Add AmazingFeature'`)
4. Push vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

## 📝 Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

## 🙏 Remerciements

- [face_recognition](https://github.com/ageitgey/face_recognition) par Adam Geitgey
- [OpenCV](https://opencv.org/)
- [dlib](http://dlib.net/)

## 📧 Contact

Pour toute question ou suggestion, n'hésitez pas à ouvrir une issue!

---

**Made with ❤️ and Python**