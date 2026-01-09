#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fonctions utilitaires pour le système de reconnaissance faciale
"""

import cv2
import face_recognition
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import pickle

#TODO réangencer les paramètres des fonctions
#TODO Regler problème import de libres externes
def load_image(image_path: str) -> np.ndarray:
    """
    Charge une image depuis un fichier
    
    Args:
        image_path (str): Chemin vers l'image
        
    Returns:
        np.ndarray: Image au format numpy array
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Impossible de charger l'image: {image_path}")
    return image


def detect_faces_opencv(image: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Détecte les visages dans une image avec OpenCV Haar Cascade
    
    Args:
        image (np.ndarray): Image à analyser
        
    Returns:
        List[Tuple]: Liste de rectangles (x, y, w, h)
    """
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    return faces


def get_face_encoding(image_path: str) -> List[np.ndarray]:
    """
    Obtient l'encodage d'un visage depuis une image
    
    Args:
        image_path (str): Chemin vers l'image
        
    Returns:
        List[np.ndarray]: Liste des encodages des visages détectés
    """
    image = face_recognition.load_image_file(image_path)
    return face_recognition.face_encodings(image)


def compare_faces(known_encodings: List[np.ndarray], 
                  face_encoding: np.ndarray, 
                  tolerance: float = 0.6) -> Tuple[List[bool], np.ndarray]:
    """
    Compare un visage avec une liste de visages connus
    
    Args:
        known_encodings: Liste des encodages connus
        face_encoding: Encodage du visage à comparer
        tolerance: Seuil de tolérance (plus bas = plus strict)
        
    Returns:
        Tuple: (Liste de matches booléens, distances)
    """
    matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance)
    distances = face_recognition.face_distance(known_encodings, face_encoding)
    
    return matches, distances


def draw_face_box(image: np.ndarray, 
                  location: Tuple[int, int, int, int], 
                  name: str, 
                  color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """
    Dessine un rectangle et un nom autour d'un visage
    
    Args:
        image: Image sur laquelle dessiner
        location: Coordonnées du visage (top, right, bottom, left)
        name: Nom à afficher
        color: Couleur du rectangle (B, G, R)
        
    Returns:
        np.ndarray: Image modifiée
    """
    top, right, bottom, left = location
    
    # Rectangle autour du visage
    cv2.rectangle(image, (left, top), (right, bottom), color, 2)
    
    # Rectangle pour le texte
    cv2.rectangle(image, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
    
    # Texte
    cv2.putText(image, name, (left + 6, bottom - 6), 
               cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
    
    return image


def save_encodings(encodings: List[np.ndarray], 
                   names: List[str], 
                   filename: str = "face_encodings.pkl"):
    """
    Sauvegarde les encodages dans un fichier
    
    Args:
        encodings: Liste des encodages
        names: Liste des noms correspondants
        filename: Nom du fichier de sortie
    """
    data = {
        'encodings': encodings,
        'names': names
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def load_encodings(filename: str = "face_encodings.pkl") -> Dict:
    """
    Charge les encodages depuis un fichier
    
    Args:
        filename: Nom du fichier à charger
        
    Returns:
        Dict: Dictionnaire contenant 'encodings' et 'names'
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)


def resize_image(image: np.ndarray, 
                 max_width: int = 800, 
                 max_height: int = 600) -> np.ndarray:
    """
    Redimensionne une image en conservant le ratio
    
    Args:
        image: Image à redimensionner
        max_width: Largeur maximale
        max_height: Hauteur maximale
        
    Returns:
        np.ndarray: Image redimensionnée
    """
    height, width = image.shape[:2]
    
    # Calculer le ratio
    ratio = min(max_width / width, max_height / height)
    
    if ratio < 1:
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        return cv2.resize(image, (new_width, new_height))
    
    return image


def blur_face(image: np.ndarray, 
              location: Tuple[int, int, int, int], 
              blur_amount: int = 99) -> np.ndarray:
    """
    Floute une région du visage
    
    Args:
        image: Image à modifier
        location: Coordonnées du visage (top, right, bottom, left)
        blur_amount: Intensité du flou (nombre impair)
        
    Returns:
        np.ndarray: Image avec le visage flouté
    """
    top, right, bottom, left = location
    
    # Extraire la région du visage
    face_region = image[top:bottom, left:right]
    
    # Appliquer le flou
    blurred_face = cv2.GaussianBlur(face_region, (blur_amount, blur_amount), 30)
    
    # Remplacer dans l'image originale
    image[top:bottom, left:right] = blurred_face
    
    return image


def create_face_mosaic(image_dir: str, output_path: str = "mosaic.jpg"):
    """
    Crée une mosaïque de toutes les images de visages
    
    Args:
        image_dir: Répertoire contenant les images
        output_path: Chemin de sortie pour la mosaïque
    """
    image_dir = Path(image_dir)
    images = []
    
    # Charger toutes les images
    for img_path in image_dir.glob("**/*.jpg"):
        img = cv2.imread(str(img_path))
        if img is not None:
            # Redimensionner à une taille fixe
            img = cv2.resize(img, (200, 200))
            images.append(img)
    
    if not images:
        print("Aucune image trouvée")
        return
    
    # Calculer les dimensions de la mosaïque
    n_images = len(images)
    cols = int(np.ceil(np.sqrt(n_images)))
    rows = int(np.ceil(n_images / cols))
    
    # Créer la mosaïque
    mosaic = np.zeros((rows * 200, cols * 200, 3), dtype=np.uint8)
    
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        mosaic[row*200:(row+1)*200, col*200:(col+1)*200] = img
    
    cv2.imwrite(output_path, mosaic)
    print(f"✓ Mosaïque créée: {output_path}")


def get_image_quality(image_path: str) -> Dict[str, float]:
    """
    Évalue la qualité d'une image pour la reconnaissance faciale
    
    Args:
        image_path: Chemin vers l'image
        
    Returns:
        Dict: Scores de qualité
    """
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Luminosité moyenne
    brightness = np.mean(gray)
    
    # Contraste (écart-type)
    contrast = np.std(gray)
    
    # Netteté (variance du Laplacien)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = laplacian.var()
    
    return {
        'brightness': brightness,
        'contrast': contrast,
        'sharpness': sharpness,
        'quality_score': (contrast * sharpness) / 10000  # Score composite
    }


if __name__ == "__main__":
    print("Module utilitaire pour la reconnaissance faciale")
    print("Importez ce module dans vos scripts pour utiliser ces fonctions")
