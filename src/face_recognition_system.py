#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Système de reconnaissance faciale en temps réel
Utilise la webcam pour détecter et reconnaître les visages
"""

import cv2
import face_recognition
import numpy as np
import os
import pickle
from pathlib import Path


class FaceRecognitionSystem:
    """Système de reconnaissance faciale"""
    
    def __init__(self, known_faces_dir="known_faces", tolerance=0.6):
        """
        Initialise le système de reconnaissance faciale
        
        Args:
            known_faces_dir (str): Répertoire contenant les images des visages connus
            tolerance (float): Seuil de tolérance pour la reconnaissance (plus bas = plus strict)
        """
        self.known_faces_dir = Path(known_faces_dir)
        self.tolerance = tolerance
        self.known_face_encodings = []
        self.known_face_names = []
        self.encodings_file = "face_encodings.pkl"
        
        # Charger les visages connus
        self.load_known_faces()
    
    def load_known_faces(self):
        """Charge les encodages des visages connus depuis le fichier ou les crée"""
        # Vérifier si un fichier d'encodages existe
        if os.path.exists(self.encodings_file):
            print("📂 Chargement des encodages depuis le fichier...")
            with open(self.encodings_file, 'rb') as f:
                data = pickle.load(f)
                self.known_face_encodings = data['encodings']
                self.known_face_names = data['names']
            print(f"✓ {len(self.known_face_names)} visages chargés")
        else:
            print("🔍 Création des encodages des visages connus...")
            self.encode_known_faces()
    
    def encode_known_faces(self):
        """Encode tous les visages du répertoire known_faces"""
        if not self.known_faces_dir.exists():
            print(f"⚠️ Le répertoire {self.known_faces_dir} n'existe pas")
            self.known_faces_dir.mkdir(exist_ok=True)
            print(f"✓ Répertoire créé. Ajoutez des images de visages dans ce dossier.")
            return
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        for person_dir in self.known_faces_dir.iterdir():
            if person_dir.is_dir():
                person_name = person_dir.name
                print(f"  Traitement de {person_name}...")
                
                for image_path in person_dir.iterdir():
                    if image_path.suffix.lower() in image_extensions:
                        try:
                            # Charger l'image
                            image = face_recognition.load_image_file(str(image_path))
                            
                            # Obtenir l'encodage du visage
                            face_encodings = face_recognition.face_encodings(image)
                            
                            if face_encodings:
                                # Prendre le premier visage trouvé
                                self.known_face_encodings.append(face_encodings[0])
                                self.known_face_names.append(person_name)
                                print(f"    ✓ {image_path.name}")
                            else:
                                print(f"    ⚠️ Aucun visage détecté dans {image_path.name}")
                        except Exception as e:
                            print(f"    ✗ Erreur avec {image_path.name}: {e}")
        
        # Sauvegarder les encodages
        if self.known_face_encodings:
            with open(self.encodings_file, 'wb') as f:
                pickle.dump({
                    'encodings': self.known_face_encodings,
                    'names': self.known_face_names
                }, f)
            print(f"✓ {len(self.known_face_encodings)} encodages sauvegardés")
        else:
            print("⚠️ Aucun visage n'a été encodé")
    
    def recognize_faces_in_frame(self, frame):
        """
        Détecte et reconnaît les visages dans une image
        
        Args:
            frame: Image à analyser (format BGR de OpenCV)
            
        Returns:
            tuple: (frame annoté, liste des noms détectés)
        """
        # Convertir BGR (OpenCV) en RGB (face_recognition)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Réduire la taille pour accélérer le traitement
        small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.25, fy=0.25)
        
        # Détecter les visages
        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)
        
        face_names = []
        
        for face_encoding in face_encodings:
            # Comparer avec les visages connus
            matches = face_recognition.compare_faces(
                self.known_face_encodings, 
                face_encoding, 
                tolerance=self.tolerance
            )
            name = "Inconnu"
            
            # Calculer les distances
            if self.known_face_encodings:
                face_distances = face_recognition.face_distance(
                    self.known_face_encodings, 
                    face_encoding
                )
                best_match_index = np.argmin(face_distances)
                
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
                    confidence = (1 - face_distances[best_match_index]) * 100
                    name = f"{name} ({confidence:.1f}%)"
            
            face_names.append(name)
        
        # Dessiner les rectangles et noms sur l'image
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Échelle inverse (on avait réduit à 25%)
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            
            # Dessiner le rectangle
            color = (0, 255, 0) if "Inconnu" not in name else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Dessiner le nom
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        
        return frame, face_names
    
    def start_video_recognition(self):
        """Démarre la reconnaissance faciale en temps réel via webcam"""
        print("\n🎥 Démarrage de la reconnaissance faciale...")
        print("Appuyez sur 'q' pour quitter")
        print("Appuyez sur 's' pour prendre une capture d'écran")
        
        # Ouvrir la webcam
        video_capture = cv2.VideoCapture(0)
        
        if not video_capture.isOpened():
            print("❌ Impossible d'ouvrir la webcam")
            return
        
        frame_count = 0
        
        try:
            while True:
                # Capturer une image
                ret, frame = video_capture.read()
                
                if not ret:
                    print("❌ Impossible de lire l'image")
                    break
                
                # Traiter une image sur deux pour améliorer les performances
                if frame_count % 2 == 0:
                    frame, detected_names = self.recognize_faces_in_frame(frame)
                
                # Afficher le nombre de visages détectés
                cv2.putText(frame, f"Visages: {len(detected_names)}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Afficher l'image
                cv2.imshow('Reconnaissance Faciale', frame)
                
                frame_count += 1
                
                # Gérer les touches
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Sauvegarder une capture d'écran
                    screenshot_path = f"screenshot_{frame_count}.jpg"
                    cv2.imwrite(screenshot_path, frame)
                    print(f"📸 Capture sauvegardée: {screenshot_path}")
        
        finally:
            # Libérer les ressources
            video_capture.release()
            cv2.destroyAllWindows()
            print("\n✓ Reconnaissance faciale arrêtée")
    
    def recognize_in_image(self, image_path):
        """
        Reconnaît les visages dans une image
        
        Args:
            image_path (str): Chemin vers l'image
        """
        print(f"\n🖼️ Analyse de l'image: {image_path}")
        
        # Charger l'image
        frame = cv2.imread(image_path)
        
        if frame is None:
            print("❌ Impossible de charger l'image")
            return
        
        # Reconnaître les visages
        frame, detected_names = self.recognize_faces_in_frame(frame)
        
        print(f"✓ {len(detected_names)} visage(s) détecté(s): {', '.join(detected_names)}")
        
        # Afficher l'image
        cv2.imshow('Reconnaissance Faciale - Image', frame)
        print("Appuyez sur une touche pour fermer...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    """Fonction principale"""
    print("=" * 60)
    print(" 👤 SYSTÈME DE RECONNAISSANCE FACIALE")
    print("=" * 60)
    
    # Créer le système de reconnaissance
    system = FaceRecognitionSystem(known_faces_dir="known_faces")
    
    if not system.known_face_encodings:
        print("\n⚠️ Aucun visage connu n'a été chargé!")
        print("📝 Instructions:")
        print("   1. Créez un dossier pour chaque personne dans 'known_faces/'")
        print("   2. Ajoutez des photos de chaque personne dans son dossier")
        print("   3. Relancez le programme")
        print("\nExemple de structure:")
        print("   known_faces/")
        print("   ├── Jean/")
        print("   │   ├── photo1.jpg")
        print("   │   └── photo2.jpg")
        print("   └── Marie/")
        print("       └── photo1.jpg")
        return
    
    # Menu
    print("\nQue voulez-vous faire?")
    print("1. Reconnaissance en temps réel (webcam)")
    print("2. Analyser une image")
    print("3. Réencoder les visages connus")
    print("4. Quitter")
    
    choice = input("\nVotre choix (1-4): ").strip()
    
    if choice == "1":
        system.start_video_recognition()
    elif choice == "2":
        image_path = input("Chemin de l'image: ").strip()
        system.recognize_in_image(image_path)
    elif choice == "3":
        if os.path.exists(system.encodings_file):
            os.remove(system.encodings_file)
        system.encode_known_faces()
        print("✓ Réencodage terminé")
    elif choice == "4":
        print("Au revoir! 👋")
    else:
        print("❌ Choix invalide")


if __name__ == "__main__":
    main()
