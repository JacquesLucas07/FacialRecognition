#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script pour enregistrer de nouveaux visages
Capture des photos via webcam et les enregistre dans le répertoire approprié
"""

import cv2
import os
from pathlib import Path
import time


class FaceRegistration:
    """Classe pour enregistrer de nouveaux visages"""
    
    def __init__(self, known_faces_dir="known_faces"):
        """
        Initialise le système d'enregistrement
        
        Args:
            known_faces_dir (str): Répertoire où sauvegarder les visages
        """
        self.known_faces_dir = Path(known_faces_dir)
        self.known_faces_dir.mkdir(exist_ok=True)
        
        # Charger le classificateur de visages Haar Cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
    
    def capture_faces_from_webcam(self, person_name, num_photos=10):
        """
        Capture plusieurs photos d'une personne via la webcam
        
        Args:
            person_name (str): Nom de la personne
            num_photos (int): Nombre de photos à capturer
        """
        # Créer le dossier pour cette personne
        person_dir = self.known_faces_dir / person_name
        person_dir.mkdir(exist_ok=True)
        
        print(f"\n📸 Capture de {num_photos} photos pour {person_name}")
        print("Instructions:")
        print("  - Regardez la caméra")
        print("  - Changez légèrement de position entre chaque photo")
        print("  - Appuyez sur ESPACE pour capturer une photo")
        print("  - Appuyez sur 'q' pour quitter")
        
        # Ouvrir la webcam
        video_capture = cv2.VideoCapture(0)
        
        if not video_capture.isOpened():
            print("❌ Impossible d'ouvrir la webcam")
            return
        
        photo_count = 0
        last_capture_time = 0
        capture_cooldown = 1.0  # Délai minimum entre deux captures (secondes)
        
        try:
            while photo_count < num_photos:
                ret, frame = video_capture.read()
                
                if not ret:
                    print("❌ Impossible de lire l'image")
                    break
                
                # Copie pour l'affichage
                display_frame = frame.copy()
                
                # Détecter les visages
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(100, 100)
                )
                
                # Dessiner les rectangles autour des visages
                for (x, y, w, h) in faces:
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(display_frame, "Visage détecté", (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Afficher le compteur
                cv2.putText(display_frame, f"Photos: {photo_count}/{num_photos}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display_frame, "ESPACE = Capturer | Q = Quitter", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Afficher l'image
                cv2.imshow(f'Enregistrement - {person_name}', display_frame)
                
                # Gérer les touches
                key = cv2.waitKey(1) & 0xFF
                current_time = time.time()
                
                if key == ord(' ') and (current_time - last_capture_time) > capture_cooldown:
                    if len(faces) > 0:
                        # Sauvegarder l'image
                        photo_count += 1
                        filename = f"{person_name}_{photo_count:03d}.jpg"
                        filepath = person_dir / filename
                        cv2.imwrite(str(filepath), frame)
                        print(f"  ✓ Photo {photo_count}/{num_photos} sauvegardée: {filename}")
                        last_capture_time = current_time
                        
                        # Flash visuel
                        white_frame = display_frame.copy()
                        white_frame[:] = (255, 255, 255)
                        cv2.imshow(f'Enregistrement - {person_name}', white_frame)
                        cv2.waitKey(100)
                    else:
                        print("  ⚠️ Aucun visage détecté. Réessayez.")
                
                elif key == ord('q'):
                    print("\n⚠️ Enregistrement annulé")
                    break
        
        finally:
            video_capture.release()
            cv2.destroyAllWindows()
        
        if photo_count >= num_photos:
            print(f"\n✓ Enregistrement terminé! {photo_count} photos sauvegardées dans {person_dir}")
            print("💡 Conseil: Relancez le système de reconnaissance pour encoder ces nouveaux visages")
        
        return photo_count
    
    def capture_single_photo(self, person_name, photo_name=None):
        """
        Capture une seule photo
        
        Args:
            person_name (str): Nom de la personne
            photo_name (str): Nom de la photo (optionnel)
        """
        # Créer le dossier pour cette personne
        person_dir = self.known_faces_dir / person_name
        person_dir.mkdir(exist_ok=True)
        
        print(f"\n📸 Capture d'une photo pour {person_name}")
        print("Appuyez sur ESPACE pour capturer")
        
        # Ouvrir la webcam
        video_capture = cv2.VideoCapture(0)
        
        if not video_capture.isOpened():
            print("❌ Impossible d'ouvrir la webcam")
            return False
        
        captured = False
        
        try:
            while not captured:
                ret, frame = video_capture.read()
                
                if not ret:
                    print("❌ Impossible de lire l'image")
                    break
                
                # Détecter les visages
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
                
                # Dessiner les rectangles
                display_frame = frame.copy()
                for (x, y, w, h) in faces:
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                cv2.putText(display_frame, "ESPACE = Capturer | Q = Quitter", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('Capture Photo', display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord(' '):
                    if len(faces) > 0:
                        # Générer un nom de fichier
                        if photo_name is None:
                            timestamp = int(time.time())
                            photo_name = f"{person_name}_{timestamp}.jpg"
                        elif not photo_name.endswith(('.jpg', '.jpeg', '.png')):
                            photo_name += '.jpg'
                        
                        filepath = person_dir / photo_name
                        cv2.imwrite(str(filepath), frame)
                        print(f"✓ Photo sauvegardée: {filepath}")
                        captured = True
                    else:
                        print("⚠️ Aucun visage détecté. Réessayez.")
                
                elif key == ord('q'):
                    print("⚠️ Capture annulée")
                    break
        
        finally:
            video_capture.release()
            cv2.destroyAllWindows()
        
        return captured
    
    def add_from_file(self, person_name, image_path):
        """
        Ajoute une photo depuis un fichier existant
        
        Args:
            person_name (str): Nom de la personne
            image_path (str): Chemin vers l'image
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            print(f"❌ Le fichier {image_path} n'existe pas")
            return False
        
        # Créer le dossier pour cette personne
        person_dir = self.known_faces_dir / person_name
        person_dir.mkdir(exist_ok=True)
        
        # Copier l'image
        destination = person_dir / image_path.name
        
        import shutil
        shutil.copy(str(image_path), str(destination))
        
        print(f"✓ Image ajoutée: {destination}")
        return True


def main():
    """Fonction principale"""
    print("=" * 60)
    print(" 📸 ENREGISTREMENT DE NOUVEAUX VISAGES")
    print("=" * 60)
    
    registration = FaceRegistration(known_faces_dir="known_faces")
    
    print("\nQue voulez-vous faire?")
    print("1. Enregistrer un nouveau visage (série de photos)")
    print("2. Capturer une seule photo")
    print("3. Ajouter une image depuis un fichier")
    print("4. Quitter")
    
    choice = input("\nVotre choix (1-4): ").strip()
    
    if choice == "1":
        person_name = input("Nom de la personne: ").strip()
        if person_name:
            try:
                num_photos = int(input("Nombre de photos à capturer (défaut: 10): ").strip() or "10")
                registration.capture_faces_from_webcam(person_name, num_photos)
            except ValueError:
                print("❌ Nombre invalide")
        else:
            print("❌ Le nom ne peut pas être vide")
    
    elif choice == "2":
        person_name = input("Nom de la personne: ").strip()
        if person_name:
            registration.capture_single_photo(person_name)
        else:
            print("❌ Le nom ne peut pas être vide")
    
    elif choice == "3":
        person_name = input("Nom de la personne: ").strip()
        image_path = input("Chemin de l'image: ").strip()
        if person_name and image_path:
            registration.add_from_file(person_name, image_path)
        else:
            print("❌ Informations manquantes")
    
    elif choice == "4":
        print("Au revoir! 👋")
    
    else:
        print("❌ Choix invalide")


if __name__ == "__main__":
    main()
