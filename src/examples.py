#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script d'exemple pour démontrer l'utilisation du système
"""

from face_recognition_system import FaceRecognitionSystem
from utils import get_image_quality
import cv2


def exemple_analyse_qualite():
    """Exemple d'analyse de la qualité d'une image"""
    print("=== Analyse de qualité d'image ===\n")
    
    image_path = input("Chemin de l'image à analyser: ").strip()
    
    try:
        quality = get_image_quality(image_path)
        
        print(f"\n📊 Résultats:")
        print(f"  Luminosité: {quality['brightness']:.1f}/255")
        print(f"  Contraste: {quality['contrast']:.1f}")
        print(f"  Netteté: {quality['sharpness']:.1f}")
        print(f"  Score qualité: {quality['quality_score']:.2f}")
        
        # Recommandations
        print("\n💡 Recommandations:")
        if quality['brightness'] < 50:
            print("  ⚠️ Image trop sombre - augmentez l'éclairage")
        elif quality['brightness'] > 200:
            print("  ⚠️ Image surexposée - réduisez l'éclairage")
        else:
            print("  ✓ Luminosité correcte")
        
        if quality['sharpness'] < 100:
            print("  ⚠️ Image floue - utilisez une meilleure caméra")
        else:
            print("  ✓ Netteté correcte")
            
    except Exception as e:
        print(f"❌ Erreur: {e}")


def exemple_reconnaissance_batch():
    """Exemple de reconnaissance sur plusieurs images"""
    print("=== Reconnaissance batch ===\n")
    
    system = FaceRecognitionSystem(known_faces_dir="known_faces")
    
    if not system.known_face_encodings:
        print("❌ Aucun visage connu. Enregistrez des visages d'abord.")
        return
    
    images = input("Chemins des images (séparés par des virgules): ").strip().split(',')
    
    results = {}
    
    for image_path in images:
        image_path = image_path.strip()
        print(f"\n📷 Analyse: {image_path}")
        
        try:
            frame = cv2.imread(image_path)
            if frame is not None:
                _, names = system.recognize_faces_in_frame(frame)
                results[image_path] = names
                print(f"  ✓ {len(names)} visage(s) détecté(s): {', '.join(names)}")
            else:
                print(f"  ❌ Impossible de charger l'image")
        except Exception as e:
            print(f"  ❌ Erreur: {e}")
    
    # Résumé
    print("\n" + "="*60)
    print("📊 RÉSUMÉ")
    print("="*60)
    for path, names in results.items():
        print(f"{path}: {len(names)} visage(s)")


def exemple_statistiques():
    """Affiche des statistiques sur les visages enregistrés"""
    print("=== Statistiques ===\n")
    
    system = FaceRecognitionSystem(known_faces_dir="known_faces")
    
    if not system.known_face_encodings:
        print("❌ Aucun visage connu.")
        return
    
    # Compter les visages par personne
    from collections import Counter
    counts = Counter(system.known_face_names)
    
    print(f"📊 Nombre total de visages: {len(system.known_face_encodings)}")
    print(f"👥 Nombre de personnes: {len(counts)}")
    print(f"\n📋 Détails:")
    
    for name, count in counts.most_common():
        print(f"  • {name}: {count} photo(s)")


def menu_principal():
    """Menu principal des exemples"""
    print("=" * 60)
    print(" 🎓 EXEMPLES D'UTILISATION")
    print("=" * 60)
    print("\n1. Analyser la qualité d'une image")
    print("2. Reconnaissance sur plusieurs images (batch)")
    print("3. Afficher les statistiques")
    print("4. Quitter")
    
    choice = input("\nVotre choix (1-4): ").strip()
    
    if choice == "1":
        exemple_analyse_qualite()
    elif choice == "2":
        exemple_reconnaissance_batch()
    elif choice == "3":
        exemple_statistiques()
    elif choice == "4":
        print("Au revoir! 👋")
    else:
        print("❌ Choix invalide")


if __name__ == "__main__":
    menu_principal()
