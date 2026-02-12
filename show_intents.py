
import sys
import os


def main():
    print("\n" + "="*70)
    print(" SAAC Intent Classification System")
    print("="*70)
    
    try:
        from saac.detectors.scene_classifier import ClipSceneClassifier
        import cv2
        
        print("\n✓ CLIP classifier available")
        
        print("\nLoading CLIP model...")
        classifier = ClipSceneClassifier(device='cpu')
        
        classifier.list_all_intents(show_descriptions=False)
        
        if len(sys.argv) > 1:
            image_path = sys.argv[1]
            
            if not os.path.exists(image_path):
                print(f"\n❌ Error: File not found: {image_path}")
                sys.exit(1)
            
            print("\n" + "="*70)
            print(f" Classifying Image: {os.path.basename(image_path)}")
            print("="*70)
            
            image = cv2.imread(image_path)
            if image is None:
                print(f"\n❌ Error: Could not read image: {image_path}")
                sys.exit(1)
            
            print(f"\nImage size: {image.shape[1]}x{image.shape[0]}")
            
            print("\n🎯 Top 10 Intent Predictions:")
            print("-"*70)
            top_predictions = classifier.classify_top_k(image, k=10, min_confidence=0.01)
            
            for i, (intent, confidence) in enumerate(top_predictions, 1):
                bar_length = int(confidence * 50)
                bar = "█" * bar_length
                desc = classifier.get_scene_description(intent)
                print(f"{i:2d}. {intent:20s} {bar} {confidence:6.2%}")
                if i <= 3:
                    print(f"    → {desc}")
            
            print("\n" + "="*70)
            primary, primary_conf, alternatives = classifier.classify_with_fallback_chain(image)
            print(f"✅ Selected Intent: {primary}")
            print(f"   Confidence: {primary_conf:.2%}")
            print(f"   {classifier.get_scene_description(primary)}")
            
            if alternatives:
                print(f"\n📋 Alternative Intents:")
                for alt_intent, alt_conf in alternatives[:3]:
                    print(f"   • {alt_intent:20s} ({alt_conf:5.2%})")
            
            print("\n💡 This intent will be used to determine compression priorities")
            print("   for objects detected in your image.")
            
        else:
            print("\n" + "="*70)
            print(" Usage")
            print("="*70)
            print("\nTo classify an image:")
            print(f"  python3 {sys.argv[0]} <image_path>")
            print("\nExample:")
            print(f"  python3 {sys.argv[0]} my_photo.jpg")
            print("\nThis will show which intent category best matches your image")
            print("and how compression priorities will be assigned.")
        
        print("\n" + "="*70 + "\n")
        
    except ImportError as e:
        print(f"\n❌ CLIP not installed: {e}")
        print("\nInstall with:")
        print("  pip install git+https://github.com/openai/CLIP.git")
        print("\nOr use the basic scene classifier (limited to 8 intents):")
        print("  python3 compress.py <image>")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
