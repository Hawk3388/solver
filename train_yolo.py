"""
YOLO Training mit Transfer Learning für Arbeitsblatt Freie Stellen Erkennung
Trainiert YOLOv11m mit vortrainierten Gewichten (nur eine Klasse: freie_stelle)
"""

from ultralytics import YOLO
from pathlib import Path
import yaml

def train_from_scratch(
    data_yaml='dataset/data.yaml',
    model_size='m',  # n, s, m, l, x (m = medium)
    epochs=300,
    img_size=640,
    batch_size=16,
    project_name='arbeitsblatt_yolo',
    run_name='from_scratch_m',
    device=0,  # 0 = GPU, 'cpu' = CPU
):
    """
    Trainiert YOLO Modell von Scratch
    
    Args:
        data_yaml: Pfad zur data.yaml Datei
        model_size: Modellgröße ('n', 's', 'm', 'l', 'x')
        epochs: Anzahl Trainings-Epochen
        img_size: Bildgröße für Training
        batch_size: Batch Size (GPU-abhängig)
        project_name: Name des Projekts
        run_name: Name dieses Trainings-Runs
        device: GPU ID oder 'cpu'
    """
    
    # Überprüfe ob Dataset existiert
    data_path = Path(data_yaml)
    if not data_path.exists():
        print(f"❌ Dataset nicht gefunden: {data_yaml}")
        print(f"💡 Führe zuerst prepare_dataset.py aus!")
        return
    
    # Dataset Info laden
    with open(data_path, 'r', encoding='utf-8') as f:
        dataset_info = yaml.safe_load(f)
    
    print("=" * 70)
    print("🚀 YOLO Training mit Transfer Learning")
    print("=" * 70)
    print(f"📦 Modell: YOLOv11{model_size}")
    print(f"📁 Dataset: {data_yaml}")
    print(f"🏷️  Klassen: {dataset_info.get('names', [])}")
    print(f"🔢 Anzahl Klassen: {dataset_info.get('nc', 0)}")
    print(f"⚙️  Epochen: {epochs}")
    print(f"📐 Bildgröße: {img_size}x{img_size}")
    print(f"📦 Batch Size: {batch_size}")
    print(f"🖥️  Device: {'GPU ' + str(device) if device != 'cpu' else 'CPU'}")
    print("=" * 70)
    
    # Info: Transfer Learning
    print("\n✅ Transfer Learning (vortrainiertes Modell wird angepasst)")
    print("   - Benötigt weniger Daten (50-100+ Bilder reichen)")
    print("   - Schnelleres Training")
    print("   - Bessere Ergebnisse als Training von Scratch")
    print("   - Vortrainierte Features werden für deine Klasse angepasst\n")
    
    response = input("Fortfahren? (j/n): ").lower()
    if response != 'j':
        print("❌ Training abgebrochen")
        return
    
    # Vortrainiertes Modell laden (Transfer Learning)
    model_config = f'yolo26{model_size}.pt'
    print(f"\n📥 Lade vortrainiertes Modell: {model_config}")
    
    try:
        model = YOLO(model_config)
    except Exception as e:
        print(f"❌ Fehler beim Laden des Modells: {e}")
        print(f"💡 Das Modell wird automatisch heruntergeladen beim ersten Mal")
        return
    
    print("✅ Vortrainiertes Modell geladen (wird für deine Klasse angepasst)")
    
    # Training starten
    print(f"\n🎯 Starte Training... (Das kann mehrere Stunden dauern)\n")
    
    try:
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            device=device,
            
            # Projekt-Einstellungen
            project=project_name,
            name=run_name,
            exist_ok=False,  # Erstelle neuen Ordner wenn Name existiert
            
            # Optimizer-Einstellungen
            optimizer='SGD',  # Standard: Stochastic Gradient Descent
            lr0=0.01,         # Initiale Learning Rate
            lrf=0.01,         # Finale Learning Rate (lr0 * lrf)
            momentum=0.937,   # SGD Momentum
            weight_decay=0.0005,  # Weight Decay
            
            # Augmentation (sehr wichtig bei wenig Daten!)
            augment=True,
            hsv_h=0.015,      # Hue Augmentation
            hsv_s=0.7,        # Saturation Augmentation
            hsv_v=0.4,        # Value Augmentation
            degrees=0.0,      # Rotation (bei Arbeitsblättern eher 0)
            translate=0.1,    # Translation
            scale=0.5,        # Scaling
            shear=0.0,        # Shear (bei Arbeitsblättern 0)
            perspective=0.0,  # Perspective Warp (bei Arbeitsblättern 0)
            flipud=0.0,       # Vertical Flip (nicht für Arbeitsblätter!)
            fliplr=0.0,       # Horizontal Flip (nicht für Arbeitsblätter!)
            mosaic=1.0,       # Mosaic Augmentation
            mixup=0.0,        # MixUp (bei 1 Klasse weniger sinnvoll)
            
            # Early Stopping & Checkpointing
            patience=50,      # Stoppe wenn keine Verbesserung nach N Epochen
            save=True,        # Speichere Checkpoints
            save_period=10,   # Speichere alle N Epochen
            
            # Validation
            val=True,
            
            # Performance
            workers=8,        # Anzahl CPU Worker für Dataloading
            pretrained=True,  # Transfer Learning von vortrainiertem Modell
            
            # Logging
            plots=True,       # Erstelle Trainings-Plots
            verbose=True,
        )
        
        print("\n" + "=" * 70)
        print("✅ TRAINING ABGESCHLOSSEN!")
        print("=" * 70)
        
        # Ergebnisse
        best_model_path = Path(project_name) / run_name / 'weights' / 'best.pt'
        last_model_path = Path(project_name) / run_name / 'weights' / 'last.pt'
        
        print(f"\n📊 Modell-Dateien:")
        print(f"   Bestes Modell: {best_model_path}")
        print(f"   Letztes Modell: {last_model_path}")
        
        print(f"\n📈 Trainings-Metriken:")
        print(f"   Results-Ordner: {Path(project_name) / run_name}")
        
        # Validierung durchführen
        print(f"\n🔍 Führe finale Validierung durch...")
        metrics = model.val()
        
        print(f"\n📊 Validierungs-Ergebnisse:")
        print(f"   mAP50: {metrics.box.map50:.4f}")
        print(f"   mAP50-95: {metrics.box.map:.4f}")
        print(f"   Precision: {metrics.box.mp:.4f}")
        print(f"   Recall: {metrics.box.mr:.4f}")
        
        print(f"\n🎯 Nächste Schritte:")
        print(f"   1. Überprüfe Trainings-Plots in: {Path(project_name) / run_name}")
        print(f"   2. Teste das Modell mit:")
        print(f"      model = YOLO('{best_model_path}')")
        print(f"      results = model.predict('test_image.jpg')")
        print(f"   3. Bei schlechten Ergebnissen:")
        print(f"      - Mehr Daten sammeln")
        print(f"      - Annotationen überprüfen")
        print(f"      - Mehr Epochen trainieren oder Hyperparameter anpassen")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training manuell abgebrochen")
    except Exception as e:
        print(f"\n❌ Fehler beim Training: {e}")
        import traceback
        traceback.print_exc()


def resume_training(weights_path, epochs=100):
    """
    Fortsetzung eines unterbrochenen Trainings
    
    Args:
        weights_path: Pfad zu last.pt
        epochs: Zusätzliche Epochen
    """
    print(f"🔄 Setze Training fort von: {weights_path}")
    
    model = YOLO(weights_path)
    results = model.train(resume=True, epochs=epochs)
    
    print("✅ Fortgesetztes Training abgeschlossen")


if __name__ == "__main__":
    # ============= KONFIGURATION =============
    
    # Projekt-Ordner (Verzeichnis des Skripts)
    SCRIPT_DIR = Path(__file__).parent
    
    # Dataset
    DATA_YAML = SCRIPT_DIR / 'dataset' / 'data.yaml'
    
    # Modell-Größe (je größer, desto genauer aber langsamer)
    # 'n' = nano (~3M params, schnellste)
    # 's' = small (~9M params)
    # 'm' = medium (~20M params) ← EMPFOHLEN
    # 'l' = large (~25M params)
    # 'x' = extra large (~50M params)
    MODEL_SIZE = 'm'
    
    # Training-Parameter
    EPOCHS = 100        # 100-200 für Transfer Learning (weniger als von Scratch)
    IMG_SIZE = 640       # Standard: 640, für hochauflösende Bilder: 1280
    BATCH_SIZE = 64      # Anpassen je nach GPU (8, 16, 32, 64)
    
    # Hardware
    DEVICE = 0           # 0 = erste GPU, 'cpu' für CPU
    
    # Projekt (wird im Skript-Ordner gespeichert)
    PROJECT_NAME = str(SCRIPT_DIR / 'arbeitsblatt_yolo')
    RUN_NAME = 'transfer_learning'
    
    # ============= TRAINING STARTEN =============
    
    train_from_scratch(
        data_yaml=str(DATA_YAML),
        model_size=MODEL_SIZE,
        epochs=EPOCHS,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        project_name=PROJECT_NAME,
        run_name=RUN_NAME,
        device=DEVICE
    )
    
    # ============= TRAINING FORTSETZEN (Optional) =============
    # Wenn Training unterbrochen wurde:
    # resume_training('arbeitsblatt_yolo/from_scratch_m/weights/last.pt', epochs=100)
