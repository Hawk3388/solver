"""Train a worksheet gap detector through YOLO transfer learning."""

from ultralytics import YOLO
from pathlib import Path
import yaml

def train(
    data_yaml='dataset/data.yaml',
    model_size='l',  # n, s, m, l, x (l = large, recommended)
    epochs=1500,
    img_size=640,
    batch_size=16,
    project_name='worksheet_yolo',
    run_name='transfer_learning',
    device=0,  # 0 = GPU, 'cpu' = CPU
):
    """
    Train a YOLO model using pretrained weights.
    
    Args:
        data_yaml: Path to the data.yaml file.
        model_size: Model size ('n', 's', 'm', 'l', or 'x').
        epochs: Number of training epochs.
        img_size: Training image size.
        batch_size: GPU-dependent batch size.
        project_name: Project name.
        run_name: Name of this training run.
        device: GPU ID or 'cpu'.
    """
    
    # Confirm that the dataset exists.
    data_path = Path(data_yaml)
    if not data_path.exists():
        print(f"❌ Dataset not found: {data_yaml}")
        print("💡 Run prepare_dataset.py first!")
        return
    
    # Load dataset metadata.
    with open(data_path, 'r', encoding='utf-8') as f:
        dataset_info = yaml.safe_load(f)
    
    print("=" * 70)
    print("🚀 YOLO transfer-learning training")
    print("=" * 70)
    print(f"📦 Model: YOLOv26{model_size}")
    print(f"📁 Dataset: {data_yaml}")
    print(f"🏷️  Classes: {dataset_info.get('names', [])}")
    print(f"🔢 Number of classes: {dataset_info.get('nc', 0)}")
    print(f"⚙️  Epochs: {epochs}")
    print(f"📐 Image size: {img_size}x{img_size}")
    print(f"📦 Batch Size: {batch_size}")
    print(f"🖥️  Device: {'GPU ' + str(device) if device != 'cpu' else 'CPU'}")
    print("=" * 70)
    
    # Load the pretrained model for transfer learning.
    model_config = f'yolo26{model_size}.pt'
    print(f"\n📥 Loading pretrained model: {model_config}")
    
    try:
        model = YOLO(model_config)
    except Exception as e:
        print(f"❌ Failed to load the model: {e}")
        print("💡 The model is downloaded automatically on first use")
        return
    
    print("✅ Pretrained model loaded and ready for class adaptation")
    
    # Start training.
    print("\n🎯 Starting training... This may take several hours.\n")
    
    try:
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            device=device,
            
            # Project settings.
            project=project_name,
            name=run_name,
            exist_ok=False,   # Create a new directory if the name exists.
            
            # Early Stopping & Checkpointing
            patience=0,       # Stop after N epochs without improvement.
            save=True,        # Save checkpoints.
            save_period=250,  # Save every N epochs.
            
            # Validation
            val=True,
            
            # Performance
            workers=4,        # CPU workers used for data loading.
            pretrained=True,  # Transfer learning from pretrained weights.
            
            # Logging
            plots=True,       # Create training plots.
            verbose=True,
        )
        
        print("\n" + "=" * 70)
        print("✅ TRAINING COMPLETE!")
        print("=" * 70)
        
        # Results.
        best_model_path = Path(project_name) / run_name / 'weights' / 'best.pt'
        last_model_path = Path(project_name) / run_name / 'weights' / 'last.pt'
        
        print("\n📊 Model files:")
        print(f"   Best model: {best_model_path}")
        print(f"   Latest model: {last_model_path}")
        
        print("\n📈 Training metrics:")
        print(f"   Results directory: {Path(project_name) / run_name}")
        
        # Run final validation.
        print("\n🔍 Running final validation...")
        metrics = model.val()
        
        print("\n📊 Validation results:")
        print(f"   mAP50: {metrics.box.map50:.4f}")
        print(f"   mAP50-95: {metrics.box.map:.4f}")
        print(f"   Precision: {metrics.box.mp:.4f}")
        print(f"   Recall: {metrics.box.mr:.4f}")
        
        print("\n🎯 Next steps:")
        print(f"   1. Review training plots in: {Path(project_name) / run_name}")
        print("   2. Test the model with:")
        print(f"      model = YOLO('{best_model_path}')")
        print(f"      results = model.predict('test_image.jpg')")
        print("   3. If results are poor:")
        print("      - Collect more data")
        print("      - Review annotations")
        print("      - Train longer or adjust hyperparameters")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training cancelled manually")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


def resume_training(weights_path, epochs=100):
    """
    Resume an interrupted training run.
    
    Args:
        weights_path: Path to last.pt.
        epochs: Additional epochs.
    """
    print(f"🔄 Resuming training from: {weights_path}")
    
    model = YOLO(weights_path)
    results = model.train(resume=True, epochs=epochs)
    
    print("✅ Resumed training complete")


if __name__ == "__main__":
    # ============= CONFIGURATION =============
    
    # Project directory (the script directory).
    SCRIPT_DIR = Path(__file__).parent
    
    # Dataset
    DATA_YAML = str(SCRIPT_DIR / 'dataset' / 'data.yaml')
    
    # Model size: larger models are more accurate but slower.
    # 'n' = nano (~3M parameters, fastest)
    # 's' = small (~9M params)
    # 'm' = medium (~20M params)
    # 'l' = large (~25M parameters) <- RECOMMENDED
    # 'x' = extra large (~50M params)
    MODEL_SIZE = 'l'
    
    # Training parameters.
    EPOCHS = 1500        # More epochs may improve results but take longer.
    IMG_SIZE = 640       # Default: 640; use 1280 for high-resolution images.
    BATCH_SIZE = 16      # Adjust for the available GPU (8, 16, 32, or 64).
    
    # Hardware
    DEVICE = 0           # 0 = first GPU; use 'cpu' for CPU training.
    
    # Project output, stored beside this script.
    PROJECT_NAME = str(SCRIPT_DIR / 'worksheet_yolo')
    RUN_NAME = 'transfer_learning'
    
    # ============= START TRAINING =============
    
    train(
        data_yaml=DATA_YAML,
        model_size=MODEL_SIZE,
        epochs=EPOCHS,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        project_name=PROJECT_NAME,
        run_name=RUN_NAME,
        device=DEVICE
    )
    
    # ============= RESUME TRAINING (OPTIONAL) =============
    # Use this when a training run was interrupted:
    # resume_training('worksheet_yolo/transfer_learning/weights/last.pt', epochs=EPOCHS)
