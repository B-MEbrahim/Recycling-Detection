import torch
from ultralytics import YOLO
import yaml
from pathlib import Path
import argparse
import shutil

def train_teacher_model(config_path: str):
    """Train teacher YOLO model on recycling dataset"""
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load pretrained YOLO model
    print(f"Loading pretrained model: {config['model']}")
    model = YOLO(config['model'])
    
    # Train the model
    results = model.train(
        data='data/dataset.yaml',
        epochs=config['epochs'],
        imgsz=config['imgsz'],
        batch=config['batch_size'],
        name=config['name'],
        project=config['project'],
        patience=config.get('patience', 50),
        save=True,
        save_period=config.get('save_period', 10),
        device=config.get('device', '0'),
        workers=config.get('workers', 8),
        optimizer=config.get('optimizer', 'auto'),
        lr0=config.get('lr0', 0.01),
        lrf=config.get('lrf', 0.01),
        momentum=config.get('momentum', 0.937),
        weight_decay=config.get('weight_decay', 0.0005),
        warmup_epochs=config.get('warmup_epochs', 3),
        warmup_momentum=config.get('warmup_momentum', 0.8),
        box=config.get('box', 7.5),
        cls=config.get('cls', 0.5),
        dfl=config.get('dfl', 1.5),
        amp=config.get('amp', True)
    )
    
    # Save best model
    best_model_path = Path(f"{config['project']}/{config['name']}/weights/best.pt")
    if best_model_path.exists():
        destination = Path("models/teacher/best.pt")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(best_model_path, destination)
        print(f"Best teacher model saved to {destination}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/teacher_config.yaml')
    args = parser.parse_args()
    
    train_teacher_model(args.config)