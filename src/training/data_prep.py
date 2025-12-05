import os
import yaml
from roboflow import Roboflow
from pathlib import Path
import shutil

class DatasetDownloader:
  def __init__(self, api_key: str, workspace: str, project_name: str):
    self.api_key = api_key
    self.workspace = workspace
    self.project_name = project_name
    self.rf = Roboflow(api_key=api_key)

  def download_dataset(self, version: int = 2, format: str = 'yolov8'):
    project = self.rf.workspace(self.workspace).project(self.project_name)
    dataset = project.version(version).download(format)
    self.create_dataset_yaml(dataset.location)

    return Path(dataset.location)

  def create_dataset_yaml(self, dataset_path: str):
    dataset_path = Path(dataset_path)

    data_yaml = dataset_path / "data.yaml"
    if data_yaml.exists():
      with open(data_yaml, 'r') as f:
        data = yaml.safe_load(f)

      config = {
          'path': str(dataset_path.absolute()),
          'train': 'train/images',
          'val': 'valid/images',
          'test': 'test/images',
          'nc': len(data['names']),
          'names': data['names']
      }

      # Ensure the 'data' directory exists
      output_dir = Path('data')
      output_dir.mkdir(parents=True, exist_ok=True)

      with open(output_dir / 'dataset.yaml', 'w') as f:
        yaml.dump(config, f)

        print(f"Dataset configuration saved to {output_dir}/dataset.yaml")

        