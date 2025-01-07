import os
from roboflow import Roboflow
import ultralytics
from ultralytics import YOLO

def main():
    # Run ultralytics checks
    ultralytics.checks()
    
    # Get current working directory
    HOME = os.getcwd()
    print(HOME)
    
    # Create datasets directory
    datasets_path = os.path.join(HOME, "datasets")
    os.makedirs(datasets_path, exist_ok=True)
    os.chdir(datasets_path)
    
    # Roboflow setup
    ROBOFLOW_API_KEY = "68SKtwKsvGKs5bJlo87h"  # Replace with your actual API key
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    
    # Access workspace and project
    workspace = rf.workspace("aetos")
    project = workspace.project("thermal1-copy")
    version = project.version(2)
    dataset = version.download("yolov11")
    
    print("Dataset downloaded successfully to:", datasets_path)
    # print("Yaml localtion: ", data=datasets_path+"datasets/thermal1-copy/data.yaml")

    print("Training model Now........................................................")
    model = YOLO("yolo11x.pt")  # load a pretrained model (recommended for training)# Train the model with 2 GPUs
    results = model.train(data=datasets_path+"/thermal1-copy/data.yaml", epochs=10, imgsz=640, plots=True)

if __name__ == "__main__":
    main()
 