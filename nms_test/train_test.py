from ultralytics import YOLO
from ultralytics.utils import SETTINGS
from ultralytics.models.yolo.detect import SSODTrainer

# Build an untrained YOLO11n model (random init, no pretrained weights)
model = YOLO("yolo11n.yaml")
#model = YOLO("yolo11n.pt")


# Train the model on the dataset
train_results = model.train(
    #data="coco.yaml",  # Path to dataset configuration file
    #data="african-wildlife_10p.yaml",
    data="kitti_10p.yaml",
    epochs=350,  # Number of training epochs
    imgsz=640,  # Image size for training
    device="3",  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
    trainer=SSODTrainer,
    # pretrained=True,  # ensure no pretrained weights are used
    burn_in_epochs=220,
    domain_adaptation=True,
    project="runs",  # wandb project name (auto-created if not exists)
    da_loss_weights = 0.5,
    name="kitti_10p_domain_adaptation_0.5",  # wandb run name
)

