import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
import io
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES_LUNG = ['Bacterial Pneumonia', 'Corona Virus Disease','Normal','Tuberculosis','Viral Pneumonia']
CLASS_NAMES_BRAIN = ['glioma','meningioma','notumor','pituitary']


class ImprovedModel(nn.Module):
    def __init__(self, num_classes):
        super(ImprovedModel, self).__init__()
        self.backbone = models.resnet50(pretrained=False)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

def load_model_lung(model_path="app/model/lung_xray_model.pth"):
    model = ImprovedModel(num_classes=len(CLASS_NAMES_LUNG))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def load_model_brain(model_path="app/model/brain_xray_model.pth"):
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES_BRAIN))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def preprocess_image_lung(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)
    return image.to(device)


def preprocess_image_brain(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    # image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    return image.to(device)





def predict_lung_from_bytes(image_bytes:bytes,model):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_tensor = preprocess_image_lung(image)
        with torch.no_grad():
            outputs = model(image_tensor)
            predicted = torch.argmax(outputs,dim=1)
            return CLASS_NAMES_LUNG[predicted.item()]

    except Exception as e:
        return f"Hata oluştu: {str(e)}"

def predict_brain_from_bytes(image_bytes:bytes,model):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_tensor = preprocess_image_brain(image)
        with torch.no_grad():
            outputs = model(image_tensor)
            predicted = torch.argmax(outputs,dim=1)
            return CLASS_NAMES_BRAIN[predicted.item()]

    except Exception as e:
        return f"Hata oluştu: {str(e)}"
