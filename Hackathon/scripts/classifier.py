import cv2
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
import os
from abc import ABC, abstractmethod

class IGroupClassifier(ABC):
    @abstractmethod
    def classify(self, img_bgr):
        pass

class GroupClassifierMobileNet(IGroupClassifier):
    def __init__(self, model_path=None, classes_path=None, device=None):
        self.device = device or (torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = model_path or os.path.join(base_dir, 'modelo_group_classifier.pth')
        classes_path = classes_path or os.path.join(base_dir, 'modelo_group_classifier_classes.txt')
        with open(classes_path, 'r', encoding='utf-8') as f:
            self.class_names = [line.strip() for line in f if line.strip()]
        self.transform = T.Compose([
            T.Resize((128, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.model = models.mobilenet_v2(pretrained=False)
        self.model.classifier[1] = nn.Linear(self.model.last_channel, len(self.class_names))
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()

    def classify(self, img_bgr):
        img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(tensor)
            pred = output.argmax(1).item()
        return self.class_names[pred]

class GroupClassifierResNet50(IGroupClassifier):
    def __init__(self, model_path=None, classes_path=None, device=None):
        self.device = device or (torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = model_path or os.path.join(base_dir, 'modelo_group_classifier_resnet50.pth')
        classes_path = classes_path or os.path.join(base_dir, 'modelo_group_classifier_classes_resnet50.txt')
        with open(classes_path, 'r', encoding='utf-8') as f:
            self.class_names = [line.strip() for line in f if line.strip()]
        self.transform = T.Compose([
            T.Resize((128, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.model = models.resnet50(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, len(self.class_names))
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()

    def classify(self, img_bgr):
        img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(tensor)
            pred = output.argmax(1).item()
        return self.class_names[pred]

# Exemplo de uso:
# classifier = GroupClassifierMobileNet()
# ou
# classifier = GroupClassifierResNet50()
# pred = classifier.classify(img_bgr)
