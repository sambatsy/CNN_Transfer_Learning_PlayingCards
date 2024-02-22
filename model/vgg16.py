import torchvision.models as models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_ft = models.vgg16(pretrained=True)

in_features = model_ft.classifier[6].in_features
num_classes = 53

# adjusting the last fully connected layer
model_ft.classifier[6] = nn.Linear(in_features, num_classes)
