import torchvision

MODEL = torchvision.models.resnet18(pretrained=True)
MODEL.eval()

with open("./data/imagenet_classes.txt", "r") as f:
    CATEGORIES = [s.strip() for s in f.readlines()]

