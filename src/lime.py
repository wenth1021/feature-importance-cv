import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.segmentation import mark_boundaries
from torch import nn
from torchvision import transforms
from lime import lime_image


def get_pil_transform():
    transf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224)
    ])

    return transf


def get_preprocess_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transf = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    return transf


pill_transf = get_pil_transform()
preprocess_transform = get_preprocess_transform()


def batch_predict(model, images):
    """

    :param model:
    :param images: image here is from Image.open(filename)
    :return:
    """
    model.eval()
    batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch = batch.to(device)

    logits = model(batch)
    probs = nn.functional.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()


def get_lime_explainer(image, top_label=0, num_features=5):
    """
    image: output of Image.open(filename)
    """
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        np.array(pill_transf(image)),
        batch_predict,  # classification function
        top_labels=5,
        hide_color=0,
        num_samples=1000)  # number of images that will be sent to classification function
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[top_label],
                                                positive_only=True, num_features=num_features, hide_rest=False)
    img_boundry1 = mark_boundaries(temp / 255.0, mask)
    plt.imshow(img_boundry1)
    plt.show()
    return explanation, temp, mask


def plot_boundaries(explanation, top_label=0, num_features=(10, 50, 100)):
    for x in num_features:
        temp, mask = explanation.get_image_and_mask(explanation.top_labels[top_label], positive_only=False,
                                                    num_features=x, hide_rest=False)
        img_boundry2 = mark_boundaries(temp / 255.0, mask)
        plt.imshow(img_boundry2)
        plt.show()
