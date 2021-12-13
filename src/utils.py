import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

plt.rcParams["savefig.bbox"] = 'tight'
torch.manual_seed(1)

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def denormalize(img):
    return img.to('cpu') * torch.Tensor([0.229, 0.224, 0.225]).view((3, 1, 1)) + torch.tensor(
        [[0.485, 0.456, 0.406]]).view((3, 1, 1))


def prepare_plots(attr, img):
    img = denormalize(img).numpy().transpose((1, 2, 0)) * 255
    attr = attr.to('cpu').numpy().transpose((1, 2, 0))
    return attr, img


def show(imgs):
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = transforms.ToPILImage()(denormalize(img).to('cpu'))
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def load_img(url):
    filename = "./data/" + url.split("/")[-1]
    print(filename)
    if not os.path.exists(filename):
        import urllib
        try:
            urllib.URLopener().retrieve(url, filename)
        except:
            urllib.request.urlretrieve(url, filename)
    image = Image.open(filename)
    input_image = preprocess(image)
    return input_image


def get_topk_pred(input_image, model, categories, k=5):
    input_batch = input_image.unsqueeze(0)
    model.eval()
    logits = model(input_batch)
    probs = torch.nn.functional.softmax(logits, dim=1)
    probs_k = probs.topk(k)
    top_k = tuple((p, c, categories[c]) for p, c in zip(probs_k[0][0].detach().numpy(), probs_k[1][0].detach().numpy()))
    return top_k


def prettyprint_tuple(t):
    for x in t:
        print(x)


def output_predictions(t, path, result_type, output_type="w"):
    with open(path, output_type) as f:
        f.write(result_type)
        f.write('\n')
        for line in t:
            f.write(str(line))
            f.write('\n')
        f.write('\n')


def normalize_arr(arr, explanation_norm_type):
    """

    :param arr: tensor of any shape
    :param explanation_norm_type: takes in std, minmax, or scale
    :return:
    """
    if explanation_norm_type == "std":
        arr = (arr - torch.mean(arr)) / torch.std(arr, unbiased=False)
    elif explanation_norm_type == "minmax":
        maxval = torch.max(arr)
        minval = torch.min(arr)
        explanation_std = (arr - minval) / (maxval - minval)
        arr = explanation_std * (maxval - minval) + minval
    elif explanation_norm_type == "scale":
        maxval = torch.max(arr)
        arr = arr / maxval
    elif explanation_norm_type == "none":
        arr = arr
    else:
        raise ValueError
    return arr