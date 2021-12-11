import torchvision
from src.integrated_gradients import *
from src.local_data_matrix import *
from src.lime import *
from src.utils import *
from VisualizationLibrary.visualization_lib import *
import PIL.Image


def evaluate_pixel_based_methods(explanation, input_image, image, model, categories, threshold=0.1):
    explanation = (explanation - torch.mean(explanation)) / torch.std(explanation, unbiased=False)
    input_image_w_gradient = input_image.clone()
    input_image_w_gradient[abs(explanation) < threshold] = 0

    pil_image(Visualize(
        *prepare_plots(explanation, input_image),
        polarity="both",
        clip_above_percentile=99,
        clip_below_percentile=0,
        overlay=True)).resize((v // 4 for v in image.size))

    pil_image(Visualize(
        *prepare_plots(explanation, input_image_w_gradient),
        polarity="both",
        clip_above_percentile=99,
        clip_below_percentile=0,
        overlay=True)).resize((v // 4 for v in image.size))

    pil_image(Visualize(
        *prepare_plots(explanation, input_image_w_gradient),
        polarity="both",
        clip_above_percentile=99,
        clip_below_percentile=0,
        overlay=False)).resize((v // 4 for v in image.size))

    topk_pred = get_topk_pred(input_image_w_gradient, model, categories, k=5)
    print(topk_pred)
    return topk_pred


def evaluate_lime(lime_explanation, input_image, model, categories, num_features=(10, 30, 80)):
    input_image_w_gradient = input_image.clone()
    for i in range(len(input_image_w_gradient)):
        # lime_explanation[-1] is the mask
        input_image_w_gradient[i][lime_explanation[-1] != 1] = 0

    # lime_explanation [0] is the lime explanation
    plot_boundaries(lime_explanation[0], top_label=0, num_features=num_features)

    topk_pred = get_topk_pred(input_image_w_gradient, model, categories, k=5)
    print(topk_pred)
    return topk_pred


if __name__ == '__main__':
    # files = ["./data/dog.jpg", "./data/fireboat.jpeg"]

    # load image
    image = PIL.Image.open("./data/dog.jpg")
    input_image = preprocess(image)
    input_batch = input_image.unsqueeze(0)
    label_name = "Samoyed"

    # load categories and label
    with open("./data/imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]

    # load model
    model = torchvision.models.resnet18(pretrained=True)
    model.eval()

    # integrated gradient
    step = 50
    zero_out_threshold = 0.1
    explanation_ig = get_explanation_ig(model, input_batch, categories, label_name)
    topk_pred_ig = evaluate_pixel_based_methods(explanation=explanation_ig, input_image=input_image, image=image,
                                                model=model, categories=categories, threshold=zero_out_threshold)

    # local data matrix
    explanation_ldm = get_explanation_ldm(model, input_batch)
    topk_pred_ldm = evaluate_pixel_based_methods(explanation=explanation_ldm, input_image=input_image, image=image,
                                                 model=model, categories=categories, threshold=zero_out_threshold)

    # lime
    num_features = (10, 30, 80)
    lime_explanation = get_lime_explainer(image, top_label=0, num_features=num_features)
    topk_pred_lime = evaluate_lime(lime_explanation, input_image, model, categories)
