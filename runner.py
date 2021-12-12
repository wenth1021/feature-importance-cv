import sys
from src.GLOBAL_VARS import CATEGORIES
from src.integrated_gradients import *
from src.local_data_matrix import *
from src.lime import *
from src.utils import *
from VisualizationLibrary.visualization_lib import *
import PIL.Image

IMAGE_OUTPUT_PATH = "outputs/images/"

torch.cuda.is_available = lambda: False;

def evaluate_pixel_based_methods(explanation, input_image, image, image_name, model, categories, threshold=0.1):
    explanation = (explanation - torch.mean(explanation)) / torch.std(explanation, unbiased=False)
    input_image_w_gradient = input_image.clone()
    input_image_w_gradient[abs(explanation) < threshold] = 0

    # plot
    gradient_and_orig = pil_image(Visualize(
        *prepare_plots(explanation, input_image),
        polarity="both",
        clip_above_percentile=99,
        clip_below_percentile=0,
        overlay=True)).resize((v // 4 for v in image.size))

    gradient_and_masked = pil_image(Visualize(
        *prepare_plots(explanation, input_image_w_gradient),
        polarity="both",
        clip_above_percentile=99,
        clip_below_percentile=0,
        overlay=True)).resize((v // 4 for v in image.size))

    gradient_only = pil_image(Visualize(
        *prepare_plots(explanation, input_image_w_gradient),
        polarity="both",
        clip_above_percentile=99,
        clip_below_percentile=0,
        overlay=False)).resize((v // 4 for v in image.size))

    gradient_and_orig.save(IMAGE_OUTPUT_PATH + image_name + "_origin.jpg")
    gradient_and_masked.save(IMAGE_OUTPUT_PATH + image_name + "_masked.jpg")
    gradient_only.save(IMAGE_OUTPUT_PATH + image_name + "_gradient_only.jpg")

    # predict
    topk_pred = get_topk_pred(input_image_w_gradient, model, categories, k=5)
    return topk_pred


def evaluate_lime(lime_explanation, input_image, model, categories, image_name, num_features_tuple_plot=(10, 30, 80)):
    input_image_w_gradient = input_image.clone()
    for i in range(len(input_image_w_gradient)):
        # lime_explanation[-1] is the mask
        input_image_w_gradient[i][lime_explanation[-1] != 1] = 0

    # plot; lime_explanation [0] is the lime explanation
    for x in num_features_tuple_plot:
        boundary = get_boundaries(lime_explanation[0], top_label=0, num_features=x)
        plt.imshow(boundary)
        plt.savefig(IMAGE_OUTPUT_PATH + image_name + "_" + str(x) + ".jpg")

    # predict
    topk_pred = get_topk_pred(input_image_w_gradient, model, categories, k=5)
    return topk_pred


if __name__ == '__main__':

    # load image
    image_path = "./data/dog.jpg"
    image = PIL.Image.open(image_path)
    input_image = preprocess(image)
    input_batch = input_image.unsqueeze(0)
    image_name = "samoyed"
    label_name = "Samoyed"

    path = './outputs/results_' + image_name + '.txt'
    sys.stdout = open(path, 'w')

    # print original predictions
    topk_pred_orig = get_topk_pred(input_image, MODEL, CATEGORIES)
    print("Original prediction")
    prettyprint_tuple(topk_pred_orig)

    # # integrated gradient
    # step = 50
    zero_out_threshold = 0.1
    # explanation_ig = get_explanation_ig(MODEL, input_batch, CATEGORIES, label_name)
    # topk_pred_ig = evaluate_pixel_based_methods(explanation=explanation_ig, input_image=input_image, image=image,
    #                                             image_name=image_name + "_ig", model=MODEL, categories=CATEGORIES,
    #                                             threshold=zero_out_threshold)
    # print("\nIntegrated Gradient prediction")
    # prettyprint_tuple(topk_pred_ig)

    # local data matrix
    explanation_ldm = get_explanation_ldm(MODEL, input_batch, CATEGORIES, label_name)
    topk_pred_ldm = evaluate_pixel_based_methods(explanation=explanation_ldm, input_image=input_image, image=image,
                                                 image_name=image_name + "_ldm", model=MODEL, categories=CATEGORIES,
                                                 threshold=zero_out_threshold)
    print("\nLocal Data Matrix prediction")
    prettyprint_tuple(topk_pred_ldm)

    # # lime
    # features_to_plot = (10, 30, 80)
    # lime_explanation = get_lime_explainer(image, top_label=0, num_features=5)
    # topk_pred_lime = evaluate_lime(lime_explanation, input_image=input_image,
    #                                image_name=image_name + "_lime_5", model=MODEL, categories=CATEGORIES,
    #                                num_features_tuple_plot=features_to_plot)
    # print("\nLIME prediction 5 features")
    # prettyprint_tuple(topk_pred_lime)

    # lime_explanation = get_lime_explainer(image, top_label=0, num_features=20)
    # topk_pred_lime = evaluate_lime(lime_explanation, input_image=input_image,
    #                                image_name=image_name + "_lime_20", model=MODEL, categories=CATEGORIES,
    #                                num_features_tuple_plot=features_to_plot)
    # print("\nLIME prediction 20 features")
    # prettyprint_tuple(topk_pred_lime)

    sys.stdout = sys.__stdout__
