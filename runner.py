from src.GLOBAL_VARS import CATEGORIES
from src.integrated_gradients import *
from src.local_data_matrix import *
from src.lime import *
from src.utils import *
from VisualizationLibrary.visualization_lib import *
import PIL.Image

IMAGE_OUTPUT_PATH = "outputs/images/"


def normalize_explanations(explanation, explanation_norm_type):
    """

    :param explanation: np array
    :param explanation_norm_type: takes in std, minmax, or scale
    :return:
    """
    if explanation_norm_type == "std":
        explanation = (explanation - torch.mean(explanation)) / torch.std(explanation, unbiased=False)
    if explanation_norm_type == "minmax":
        maxval = torch.max(explanation)
        minval = torch.min(explanation)
        explanation_std = (explanation - minval) / (maxval - minval)
        explanation = explanation_std * (maxval - minval) + minval
    if explanation_norm_type == "scale":
        maxval = torch.max(explanation)
        explanation = explanation/maxval
    if explanation_norm_type == "none":
        explanation = explanation
    else:
        raise ValueError
    return explanation


def evaluate_pixel_based_methods(explanation, input_image, image, image_name, model, categories,
                                 threshold, explanation_norm_type):
    explanation = normalize_explanations(explanation, explanation_norm_type=explanation_norm_type)
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
    # setting vars
    image_path = "./data/fireboat.jpeg"
    image_name = "fireboat"
    label_name = "fireboat"
    explanation_norm_type = "minmax"
    zero_out_threshold = 0.01
    path = './outputs/results_' + image_name + "_" + \
           explanation_norm_type + "_" + "{:.2f}".format(zero_out_threshold) + '.txt'

    # load image
    image = PIL.Image.open(image_path)
    input_image = preprocess(image)
    input_batch = input_image.unsqueeze(0)

    # print original predictions
    topk_pred_orig = get_topk_pred(input_image, MODEL, CATEGORIES)
    print("Original prediction")
    prettyprint_tuple(topk_pred_orig)
    output_predictions(topk_pred_orig, path, result_type="Original_prediction", output_type="w")

    # integrated gradient
    step = 50
    explanation_ig = get_explanation_ig(MODEL, input_batch, CATEGORIES, label_name)
    topk_pred_ig = evaluate_pixel_based_methods(
        explanation=explanation_ig, input_image=input_image, image=image,
        image_name=image_name + "_ig" + "_" + explanation_norm_type + "_" + "{:.2f}".format(zero_out_threshold),
        model=MODEL,
        categories=CATEGORIES,
        threshold=zero_out_threshold,
        explanation_norm_type=explanation_norm_type,
    )
    print("\nIntegrated Gradient prediction")
    prettyprint_tuple(topk_pred_ig)
    output_predictions(topk_pred_ig, path, result_type="Integrated Gradient prediction", output_type="a")

    # local data matrix
    explanation_ldm = get_explanation_ldm(MODEL, input_batch)
    topk_pred_ldm = evaluate_pixel_based_methods(
        explanation=explanation_ldm, input_image=input_image, image=image,
        image_name=image_name + "_ldm" + "_" + explanation_norm_type + "_" + "{:.2f}".format(zero_out_threshold),
        model=MODEL,
        categories=CATEGORIES,
        threshold=zero_out_threshold,
        explanation_norm_type=explanation_norm_type,
    )
    print("\nLocal Data Matrix prediction")
    prettyprint_tuple(topk_pred_ldm)
    output_predictions(topk_pred_ldm, path, result_type="Local Data Matrix prediction", output_type="a")

    # lime
    features_to_plot = (10, 30, 80)
    lime_explanation = get_lime_explainer(image, top_label=0, num_features=5)
    topk_pred_lime = evaluate_lime(lime_explanation, input_image=input_image,
                                   image_name=image_name + "_lime_5", model=MODEL,
                                   categories=CATEGORIES,
                                   num_features_tuple_plot=features_to_plot)
    print("\nLIME prediction 5 features")
    prettyprint_tuple(topk_pred_lime)
    output_predictions(topk_pred_lime, path, result_type="LIME prediction 5 features", output_type="a")

    lime_explanation = get_lime_explainer(image, top_label=0, num_features=20)
    topk_pred_lime = evaluate_lime(lime_explanation, input_image=input_image,
                                   image_name=image_name + "_lime_20", model=MODEL,
                                   categories=CATEGORIES,
                                   num_features_tuple_plot=features_to_plot)
    print("\nLIME prediction 20 features")
    prettyprint_tuple(topk_pred_lime)
    output_predictions(topk_pred_lime, path, result_type="LIME prediction 20 features", output_type="a")
