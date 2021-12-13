from src.GLOBAL_VARS import CATEGORIES
from src.integrated_gradients import *
from src.local_data_matrix import *
from src.lime import *
from src.utils import *
from VisualizationLibrary.visualization_lib import *
import PIL.Image

from src.utils import normalize_arr

IMAGE_OUTPUT_PATH = "outputs/images/"
PLOT_OUTPUT_PATH = "outputs/plots/"

torch.cuda.is_available = lambda: False


# Common method

def plot_range(x_range, prob_correct_label, image_name):
    """

    :param x_range:
    :param prob_correct_label: an array of probability of the correct label based on x
    :param image_name:
    :return:
    """
    max_id = np.argmax(np.array(prob_correct_label))
    max_x, max_probability = x_range[max_id], prob_correct_label[max_id]
    textstr = '\n'.join((
        r'$Max Prob=%.4f$' % (max_probability,),
        r'$Max Thres=%.4f$' % (max_x,)))
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)

    fig, ax = plt.subplots()
    ax.set_title(image_name)
    ax.plot(x_range, prob_correct_label)
    ax.set_xlabel('Lower Percentile Eliminated')
    ax.set_ylabel('Probability of Correct Label')
    ax.text(0.7, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    plt.show()
    plt.savefig(PLOT_OUTPUT_PATH + image_name + ".jpg")
    return max_x, max_probability


# Integrated Gradient and Local Data Matrix.

def plot_gradients_w_image(explanation, input_image, image, image_name, explanation_norm_type="none"):
    explanation = normalize_arr(explanation, explanation_norm_type=explanation_norm_type)

    # plot
    gradient_and_orig = pil_image(Visualize(
        *prepare_plots(explanation, input_image),
        polarity="both",
        clip_above_percentile=99,
        clip_below_percentile=0,
        overlay=True)).resize((v // 4 for v in image.size))

    gradient_only = pil_image(Visualize(
        *prepare_plots(explanation, input_image),
        polarity="both",
        clip_above_percentile=99,
        clip_below_percentile=0,
        overlay=False)).resize((v // 4 for v in image.size))

    gradient_and_orig.save(IMAGE_OUTPUT_PATH + image_name + "_origin.jpg")
    gradient_only.save(IMAGE_OUTPUT_PATH + image_name + "_gradient_only.jpg")


def evaluate_pixel_based_methods_range(explanation, input_image, image_name, model, categories,
                                       correct_label, explanation_norm_type, num_x, x_type):
    """

    :param explanation:
    :param input_image:
    :param image_name:
    :param model:
    :param categories:
    :param correct_label: true label of this image
    :param explanation_norm_type: {"none", "minmax", "scale", "std"}; when x_type is percentile, should be none
    :param num_x: number of points in x axis
    :param x_type: threshold type, can only take value in {"threshold", "percentile"}
    :return:
    """
    # takes in single image!
    explanation = normalize_arr(explanation, explanation_norm_type=explanation_norm_type)
    prob_correct_label = []
    x_range = np.arange(num_x) * 1 / num_x

    # for each threshold
    for i in range(x_range.shape[0]):
        input_image_w_gradient = input_image.clone()
        # save threshold
        if x_type == "threshold":
            x = x_range[i]
        # if one uses percentile, find the corresponding value for that percentile
        elif x_type == "percentile":
            x = torch.quantile(abs(explanation), x_range[i])
        else:
            raise ValueError
        input_image_w_gradient[abs(explanation) < x] = 0
        output = model(input_image_w_gradient.unsqueeze(0))
        probabilities = torch.nn.functional.softmax(output[-1], dim=0)
        prob = probabilities[categories.index(correct_label)].item()
        prob_correct_label.append(prob)

    # plot range
    output_name_tag = image_name + "_" + x_type
    max_percentile, max_probability = plot_range(x_range, prob_correct_label, output_name_tag)
    return max_percentile, max_probability


def evaluate_pixel_based_methods_features_range(explanation, input_image, image_name, model, categories,
                                                correct_label, explanation_norm_type, num_x, x_type):
    # takes in single image!
    explanation = normalize_arr(explanation, explanation_norm_type=explanation_norm_type)
    agg_explanation = None  # Todo
    prob_correct_label = []
    x_range = np.arange(num_x) * 1 / num_x

    for i in range(x_range.shape[0]):
        input_image_w_gradient = input_image.clone()
        thres = torch.quantile(abs(explanation), x_range[i])
        input_image_w_gradient[abs(explanation) < thres] = 0
        output = model(input_image_w_gradient.unsqueeze(0))
        probabilities = torch.nn.functional.softmax(output[-1], dim=0)
        prob = probabilities[categories.index(correct_label)].item()
        prob_correct_label.append(prob)

    # plot range
    output_name_tag = image_name + "_" + x_type
    max_x, max_probability = plot_range(x_range, prob_correct_label, output_name_tag)
    return max_x, max_probability


# LIME.

def plot_lime_given_num_features(lime_explanation, image_name, x):
    boundary = get_boundaries(lime_explanation[0], top_label=0, num_features=x)
    plt.imshow(boundary)
    plt.savefig(IMAGE_OUTPUT_PATH + image_name + "_" + str(x) + ".jpg")


def evaluate_lime(lime_explanation, input_image, model, categories, image_name, num_features_tuple_plot=(10, 30, 80)):
    input_image_w_gradient = input_image.clone()
    for i in range(len(input_image_w_gradient)):
        # lime_explanation[-1] is the mask
        input_image_w_gradient[i][lime_explanation[-1] != 1] = 0

    # plot; lime_explanation [0] is the lime explanation
    for x in num_features_tuple_plot:
        plot_lime_given_num_features(lime_explanation, image_name, x)

    # predict
    topk_pred = get_topk_pred(input_image_w_gradient, model, categories, k=5)
    return topk_pred


if __name__ == '__main__':
    # setting vars
    image_path = "./data/dog.jpg"
    image_name = "samoyed"
    label_name = "Samoyed"
    explanation_norm_type = "scale"
    zero_out_threshold = 0.01
    path = './outputs/results_' + image_name + "_" + \
           explanation_norm_type + "_" + "{:.5f}".format(zero_out_threshold) + '.txt'

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
    topk_pred_ig = plot_gradients_w_image(
        explanation=explanation_ig, input_image=input_image, image=image,
        image_name=image_name + "_ig" + "_" + explanation_norm_type + "_" + "{:.5f}".format(zero_out_threshold),
        model=MODEL,
        categories=CATEGORIES,
        threshold=zero_out_threshold,
        explanation_norm_type=explanation_norm_type,
    )
    print("\nIntegrated Gradient prediction")
    prettyprint_tuple(topk_pred_ig)
    output_predictions(topk_pred_ig, path, result_type="Integrated Gradient prediction", output_type="a")

    max_percentile, max_probability = evaluate_pixel_based_methods_range(explanation=explanation_ig,
                                                                         input_image=input_image,
                                                                         image_name=image_name + "_ig",
                                                                         model=MODEL,
                                                                         categories=CATEGORIES,
                                                                         correct_label=label_name,
                                                                         explanation_norm_type='std')

    print(f'Maximum Probability of {label_name}: {max_probability}')
    print(f'Maximum Threshold: {max_percentile}')

    # local data matrix
    explanation_ldm = get_explanation_ldm(MODEL, input_batch)
    topk_pred_ldm = plot_gradients_w_image(
        explanation=explanation_ldm, input_image=input_image, image=image,
        image_name=image_name + "_ldm" + "_" + explanation_norm_type + "_" + "{:.5f}".format(zero_out_threshold),
        model=MODEL,
        categories=CATEGORIES,
        threshold=zero_out_threshold,
        explanation_norm_type=explanation_norm_type,
    )
    print("\nLocal Data Matrix prediction")
    prettyprint_tuple(topk_pred_ldm)
    output_predictions(topk_pred_ldm, path, result_type="Local Data Matrix prediction", output_type="a")

    max_percentile, max_probability = evaluate_pixel_based_methods_range(explanation=explanation_ldm,
                                                                         input_image=input_image,
                                                                         image_name=image_name + "_ldm",
                                                                         model=MODEL,
                                                                         categories=CATEGORIES,
                                                                         correct_label=label_name,
                                                                         explanation_norm_type='std')

    print(f'Maximum Probability of {label_name}: {max_probability}')
    print(f'Maximum Threshold: {max_percentile}')

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
