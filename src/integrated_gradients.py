import torch
from torch import nn


def get_integrated_gradients(model, input_batch, categories, label_name, steps):
    baseline = 0 * input_batch
    label = torch.LongTensor([categories.index(label_name)] * (steps + 1))

    if torch.cuda.is_available():
        baseline = baseline.to('cuda')
        input_batch = input_batch.to('cuda')

    # Scale input and compute gradients.
    scaled_inputs = [baseline + (float(i) / steps) * (input_batch - baseline) for i in range(0, steps + 1)]
    scaled_inputs = torch.cat(scaled_inputs)

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        scaled_inputs = scaled_inputs.to('cuda')
        label = label.to('cuda')
        model.to('cuda')

    # inference
    scaled_inputs.requires_grad = True
    output = model(scaled_inputs)

    # get gradient for input
    criterion = nn.CrossEntropyLoss()
    criterion(output, label).backward()
    grads = scaled_inputs.grad

    # Use trapezoidal rule to approximate the integral.
    # See Section 4 of the following paper for an accuracy comparison between
    # left, right, and trapezoidal IG approximations:
    # "Computing Linear Restrictions of Neural Networks", Matthew Sotoudeh, Aditya V. Thakur
    # https://arxiv.org/abs/1908.06214
    grads = (grads[:-1] + grads[1:]) / 2.0
    avg_grads = grads.mean(0)
    integrated_gradients = (input_batch - baseline) * avg_grads  # shape: <inp.shape>
    return integrated_gradients, output  # integrated_gradients[0] is the explanation


def get_explanation_ig(model, input_batch, categories, label_name, steps=50):
    integrated_gradients, output = get_integrated_gradients(model, input_batch, categories, label_name, steps)
    explanation_ig = integrated_gradients[0]
    return explanation_ig

