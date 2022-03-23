from functools import reduce
import torch


def get_eigenvectors_ldm(model, input_batch, categories, label_name):
    input_batch.requires_grad = True
    output = model(input_batch)
    prob = output.softmax(-1)
    log_prob = prob.log()
    grad_vecs = torch.zeros((log_prob.shape[-1], reduce(lambda x, y: x * y, input_batch.shape),))

    if torch.cuda.is_available():
        grad_vecs = grad_vecs.to("cuda")

    for i in range(log_prob.shape[-1]):
        index = torch.zeros_like(log_prob)
        index[..., i] = 1
        input_batch.grad.zero_()
        log_prob.backward(index, retain_graph=True)
        grad_vecs[i] += input_batch.grad.view(-1) * prob[..., i].sqrt()

    # Get rank-1 approximation 
    u, s, v = torch.svd_lowrank(grad_vecs.T, q=100, niter=5)  # u is the eigenvalue of the grad_vecs
    vp = u[:, 0].reshape(-1, 1)

    # Get gradient
    label = torch.LongTensor([categories.index(label_name)])

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        label = label.to('cuda')
        model.to('cuda')

    # inference
    input_batch.requires_grad = True
    input_batch.grad.zero_()
    output = model(input_batch)

    # get gradient for input
    criterion = torch.nn.CrossEntropyLoss()
    criterion(output, label).backward()
    grads = input_batch.grad.reshape(-1, 1)

    # get natural gradient
    natural_gradient = vp * (vp.T @ grads)
    return natural_gradient  # explanation is u[:, 0].detach().view(3, 224, 224)


def get_explanation_ldm(model, input_batch, categories, label_name):
    natural_gradient = get_eigenvectors_ldm(model, input_batch, categories, label_name)
    return natural_gradient.detach().view(3, 224, 224)
