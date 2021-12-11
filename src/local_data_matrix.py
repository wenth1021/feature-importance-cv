from functools import reduce
import torch


def get_eigenvectors_ldm(model, input_batch):
    output = model(input_batch)
    prob = output.softmax(-1)
    log_prob = prob.log()
    grad_vecs = torch.zeros((log_prob.shape[-1], reduce(lambda x, y: x * y, input_batch.shape),))

    if torch.cuda.is_available():
        grad_vecs = grad_vecs.to("cuda")

    for i in range(log_prob.shape[-1]):
        index = torch.zeros_like(log_prob)
        index[..., i] = 1
        log_prob.backward(index, retain_graph=True)
        grad_vecs[i] += input_batch.grad.view(-1) * prob[..., i].sqrt()

    u, s, v = torch.svd_lowrank(grad_vecs.T, q=100, niter=5)  # u is the eigenvalue of the grad_vecs
    return u  # explanation is u[:, 0].detach().view(3, 224, 224)


def get_explanation_ldm(model, input_batch):
    u = get_eigenvectors_ldm(model, input_batch)
    return u[:, 0].detach().view(3, 224, 224)
