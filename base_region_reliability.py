import torch

def log_integral(information, vector, mean, covariance):
    information = information.double()
    vector = vector.double()
    mean = mean.double()
    covariance = covariance.double()
    root = torch.linalg.cholesky(0.5 * (covariance + covariance.transpose(-1, -2)) + torch.eye(6, device=mean.device) * 1e-09)
    whitened = root.transpose(-1, -2) @ information @ root
    precision = torch.eye(6, device=mean.device) + 0.5 * (whitened + whitened.transpose(-1, -2))
    factor = torch.linalg.cholesky(precision)
    h = (root.transpose(-1, -2) @ (vector - (information @ mean[..., None]).squeeze(-1))[..., None]).squeeze(-1)
    solution = torch.cholesky_solve(h[..., None], factor).squeeze(-1)
    return (vector * mean).sum(-1) - 0.5 * torch.einsum('...d,...de,...e->...', mean, information, mean) + 0.5 * (h * solution).sum(-1) - factor.diagonal(dim1=-2, dim2=-1).log().sum(-1)

def shared_probability(calibration, information, vector, reference_mean, reference_covariance, base_mean, base_covariance):
    shared = log_integral(information, vector, reference_mean, reference_covariance)
    independent = log_integral(information, vector, base_mean, base_covariance)
    bayes_factor = shared - independent
    probability = torch.sigmoid(calibration.region_logit_prior.double() + bayes_factor).to(vector.dtype)
    return (probability, bayes_factor.to(vector.dtype))

def leave_one_out_references(information, vector, retired_information, retired_vector):
    count = vector.shape[-2]
    other = torch.ones(count, count, device=vector.device) - torch.eye(count, device=vector.device)
    return (retired_information[..., None, :, :] + torch.einsum('rs,...sde->...rde', other, information), retired_vector[..., None, :] + torch.einsum('rs,...sd->...rd', other, vector))

def closed_factors(calibration, mode, information, vector, original_probability, retired_information, retired_vector, base_mean, base_covariance):
    if mode == 'crossfit':
        reference_information, reference_vector = leave_one_out_references(information, vector, retired_information, retired_vector)
        mean, covariance = calibration.posterior(reference_information, reference_vector, base_mean[..., None, :], base_covariance[..., None, :, :])
        probability, _ = shared_probability(calibration, information, vector, mean, covariance, base_mean[..., None, :], base_covariance[..., None, :, :])
    elif mode == 'prequential':
        probability = original_probability
    else:
        raise ValueError('Unknown region evidence reliability mode')
    return (retired_information + (information * probability[..., None, None]).sum(-3), retired_vector + (vector * probability[..., None]).sum(-2), probability)
