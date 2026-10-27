import numpy as np
import torch

def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / (e.sum() + 1e-12)

# Normalize interaction scores for visualization
def scores_to_alpha(scores, alpha_min=0., temperature=0.3):
    scores = torch.as_tensor(scores).detach().cpu().float()
    scores = scores / scores.max()
    scores = torch.softmax(abs(scores)/temperature, dim=0) * scores.sign()
    vmax = np.max(np.abs(scores.numpy())) + 1e-8
    strengths = [abs(s) / vmax for s in scores.tolist()]
    intensities = [alpha_min + (1-alpha_min) * s for s in strengths]
    return intensities
