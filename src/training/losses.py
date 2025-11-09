import torch, torch.nn.functional as F
def cb_focal_loss(logits, targets, gamma=2.0):
    ce = F.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce)
    return ((1-pt)**gamma * ce).mean()
