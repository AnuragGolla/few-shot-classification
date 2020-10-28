"""
Metrics and analysis methods for accuracies.
"""

def Loss(logits, labels, device):
    size = logits.size()
    sample_ct = size[0]
    n_samples = torch.tensor([sample_ct], dtype=torch.float, device=device, requires_grad=False)
    log_py = torch.empty(size=(size[0], size[1]), dtype=torch.float, device=device)
    for sample in range(sample_ct):
        log_py[sample] = -F.cross_entropy(logits[sample], labels, reduction='none')
    score = torch.logsumexp(log_py, dim=0) - torch.log(n_samples)
    return -torch.sum(score, dim=0)

def Accuracy(logits, labels):
    avg_preds = torch.logsumexp(logits, dim=0)
    return torch.mean(torch.eq(labels, torch.argmax(avg_preds, dim=-1)).float())


