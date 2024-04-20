import torch
import torch.nn.functional as F

def d_loss(d_fake, d_real):
    real_loss = F.binary_cross_entropy(d_real, torch.ones_like(d_real))
    fake_loss = F.binary_cross_entropy(d_fake, torch.zeros_like(d_fake))
    return (real_loss + fake_loss) / 2

def g_loss(d_fake):
    return F.binary_cross_entropy(d_fake, torch.ones_like(d_fake))

def calculate_accuracy(y_pred, y_true):
    predicted = y_pred.round()
    correct = (predicted == y_true).float()
    accuracy = correct.sum() / len(correct)
    return accuracy

def d_acc_fake(d_fake):
    return calculate_accuracy(d_fake, torch.zeros_like(d_fake))

def d_acc_real(d_real):
    return calculate_accuracy(d_real, torch.ones_like(d_real))

def g_acc(d_fake):
    return calculate_accuracy(d_fake, torch.ones_like(d_fake))