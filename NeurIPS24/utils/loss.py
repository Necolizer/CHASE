import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import linalg as LA

def get_loss_func(loss_func, loss_args):
    if loss_func == 'LabelSmoothingCrossEntropy':
        loss = LabelSmoothingCrossEntropy(smoothing=loss_args['smoothing'], temperature=loss_args['temperature'])
    elif loss_func == 'CrossEntropy':
        loss = nn.CrossEntropyLoss()
    elif loss_func == 'CE_MBMMD':
        loss = CE_MBMMD(
            CrossEntropy=nn.CrossEntropyLoss(),
            MMDLoss=MMDLoss(),
            weights=loss_args['weights'],
        )
    elif loss_func == 'LSCE_MBMMD':
        loss = LSCE_MBMMD(
            LabelSmoothingCrossEntropy=LabelSmoothingCrossEntropy(smoothing=loss_args['smoothing'], temperature=loss_args['temperature']),
            MMDLoss=MMDLoss(),
            weights=loss_args['weights'],
        )
    elif loss_func == 'InfoGCN_Loss':
        loss = InfoGCN_Loss(
            LabelSmoothingCrossEntropy=LabelSmoothingCrossEntropy(smoothing=loss_args['smoothing'], temperature=loss_args['temperature']),
            weights=loss_args['weights'],
            class_num=loss_args['class_num'],
            out_channels=loss_args['out_channels'],
            gain=loss_args['gain'],
        )
    elif loss_func == 'InfoGCN_Loss_MBMMD':
        loss = InfoGCN_Loss_MBMMD(
            LabelSmoothingCrossEntropy=LabelSmoothingCrossEntropy(smoothing=loss_args['smoothing'], temperature=loss_args['temperature']),
            MMDLoss=MMDLoss(),
            weights=loss_args['weights'],
            class_num=loss_args['class_num'],
            out_channels=loss_args['out_channels'],
            gain=loss_args['gain'],
        )
    elif loss_func == 'LSCE_GROUP':
        loss = LSCE_GROUP(LSCE=LabelSmoothingCrossEntropy(smoothing=loss_args['smoothing'], temperature=loss_args['temperature']))
    elif loss_func == 'LSCE_MBMMD_GROUP':
        loss = LSCE_MBMMD_GROUP(
            LSCE_GROUP=LSCE_GROUP(LSCE=LabelSmoothingCrossEntropy(smoothing=loss_args['smoothing'], temperature=loss_args['temperature'])),
            MMDLoss=MMDLoss(),
            weights=loss_args['weights'],
        )
    elif loss_func == 'InfoGCN_Loss_GROUP':
        loss = InfoGCN_Loss_GROUP(
            LabelSmoothingCrossEntropy=LabelSmoothingCrossEntropy(smoothing=loss_args['smoothing'], temperature=loss_args['temperature']),
            weights=loss_args['weights'],
            class_num=loss_args['class_num'],
            out_channels=loss_args['out_channels'],
            gain=loss_args['gain'],
        )
    elif loss_func == 'InfoGCN_Loss_MBMMD_GROUP':
        loss = InfoGCN_Loss_MBMMD_GROUP(
            LabelSmoothingCrossEntropy=LabelSmoothingCrossEntropy(smoothing=loss_args['smoothing'], temperature=loss_args['temperature']),
            MMDLoss=MMDLoss(),
            weights=loss_args['weights'],
            class_num=loss_args['class_num'],
            out_channels=loss_args['out_channels'],
            gain=loss_args['gain'],
        )
    else:
        print('Loss Not Included')
        loss = None
    
    return loss

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1, temperature=1.0):
        super().__init__()
        self.smoothing = smoothing
        self.temperature = temperature

    def forward(self, x, target):
        log_probs = F.log_softmax(x / self.temperature, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(dim=-1)).squeeze(dim=-1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class MMDLoss(nn.Module):
    '''
    Params:
    source: (n * len(x))
    target: (m * len(y))
    kernel_mul:
    kernel_num: 
    fix_sigma: 
    Return:
    loss: MMD loss
    '''
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, **kwargs):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss


class CE_MBMMD(nn.Module):
    def __init__(self, CrossEntropy, MMDLoss, weights=[1.0, 0.1]):
        super(CE_MBMMD, self).__init__()

        assert len(weights) == 2

        self.CE = CrossEntropy
        self.MMD = MMDLoss
        self.weights = weights

    def forward(self, x_tuple, target):

        N, M, C = x_tuple[-1].size()

        mpmmd = self.MMD(x_tuple[-2].view(N*M, C).contiguous(), x_tuple[-1].view(N*M, C).contiguous())

        return self.weights[0] * self.CE(x_tuple[0], target) + self.weights[1] * mpmmd


class LSCE_MBMMD(nn.Module):
    def __init__(self, LabelSmoothingCrossEntropy, MMDLoss, weights=[1.0, 0.1]):
        super(LSCE_MBMMD, self).__init__()

        assert len(weights) == 2

        self.LSCE = LabelSmoothingCrossEntropy
        self.MMD = MMDLoss
        self.weights = weights

    def forward(self, x_tuple, target):

        N, M, C = x_tuple[-1].size()

        mpmmd = self.MMD(x_tuple[-2].view(N*M, C).contiguous(), x_tuple[-1].view(N*M, C).contiguous())

        return self.weights[0] * self.LSCE(x_tuple[0], target) + self.weights[1] * mpmmd
        

def get_mmd_loss_infogcn(z, z_prior, y, num_cls):
    y_valid = [i_cls in y for i_cls in range(num_cls)]
    z_mean = torch.stack([z[y==i_cls].mean(dim=0) for i_cls in range(num_cls)], dim=0)
    l2_z_mean= LA.norm(z.mean(dim=0), ord=2)
    mmd_loss = F.mse_loss(z_mean[y_valid], z_prior[y_valid])
    return mmd_loss, l2_z_mean, z_mean[y_valid]

class InfoGCN_Loss(nn.Module):
    def __init__(self, LabelSmoothingCrossEntropy, weights=[1.0, 0.1, 0.0001], class_num=26, out_channels=256, gain=3):
        super(InfoGCN_Loss, self).__init__()

        assert len(weights) == 3

        self.LSCE = LabelSmoothingCrossEntropy
        self.weights = weights
        self.class_num = class_num
        self.z_prior = torch.empty(class_num, out_channels)
        nn.init.orthogonal_(self.z_prior, gain=gain)

    def forward(self, x_tuple, target):

        info_mmd_loss, l2_z_mean, z_mean = get_mmd_loss_infogcn(x_tuple[1], self.z_prior.to(x_tuple[1].device), target, self.class_num)

        return self.weights[0] * self.LSCE(x_tuple[0], target) + self.weights[1] * info_mmd_loss + self.weights[2] * l2_z_mean
    

class InfoGCN_Loss_MBMMD(nn.Module):
    def __init__(self, LabelSmoothingCrossEntropy, MMDLoss, weights=[1.0, 0.1, 0.0001, 0.1], class_num=26, out_channels=256, gain=3):
        super(InfoGCN_Loss_MBMMD, self).__init__()

        assert len(weights) == 4

        self.LSCE = LabelSmoothingCrossEntropy
        self.MMD = MMDLoss
        self.weights = weights
        self.class_num = class_num
        self.z_prior = torch.empty(class_num, out_channels)
        nn.init.orthogonal_(self.z_prior, gain=gain)

    def forward(self, x_tuple, target):

        info_mmd_loss, l2_z_mean, z_mean = get_mmd_loss_infogcn(x_tuple[1], self.z_prior.to(x_tuple[1].device), target, self.class_num)

        N, M, C = x_tuple[-1].size()

        mpmmd = self.MMD(x_tuple[-2].view(N*M, C).contiguous(), x_tuple[-1].view(N*M, C).contiguous())

        return self.weights[0] * self.LSCE(x_tuple[0], target) + self.weights[1] * info_mmd_loss + self.weights[2] * l2_z_mean + self.weights[-1] * mpmmd
    

class LSCE_GROUP(nn.Module):
    def __init__(self, LSCE):
        super().__init__()
        self.LSCE = LSCE

    def forward(self, x, target, target_person):
        N, M, C = x[1].size()
        return self.LSCE(x[0], target) + self.LSCE(x[1].view(N*M, C), target_person.view(N*M))
    
class LSCE_MBMMD_GROUP(nn.Module):
    def __init__(self, LSCE_GROUP, MMDLoss, weights=[1.0, 0.1]):
        super(LSCE_MBMMD_GROUP, self).__init__()

        assert len(weights) == 2

        self.LSCE_GROUP = LSCE_GROUP
        self.MMD = MMDLoss
        self.weights = weights

    def forward(self, x_tuple, target, target_person):

        N, M, C = x_tuple[-1].size()

        mpmmd = self.MMD(x_tuple[-2].view(N*M, C).contiguous(), x_tuple[-1].view(N*M, C).contiguous())

        return self.weights[0] * self.LSCE_GROUP((x_tuple[0], x_tuple[1]), target, target_person) + self.weights[1] * mpmmd
    

class InfoGCN_Loss_GROUP(nn.Module):
    def __init__(self, LabelSmoothingCrossEntropy, weights=[1.0, 0.1, 0.0001], class_num=26, out_channels=256, gain=3):
        super(InfoGCN_Loss_GROUP, self).__init__()

        assert len(weights) == 3

        self.LSCE = LabelSmoothingCrossEntropy
        self.weights = weights
        self.class_num = class_num
        self.z_prior = torch.empty(class_num, out_channels)
        nn.init.orthogonal_(self.z_prior, gain=gain)

    def forward(self, x_tuple, target, target_person):

        info_mmd_loss, l2_z_mean, z_mean = get_mmd_loss_infogcn(x_tuple[1], self.z_prior.to(x_tuple[1].device), target, self.class_num)
        N, M, C = x_tuple[2].size()

        return self.weights[0] * (self.LSCE(x_tuple[0], target) + self.LSCE(x_tuple[2].view(N*M, C), target_person.view(N*M))) + self.weights[1] * info_mmd_loss + self.weights[2] * l2_z_mean
    
    
class InfoGCN_Loss_MBMMD_GROUP(nn.Module):
    def __init__(self, LabelSmoothingCrossEntropy, MMDLoss, weights=[1.0, 0.1, 0.0001, 0.1], class_num=26, out_channels=256, gain=3):
        super(InfoGCN_Loss_MBMMD_GROUP, self).__init__()

        assert len(weights) == 4

        self.LSCE = LabelSmoothingCrossEntropy
        self.MMD = MMDLoss
        self.weights = weights
        self.class_num = class_num
        self.z_prior = torch.empty(class_num, out_channels)
        nn.init.orthogonal_(self.z_prior, gain=gain)

    def forward(self, x_tuple, target, target_person):

        info_mmd_loss, l2_z_mean, z_mean = get_mmd_loss_infogcn(x_tuple[1], self.z_prior.to(x_tuple[1].device), target, self.class_num)

        N, M, C = x_tuple[-1].size()
        mpmmd = self.MMD(x_tuple[-2].view(N*M, C).contiguous(), x_tuple[-1].view(N*M, C).contiguous())

        Np, Mp, Cp = x_tuple[2].size()

        return self.weights[0] * (self.LSCE(x_tuple[0], target) + self.LSCE(x_tuple[2].view(Np*Mp, Cp), target_person.view(Np*Mp)) ) + self.weights[1] * info_mmd_loss + self.weights[2] * l2_z_mean + self.weights[-1] * mpmmd