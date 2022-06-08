import numpy as np
import torch
import gpytorch
import matplotlib.pyplot as plt
import copy
import botorch


class PhysicsExactGPModel(gpytorch.models.ExactGP, botorch.models.gpytorch.GPyTorchModel):
    num_outputs = 1

    def __init__(self, train_x, train_y, likelihood, precision):
        super(PhysicsExactGPModel, self).__init__(train_x, train_y, likelihood)
        L = torch.cholesky(precision)

        # register parameters
        L_param = torch.nn.parameter.Parameter(L, requires_grad= False)
        self.register_parameter('L', L_param)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel())

        # freeze lengthscale parameter to 1
        self.covar_module.base_kernel.raw_lengthscale.requires_grad = False

    def forward(self, x):
        mean = self.mean_module(x)

        # transform x to intermediate space
        inter_x = x @ self.L
        covar = self.covar_module(inter_x)

        return gpytorch.distributions.MultivariateNormal(mean, covar)

    def get_precision(self):
        return self.L.data @ self.L.data.T