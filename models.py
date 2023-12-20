import torch
from torch.nn import *


# Parameters-to-coefficients map for velocity

class PTCMapVelocity(Module):

    def __init__(self, K, space, time):
        super(PTCMapVelocity, self).__init__()

        self.linear1 = Linear(6, K)
        self.norm1 = LayerNorm(K)
        self.activation1 = GELU()
        self.linear2 = Linear(K, K)
        self.norm2 = LayerNorm(K)
        self.activation2 = GELU()
        self.linear3 = Linear(K, K)
        self.norm3 = LayerNorm(K)
        self.activation3 = GELU()
        self.linear4 = Linear(K, K)
        self.norm4 = LayerNorm(K)
        self.activation4 = GELU()
        self.linearf = Linear(K, space * time)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, Linear):
            init.kaiming_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0.01)

    def forward(self, x):
        x = self.linear1(x)
        x = self.norm1(x)
        x = self.activation1(x)
        x = self.linear2(x)
        x = self.norm2(x)
        x = self.activation2(x)
        x = self.linear3(x)
        x = self.norm3(x)
        x = self.activation3(x)
        x = self.linear4(x)
        x = self.norm4(x)
        x = self.activation4(x)
        x = self.linearf(x)
        return x


# Parameters-to-coefficients map for pressure

class PTCMapPressure(Module):

    def __init__(self, K, space, time):
        super(PTCMapPressure, self).__init__()

        self.linear1 = Linear(6, K)
        self.norm1 = LayerNorm(K)
        self.activation1 = GELU()
        self.linear2 = Linear(K, K)
        self.norm2 = LayerNorm(K)
        self.activation2 = GELU()
        self.linear3 = Linear(K, K)
        self.norm3 = LayerNorm(K)
        self.activation3 = GELU()
        self.linearf = Linear(K, space * time)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, Linear):
            init.kaiming_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0.01)

    def forward(self, x):
        x = self.linear1(x)
        x = self.norm1(x)
        x = self.activation1(x)
        x = self.linear2(x)
        x = self.norm2(x)
        x = self.activation2(x)
        x = self.linear3(x)
        x = self.norm3(x)
        x = self.activation3(x)
        x = self.linearf(x)
        return x