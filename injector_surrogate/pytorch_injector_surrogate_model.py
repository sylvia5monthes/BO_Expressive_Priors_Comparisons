import torch
import torch.nn as nn

from tensorflow.keras.layers import InputLayer, Dense, Dropout

class Surrogate_NN_PyTorch(nn.Module):
    def __init__(self, model=None):
        """
        creates Pytorch version of specified model 
        :param model: the model to be recreated 
        """
        super(Surrogate_NN_PyTorch, self).__init__()

        self.modules = []

        if model != None: 
            for layer in model.model_1.layers:  
                config = layer.get_config()
                weights = layer.get_weights()

                if (isinstance(layer,Dense)):
                    w = torch.tensor(weights[0])
                    b = torch.tensor(weights[1])

                    linear = nn.Linear(w.shape[0], 
                                        w.shape[1], 
                                        bias=config["use_bias"])

                    with torch.no_grad():
                        linear.weight = torch.nn.Parameter(w.transpose(0,1))
                        linear.bias = torch.nn.Parameter(b)

                    self.modules.append(linear) 

                    # activation function
                    if (config["activation"]=='elu'):
                        self.modules.append(nn.ELU())

                elif (isinstance(layer,Dropout)):
                    self.modules.append(nn.Dropout(p=config["rate"]))

                elif (isinstance(layer,InputLayer) == False):
                    print("error")
        
        self.network = nn.Sequential(*self.modules)
    
    def forward(self, x):
        y = self.network(x)
        return y


class MinMaxScaler_PyTorch(object):
    def __init__(self, transformer):
        """
        creates Pytorch version of specified MinMaxScaler: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html 
        :param transformer: the MinMaxScaler object to be recreated 
        """
        super(MinMaxScaler_PyTorch, self).__init__()
        self.scale = torch.tensor(transformer.scale_)
        self.min = torch.tensor(transformer.min_)

    def transform(self, x): 
        x *= self.scale
        x += self.min
        return x
    
    def inverse_transform(self, x):
        x -= self.min
        x /= self.scale
        return x
    
