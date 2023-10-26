from torch import nn


def init(module, weight_init, bias_init, gain=1):
    if hasattr(module, 'weight'):
        weight_init(module.weight.data, gain=gain)
    if hasattr(module, 'bias'):
        bias_init(module.bias.data)
    return module

def get_init_linear(input_dim, output_dim):
    init_method = nn.init.orthogonal_
    def init_(m):
        return init(m, init_method, lambda x: nn.init.constant_(x, 0))
    return init_(nn.Linear(input_dim, output_dim))


class MLPNetwork(nn.Module):
    """
    Class implementing a generic Multi-Layer Perceptron network
    """
    def __init__(self, 
            input_dim, 
            out_dim, 
            hidden_dim=64, 
            n_hidden_layers=1, 
            activation_fn='relu',
            out_activation_fn=None,
            norm_in=True):
        """
        Inputs:
            :param input_dim (int): Dimension of the input
            :param out_dim (int): Dimension of the output
            :param hidden_dim (int): Dimension of the hidden layer
            :param n_hidden_layers (int): Number of hidden layers
            :param activation_fn (str): Activation function after each layer,
                must be in ['relu', 'tanh']
            :param out_activation_fn (str): Activation function of the output
                layer, must be in [None, 'tanh']
            :param norm_in (bool): Whether to perform BatchNorm on model input
        """
        super(MLPNetwork, self).__init__()
        self.n_hidden_layers = n_hidden_layers

        # Normalisation of inputs
        if norm_in:
            self.in_fn = nn.BatchNorm1d(input_dim)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x

        # Choice for activation function
        if activation_fn not in ['tanh', 'relu']:
            print("ERROR in MLPNetwork: bad activation_fn with", activation_fn)
            print("     must be in ['tanh', 'relu']")
            exit()
        activ_fn = {
            'tanh': nn.Tanh(),
            'relu': nn.ReLU()
        }[activation_fn]

        # Choice for activation function at the last layer
        if out_activation_fn not in [None, 'tanh']:
            print("ERROR in MLPNetwork: bad out_activation_fn with", 
                out_activation_fn)
            print("     must be in [None, 'tanh']")
            exit()
        self.out_activ_fn = {
            None: lambda x: x,
            'tanh': nn.Tanh()
        }[out_activation_fn]

        # Method for initialising weights
        init_method = nn.init.orthogonal_
        gain = nn.init.calculate_gain(activation_fn)
        def init_(m):
            return init(
                m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), 
            activ_fn,
            *[nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    activ_fn
                ) for _ in range(self.n_hidden_layers)],
            nn.Linear(hidden_dim, out_dim)
        )
        self.mlp.apply(init_)

    def forward(self, X):
        """
        Foward pass of the model
        Inputs:
            X (PyTorch Tensor): Batch of inputs, 
                dim=(batch_size, input_dim)
        Outputs:
            out (PyTorch Tensor): Batch of outputs,
                dim=(batch_size, output_dim)
        """
        out = self.mlp(self.in_fn(X))
        return self.out_activ_fn(out)
