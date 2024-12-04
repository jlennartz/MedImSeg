from torch import nn, Tensor

class LayerAdapter(nn.Module):
    """Layer Adapter to store activations for a given layer per batch.

    Attach this adapter via hooks (forward, preforward) to a layer in a 
    model by name of that layer.
    After each forward pass, the activations
    are stored in the attribute `activations` of this adapter.

    Args:
        swivel (str): Name of the layer in the model to attach to.
        device (str, optional): Device to store the input means on. Defaults to 'cuda:0'.
    """
    def __init__(
        self,
        swivel: str,
        device: str  = 'cuda:0'
    ):
        super().__init__()
        # init args
        self.swivel = swivel
        self.device = device
        self.to(device)


    ### private methods ###
        
    def _aggregate(
        self, 
        x: Tensor
    ) -> Tensor:
        
        return x
    

    ### public methods ###

    def forward(
        self, 
        x: Tensor
    ) -> Tensor:
        self.activations = self._aggregate(x).detach()

        return x
    