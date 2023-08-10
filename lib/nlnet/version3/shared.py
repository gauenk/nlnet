import torch.nn as nn

def get_norm_layer(layer_s):
    if layer_s == "LayerNorm":
        return nn.LayerNorm
    else:
        raise ValueError("Uknown norm layer %s" % layer_s)
