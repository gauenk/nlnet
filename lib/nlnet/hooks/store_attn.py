
# -- logic --
import importlib
import numpy as np
import torch as th
import torch.nn as nn

class StoreNonLocalActivations(nn.Module):

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

        # -- record info --
        self.vids0 = []
        self.vids1 = []
        self.dists = []
        self.inds = []
        self.nheads = []

        # -- attach to search modules --
        for name0, layer0 in self.model.named_children():
            if hasattr(layer0,"blocks"):
                for name1, layer1 in layer0.blocks.named_children():
                    self.apply_hook(layer1.attn.search)

    def apply_hook(self,layer):
        def hook(layer, inputs, output):
            q_vid,k_vid = inputs[0],inputs[1]
            dists,inds = output[0],output[1]
            self.vids0.append(q_vid)
            self.vids1.append(q_vid)
            self.dists.append(dists)
            self.inds.append(inds)
        layer.register_forward_hook(hook)

    def forward(self, x: th.Tensor, flows: dict) -> th.Tensor:
        return self.model(x,flows)

