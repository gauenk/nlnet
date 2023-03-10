"""

Creating/Checking the network

"""

import nlnet
import torch as th
from easydict import EasyDict as edict

def main():
    cfg = edict()
    model = srnet.load_model(cfg)
    model.inspect_print = True
    vid = th.randn(1,5,3,128,128).to("cuda:0")
    print(vid.shape)
    out = model(vid)
    print(out.shape)

if __name__ == "__main__":
    main()
