"""

Creating/Checking the network

"""

import srnet
import torch as th
from easydict import EasyDict as edict

def main():
    cfg = edict()
    model = srnet.load_model(cfg)
    vid = th.randn(1,5,3,128,128).to("cuda:0")
    print(vid.shape)
    out = model(vid)
    print(out.shape)

if __name__ == "__main__":
    main()
