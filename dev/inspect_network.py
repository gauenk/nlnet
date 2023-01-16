"""

Creating/Checking the network

"""

import srnet
from easydict import EasyDict as edict

def main():
    cfg = edict()
    model = srnet.load_model(cfg)
    print(model)

if __name__ == "__main__":
    main()
