
# -- code api --
from . import original
from . import bidir
from . import lightning
from .original import extract_model_config

# -- api for searching --
# from . import search
# from .search import get_search,extract_search_config

# -- [dev] api for train/test --
from . import train_model
from . import test_model

# -- model api --
from .utils import optional

def load_model(cfg):
    mtype = optional(cfg,'model_type','unl')
    if mtype == "srnet":
        return original.load_model(cfg)
    elif mtype == "bidir":
        return bidir.load_model(cfg)
    else:
        raise ValueError(f"Uknown model type [{mtype}]")
