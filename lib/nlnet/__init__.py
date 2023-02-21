
# -- code api --
from . import original
# from . import bidir
from . import lightning
from .original import extract_config
from .original import extract_config as extract_model_config

# -- hooks --
from . import hooks

# -- api for searching --
from . import search
from .search import init_search#,extract_search_config

# -- api for normalization --
from . import normz
from .normz import init_normz#,extract_normz_config

# -- api for aggregation --
from . import agg
from .agg import init_agg#,extract_agg_config

# -- [dev] api for train/test --
from . import train_model
from . import test_model

# -- model api --
from .utils import optional

def load_model(cfg):
    mtype = optional(cfg,'model_type','nlnet')
    if mtype in ["nlnet","original"]:
        return original.load_model(cfg)
    else:
        raise ValueError(f"Uknown model type [{mtype}]")
