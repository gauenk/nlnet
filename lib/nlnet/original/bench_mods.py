# -- misc --
import dnls
from copy import deepcopy as dcopy
from easydict import EasyDict as edict
from ..utils import optional

# -- torch --
import torch as th
from einops import rearrange

# -- clean code --
from dev_basics.utils import clean_code
__methods__ = [] # self is a DataStore
register_method = clean_code.register_method(__methods__)


@register_method
def update_times(self,timer):
    # print(timer.names)
    if not(self.use_timer): return
    for key in timer.names:
        if key in self.times.names:
            self.times[key].append(timer[key])
        else:
            self.times[key] = [timer[key]]

