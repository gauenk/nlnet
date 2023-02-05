"""

Test a set of baseline_v2 Networks

"""

# -- sys --
import os

# -- data summary --
import numpy as np
import pandas as pd

# -- caching results --
import cache_io

# -- network configs --
from dev_basics.trte import test
from dev_basics.utils.misc import nice_pretrained_path

def main():

    # -- start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- records --
    approx_exps = cache_io.get_exps("exps/test_baseline_v2/first_grid.cfg")
    exact_exps = cache_io.get_exps("exps/test_baseline_v2/exact_grid.cfg")
    # exps = approx_exps + exact_exps
    exps = exact_exps
    def clear_fxn(num,cfg):
        return True
    results = cache_io.run_exps(exps,test.run,
                                name = ".cache_io/test_baseline_v2",
                                version = "v1",
                                clear=False,skip_loop=False,
                                clear_fxn=clear_fxn,
                                enable_dispatch="slurm")
    # -- summary --
    print("\n"*3)
    print("-"*5 + " Summary " + "-"*5)

    # gfields = ['ws','wt','k','search_menu_name','search_v0','search_v1']
    results['pretrained_path'] = nice_pretrained_path(results['pretrained_path'])
    gfields = ['ws','wt','k','search_v0','search_v1','pretrained_path']
    afields = ['psnrs','ssims','strred','timer_deno']
    agg_fxn = lambda x: np.mean(np.stack(x))
    summary = results.groupby(gfields).agg({k:agg_fxn for k in afields})
    summary = summary.reset_index()
    print(summary)
    print(summary[afields+gfields])

if __name__ == "__main__":
    main()
