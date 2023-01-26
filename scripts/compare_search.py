"""

Comparing the search

"""


# -- sys --
import os
import numpy as np

# -- experiment --
from nlnet.search import compare
# from nlnet import plots

# -- caching results --
import cache_io


def main():

    # -- start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- get/run experiments --
    def clear_fxn(num,cfg):
        # if cfg.search_name in ["nlat"]:
        #     return True
        # else:
        return False
    results = cache_io.run_exps("exps/compare_search.cfg",
                                compare.run,name=".cache_io/compare_search",
                                version="v1",skip_loop=False,
                                clear_fxn=clear_fxn,
                                clear=False,enable_dispatch="slurm",
                                records_fn=".cache_io_pkl/compare_search.pkl",
                                records_reload=True)

    # -- load results --
    gfields = ["search_name"]
    afields = ['time','flops','mem_res']
    agg_fxn = lambda x: np.mean(x)
    results = results.groupby(gfields).agg({k:agg_fxn for k in afields})
    results = results.reset_index()
    print(results[gfields + afields])

    #
    # -- plotting --
    #

    # -- labels --
    # label_names = {"swin":"SWIN","nat":"NAT","refine":"NLR",
    #                "csa":"CSA","nl":"NL"}
    # label_names = {"nl":"NL","nlp":"NLP"}
    # plots.compare_search.time_vs_radius(results,label_names)
    # plots.compare_search.time_vs_radius_no_refine(results,label_names)


if __name__ == "__main__":
    main()

