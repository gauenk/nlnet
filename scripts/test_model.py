"""

Train Search-Refine Networks

"""

# -- sys --
import os

# -- caching results --
import cache_io

# -- network configs --
from dev_basics.trte import test

def main():

    # -- start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- records --
    cfg_file = "exps/test_model.cfg"
    exps = cache_io.get_exps(cfg_file)
    records = cache_io.run_exps(exps,test.run,
                                name = ".cache_io/test_model",
                                version = "v1",
                                clear=False,skip_loop=False,
                                enable_dispatch="slurm")

    # -- summary --
    print(list(records.columns))
    print("\n"*3)
    print("-"*5 + " Summary " + "-"*5)
    fields = ['ws','wt','rbwd','timer_deno','depth']
    print(records[fields])


if __name__ == "__main__":
    main()
