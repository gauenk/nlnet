"""

Train a set of Baseline Networks

"""

# -- sys --
import os

# -- caching results --
import cache_io

# -- network configs --
from dev_basics.trte import train

def main():

    # -- start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- records --
    cfg_file = ["exps/train_baseline/first_grid.cfg"]
    cfg_file += ["exps/train_baseline/exact_grid.cfg"]
    exps = cache_io.get_exps(cfg_file)
    records = cache_io.run_exps(exps,train.run,
                                name = ".cache_io/train_baseline",
                                version = "v1",
                                clear=False,skip_loop=False,
                                enable_dispatch="slurm")
    # -- summary --
    print("\n"*3)
    print("-"*5 + " Summary " + "-"*5)
    fields = ['ws','train_time','init_test_psnr','final_test_psnr']
    fields += ['search_menu_name','search_v0','search_v1']
    print(records[fields])


if __name__ == "__main__":
    main()
