"""

Compare metrics for different architectures and their settings

"""

# -- sys --
import os

# -- caching results --
import cache_io

# -- network configs --
# from srnet import train_model
from dev_basics.trte import test

def main():

    # -- start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- records --
    cfg_file = "exps/compare_arch.cfg"
    exps = cache_io.get_exps(cfg_file)
    records = cache_io.run_exps(exps,test.run,
                                name = ".cache_io/compare_arch",
                                version = "v1",
                                clear=False,skip_loop=False,
                                enable_dispatch="slurm")

    # -- summary --
    print("\n"*3)
    print("-"*5 + " Summary " + "-"*5)
    fields = ['timer_deno','depth','search_menu_name','search_v0','search_v1']
    print(records[fields])

    # -- details --
    # print("\n"*3)
    # print("-"*5 + " Details " + "-"*5)
    # fields = ['init_val_results_fn','val_results_fn','best_model_path']
    # print(records[fields].to_dict())

if __name__ == "__main__":
    main()
