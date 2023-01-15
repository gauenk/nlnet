"""

Produces weights for Table 1

"""

# -- sys --
import os

# -- caching results --
import cache_io

# -- network configs --
from srnet import train_model

def main():

    # -- start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- records --
    cfg_file = "exps/train_model.cfg"
    exps = cache_io.get_exps(cfg_file)
    exps = [exps[2]]
    records = cache_io.run_exps(exps,train_model.run,
                                name = ".cache_io/train_model",
                                version = "v1",
                                clear=False,skip_loop=True,
                                enable_dispatch="slurm")

    # -- summary --
    print("\n"*3)
    print("-"*5 + " Summary " + "-"*5)
    fields = ['ws','wt','rbwd','train_time','init_test_psnr','final_test_psnr']
    print(records[fields])

    # -- details --
    # print("\n"*3)
    # print("-"*5 + " Details " + "-"*5)
    # fields = ['init_val_results_fn','val_results_fn','best_model_path']
    # print(records[fields].to_dict())

if __name__ == "__main__":
    main()
