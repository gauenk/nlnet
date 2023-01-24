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
    fields = ['ws','wt','rbwd','timer_deno','depth','timer_bwd']
    print(records[fields])


"""

[_Refinement_ Search Patterns] x7

Full Search
One Search
First-Block Search (e.g. first one of each block size in [5,5,5])
Every Nth Search (e.g. search every N blocks, resetting at each block layer)
- N grid = {2,5}. but only two or three. look at the runtimes to see what is dramatic
Encoder-Only Search (e..g only the encorders search; can be combined with others)

[Archs] x3

Criteria for choosing:
1. Inspect the runtimes to see what "refinement schemes" yield significant differences.
2. Inspect the runtimes of the full-search models and ensure varied runtimes.
3. Fix upper-limit for training each network.
Pick the best _3_ for example.

[Search Types] x5

Exact
Layer Refinement
Approximate Space-Time
Approximate Space
Approximate Time

[Noise Sims] x2
- Poisson(5)
- Gaussian(30)


[The Mixing Problem] TLDR; no/limited mixing in first paper since its too expensive.

E-L-L-L-L this one makes sense
A-L-L-L-L this one _also_ makes
E-L-A-L-A this one actually also makes sense :/

This might be solved by inspecting the relatives speeds of
the approximate search methods...


...But then this also depends on their parametrs


XP


So maybe some parameters are also struck by this?

How many models can I train?... Can I just grow to 1000?
...

Two stages of training...
early training & late training

Early training picks the remaining models. We use one epoch?

1000 -> 500 @ epoch = 1

Inspect if losing out on good ones. If not, then...

500 -> 250 @ epoch = 5
250 -> 125 @ epoch = 10
125 -> 70 @ epoch = 20
70 -> 35 @ epoch = 30
35 until finished.

How do we check if we are losing out on good ones? Maybe simple
bincounting of types? For example, check we have some of each
approximate type lost at each step.

Instead of "refining" couldn't we use an approximate search? So instead of:

exact-refine-refine

we have:

exact-appox-approx

In other words, the refined search is a _type_ of approximation with the condition we must have the previous layer's output. This marries the whole sets of methods together, so we should do this..




"""

if __name__ == "__main__":
    main()
