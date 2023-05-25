# -- misc --
import os,copy

# -- linalg --
import numpy as np
import torch as th
from einops import rearrange,repeat

# -- data mngmnt --
from pathlib import Path
from easydict import EasyDict as edict

# -- data --
import data_hub

# -- networks --
import nlnet
from . import lightning

# -- pytorch-lit --
import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import StochasticWeightAveraging
from pytorch_lightning.utilities.rank_zero import rank_zero_only

# -- dev basics --
# from dev_basics.report import deno_report
from functools import partial
from dev_basics.aug_test import test_x8
from dev_basics import flow
from dev_basics import net_chunks
from dev_basics.utils.misc import optional,slice_flows,set_seed
from dev_basics.utils.misc import rslice,write_pickle,read_pickle
from dev_basics.utils.gpu_mem import GpuMemer,MemIt
from dev_basics.utils.timer import ExpTimer,TimeIt
from dev_basics.utils.metrics import compute_psnrs,compute_ssims
from dev_basics.utils import vid_io


# -- extract train config --
# from functools import partial
from dev_basics.common import set_defaults as _set_defaults
# from dev_basics.common import optional as _optional
# from dev_basics.common import optional_fields,extract_config,extract_pairs
# _fields = []
# optional_full = partial(optional_fields,_fields)
# extract_train_config = partial(extract_config,_fields)

def set_defaults(cfg):
    defs = {"num_workers":8}
    _set_defaults(defs,cfg,overwrite=False)

def run(cfg):

    # -=-=-=-=-=-=-=-=-
    #
    #     Init Exp
    #
    # -=-=-=-=-=-=-=-=-

    # -- config --
    set_defaults(cfg)
    ModelLit,model_extract = get_lit(cfg)
    model_cfg = model_extract(cfg)

    # -- set-up --
    print("PID: ",os.getpid())
    set_seed(cfg.seed)
    root = Path(cfg.root)

    # -- create timer --
    timer = ExpTimer()

    # -- init log dir --
    log_dir = root / "output/log/" / str(cfg.uuid)
    print("Log Dir: ",log_dir)
    if not log_dir.exists():
        log_dir.mkdir(parents=True)
    log_subdirs = ["train"]
    for sub in log_subdirs:
        log_subdir = log_dir / sub
        if not log_subdir.exists(): log_subdir.mkdir()

    # -- prepare save directory for pickles --
    save_dir = root / "output/training/" / cfg.uuid
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    # -- network --
    # print(model_cfg)
    model = ModelLit(model_cfg,flow=cfg.flow,isize=cfg.isize,
                     batch_size=cfg.batch_size_tr,
                     lr_init=cfg.lr_init,lr_final=cfg.lr_final,
                     scheduler=cfg.scheduler,weight_decay=cfg.weight_decay,
                     nepochs=cfg.nepochs,task=cfg.task,
                     warmup_epochs=cfg.warmup_epochs,uuid=str(cfg.uuid),
                     sim_device=cfg.sim_device,sim_type=cfg.sim_type)

    # -- load dataset with testing mods isizes --
    # model.isize = None
    cfg_clone = copy.deepcopy(cfg)
    # cfg_clone.isize = None
    # cfg_clone.cropmode = "center"
    cfg_clone.nsamples_val = cfg.nsamples_at_testing
    data,loaders = data_hub.sets.load(cfg_clone)

    # -- init validation performance --
    init_val_report = MetricsCallback()
    logger = CSVLogger(log_dir,name="init_val_te",flush_logs_every_n_steps=1)
    trainer = pl.Trainer(accelerator="gpu",devices=1,precision=32,
                         limit_train_batches=1.,
                         max_epochs=3,log_every_n_steps=1,
                         callbacks=[init_val_report],logger=logger)
    timer.start("init_val_te")
    trainer.test(model, loaders.val)
    timer.stop("init_val_te")
    init_val_results = init_val_report.metrics
    print("--- Init Validation Results ---")
    print(init_val_results)
    init_val_res_fn = save_dir / "init_val.pkl"
    write_pickle(init_val_res_fn,init_val_results)
    print(timer)

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    #
    #          Training
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    # -- reset model --
    model.isize = cfg.isize

    # -- data --
    data,loaders = data_hub.sets.load(cfg)
    print(cfg.uuid)
    print("Num Training Vids: ",len(data.tr))
    print("Log Dir: ",log_dir)

    # -- pytorch_lightning training --
    logger = CSVLogger(log_dir,name="train",flush_logs_every_n_steps=1)
    ckpt_fn_val = cfg.uuid + "-{epoch:02d}-{val_loss:2.2e}"
    checkpoint_callback = ModelCheckpoint(monitor="val_loss",save_top_k=10,
                                          mode="min",dirpath=cfg.checkpoint_dir,
                                          filename=ckpt_fn_val)
    ckpt_fn_epoch = cfg.uuid + "-{epoch:02d}"
    cc_recent = ModelCheckpoint(monitor="epoch",save_top_k=10,mode="max",
                                dirpath=cfg.checkpoint_dir,filename=ckpt_fn_epoch)
    callbacks = [checkpoint_callback,cc_recent]
    if cfg.swa:
        swa_callback = StochasticWeightAveraging(swa_lrs=cfg.lr_init)
        callbacks += [swa_callback]
    trainer = pl.Trainer(accelerator="gpu",devices=cfg.ndevices,precision=32,
                         accumulate_grad_batches=cfg.accumulate_grad_batches,
                         limit_train_batches=cfg.limit_train_batches,
                         limit_val_batches=5,max_epochs=cfg.nepochs,
                         log_every_n_steps=1,logger=logger,
                         gradient_clip_val=cfg.gradient_clip_val,
                         callbacks=callbacks)
                         # strategy="ddp_find_unused_parameters_false")
    timer.start("train")

    # -- resume --
    ckpt_path = get_checkpoint(cfg.checkpoint_dir,cfg.uuid,cfg.nepochs)
    trainer.fit(model, loaders.tr, loaders.val, ckpt_path=ckpt_path)
    timer.stop("train")
    best_model_path = checkpoint_callback.best_model_path


    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    #
    #       Validation Testing
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    # -- reload dataset with no isizes --
    # model.isize = None
    # cfg_clone = copy.deepcopy(cfg)
    # cfg_clone.isize = None
    # cfg_clone.cropmode = "center"
    cfg_clone.nsamples_tr = cfg.nsamples_at_testing
    cfg_clone.nsamples_val = cfg.nsamples_at_testing
    data,loaders = data_hub.sets.load(cfg_clone)

    # -- training performance --
    tr_report = MetricsCallback()
    logger = CSVLogger(log_dir,name="train_te",flush_logs_every_n_steps=1)
    trainer = pl.Trainer(gpus=1,precision=32,limit_train_batches=1.,
                         max_epochs=1,log_every_n_steps=1,
                         callbacks=[tr_report],logger=logger)
    timer.start("train_te")
    trainer.test(model, loaders.tr)
    timer.stop("train_te")
    tr_results = tr_report.metrics
    tr_res_fn = save_dir / "train.pkl"
    write_pickle(tr_res_fn,tr_results)

    # -- validation performance --
    val_report = MetricsCallback()
    logger = CSVLogger(log_dir,name="val_te",flush_logs_every_n_steps=1)
    trainer = pl.Trainer(gpus=1,precision=32,limit_train_batches=1.,
                         max_epochs=1,log_every_n_steps=1,
                         callbacks=[val_report],logger=logger)
    timer.start("val_te")
    trainer.test(model, loaders.val)
    timer.stop("val_te")
    val_results = val_report.metrics
    print("--- Tuned Validation Results ---")
    print(val_results)
    val_res_fn = save_dir / "val.pkl"
    write_pickle(val_res_fn,val_results)

    # -- report --
    results = edict()
    results.best_model_path = [best_model_path]
    results.init_val_results_fn = [init_val_res_fn]
    results.train_results_fn = [tr_res_fn]
    results.val_results_fn = [val_res_fn]
    results.train_time = [timer["train"]]
    results.test_train_time = [timer["train_te"]]
    results.test_val_time = [timer["val_te"]]
    results.test_init_val_time = [timer["init_val_te"]]
    for f,val in init_val_results.items():
        results["init_"+f] = val
    for f,val in val_results.items():
        results["final_"+f] = val
    print(results)

    return results

def get_lit(cfg):
    if cfg.arch_name == "nlnet":
        lit = lightning.LitModel
        extract = nlnet.extract_model_config
    else:
        raise ValueError(f"Uknown arch name [{cfg.arch_name}]")
    return lit,extract

def get_checkpoint(checkpoint_dir,uuid,nepochs):
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return ""
    prev_ckpt = ""
    for epoch in range(nepochs):
        ckpt_fn = checkpoint_dir / ("%s-epoch=%02d.ckpt" % (uuid,epoch))
        if ckpt_fn.exists(): prev_ckpt = ckpt_fn
        else: break
    assert ((prev_ckpt == "") or prev_ckpt.exists())
    if prev_ckpt != "":
        print("Resuming training from {%s}" % (str(prev_ckpt)))
    return str(prev_ckpt)

class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = {}

    def _accumulate_results(self,each_me):
        for key,val in each_me.items():
            if not(key in self.metrics):
                self.metrics[key] = []
            if hasattr(val,"ndim"):
                ndim = val.ndim
                val = val.cpu().numpy().item()
            self.metrics[key].append(val)

    @rank_zero_only
    def log_metrics(self, metrics, step):
        # metrics is a dictionary of metric names and values
        # your code to record metrics goes here
        print("logging metrics: ",metrics,step)

    def on_train_epoch_end(self, trainer, pl_module):
        each_me = copy.deepcopy(trainer.callback_metrics)
        self._accumulate_results(each_me)

    def on_validation_epoch_end(self, trainer, pl_module):
        each_me = copy.deepcopy(trainer.callback_metrics)
        self._accumulate_results(each_me)

    def on_test_epoch_end(self, trainer, pl_module):
        each_me = copy.deepcopy(trainer.callback_metrics)
        self._accumulate_results(each_me)

    def on_train_batch_end(self, trainer, pl_module, outs,
                           batch, batch_idx, dl_idx):
        each_me = copy.deepcopy(trainer.callback_metrics)
        self._accumulate_results(each_me)


    def on_validation_batch_end(self, trainer, pl_module, outs,
                                batch, batch_idx, dl_idx):
        each_me = copy.deepcopy(trainer.callback_metrics)
        self._accumulate_results(each_me)

    def on_test_batch_end(self, trainer, pl_module, outs,
                          batch, batch_idx, dl_idx):
        self._accumulate_results(outs)
