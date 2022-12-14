import logging
import os
import os.path as osp
import shutil

import hydra
import torch
import yaml
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import wandb
from pyg_gym.config_override import class_out_channels
from pyg_gym.runner import Runner
from wandb import AlertLevel

log = logging.getLogger(__name__)


log.info(f"cwd: {os.getcwd()}")
global CONFIG_NUM
CONFIG_NUM = 1
config_path = "conf"
config_name="config"


@hydra.main(version_base=None, config_path=config_path, config_name=config_name)
def run_experiment(cfg: DictConfig):
    #try:
    # config
    cfg = class_out_channels(cfg)
    # convert config to wandb
    wandb_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    wandb.init(mode=cfg.wandb.mode, project=cfg.project, config=wandb_cfg)
    # save local version of config
    if wandb.config.wandb["sweep"]:
        wandb.save(LOCAL_SWEEP_CONFIG)
    # Initialize wandb from hydra config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"device: {device}")
    #TODO breaks on wandb['mode'] = offline
    log.info(
        f"wandb sweep count/config_num : {int(wandb.run.name.split('-')[-1])}/{CONFIG_NUM}"
    )
    runner = Runner(device=device, **wandb.config.init)
    # TODO set_optimizer and set_criterion should be initialized on creation
    runner.load_data(**wandb.config.data)
    runner.add_model(**wandb.config.model)
    runner.set_optimizer(**wandb.config.optim)
    runner.set_criterion(**wandb.config.criterion)
    wandb.log({**runner.trainable_parameters})
    log.info("training parameters set, starting train!")
    if cfg.wandb.model_watch:
        # logs a lot of data - params, and gradients
        wandb.watch(runner.model, log="all", log_freq=10)
    train_metrics, valid_metric, test_metrics = runner.run(**wandb.config.run)
    # logging model artifacts for reloading later.
    model_path = "./models"
    if not osp.exists(model_path):
        os.makedir(model_path)
    model_path = osp.join(model_path, "model.pth")
    torch.save(runner.model.state_dict(), model_path)
    wandb.log_artifact(model_path, name="model", type="model", aliases=wandb.run.id)
    os.remove(model_path)
    wandb.finish()
    #except Exception as e:
    #    log.warning(f"Error in run_experiment. Run Aborted!!!\n Error: {e}")
    #    log.error(e, exc_info=True)

def get_values(d):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from get_values(v)
        else:
            yield v

def get_config_num(sweep_config):
    config_num = 1
    for i in get_values(sweep_config["parameters"]):
        config_num *= len(i)
    return config_num


def run_sweep():
    sweep_file = "./pyg_gym//conf/sweep.yaml"
    with open(sweep_file, "r") as file:
        sweep_config = yaml.load(file, Loader=yaml.FullLoader)
    global CONFIG_NUM
    CONFIG_NUM = get_config_num(sweep_config)
    sweep_id = wandb.sweep(sweep_config)
    wandb_sweep_path = f"wandb/sweep-{sweep_id}"
    if not osp.exists(wandb_sweep_path):
        os.makedirs(wandb_sweep_path)
    global LOCAL_SWEEP_CONFIG
    LOCAL_SWEEP_CONFIG = osp.join(wandb_sweep_path,"sweep.yaml")
    shutil.copyfile(sweep_file, LOCAL_SWEEP_CONFIG)
    wandb.agent(sweep_id, function=run_experiment)
    wandb.finish()


def main():
    config_file = "./pyg_gym/conf/config.yaml"
    with open(config_file, "r") as f:
        sweep_config = yaml.load(f, Loader=yaml.FullLoader)
    if sweep_config["wandb"]["sweep"]:
        run_sweep()
    else:
        run_experiment()


if __name__ == "__main__":
    main()
