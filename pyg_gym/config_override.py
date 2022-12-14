import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

config_path = "conf"
config_name="config"
@hydra.main(version_base=None, config_path=config_path, config_name=config_name)
def run(cfg: DictConfig):
    return cfg

def class_out_channels(cfg:DictConfig) -> DictConfig:
    # The last output must be size of classes, but with a sweep we don't know which layer will be last
    if cfg['model']['global_mlp']['num_layers'] == 0 and cfg['model']['pool']['name'] in ['diffpool']:
        cfg['model']['pool']['out_channels'] = cfg['data']['num_classes']
    if cfg['model']['global_mlp']['num_layers'] == 0 and cfg['model']['pool']['name'] not in ['diffpool']:
        cfg['model']['post_mlp']['out_channels'] = cfg['data']['num_classes']
    if cfg['model']['global_mlp']['num_layers'] == 0 and cfg['model']['post_mlp']['num_layers'] == 0:
        cfg['model']['mp']['out_channels'] = cfg['data']['num_classes']
    if cfg['model']['global_mlp']['num_layers'] == 0 and cfg['model']['post_mlp']['num_layers'] == 0 and cfg['model']['mp']['num_layers'] == 0:
        cfg['model']['pre_mlp']['out_channels'] = cfg['data']['num_classes']
    return cfg

def main():
    cfg = run()
    cfg_test = cfg.copy()
    cfg_class_out_channels = class_out_channels(cdf_test)
    print(cfg_class_out_channels)

if __name__ == "__main__":
    main()