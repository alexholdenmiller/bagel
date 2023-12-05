import hydra
import logging
import os
import wandb

from omegaconf import DictConfig, OmegaConf
from train_cifar import main

log = logging.getLogger(__name__)

# use hydra for config / savings outputs
@hydra.main(config_path=".", config_name="config", version_base="1.1")
def setup(flags : DictConfig):
    if os.path.exists("config.yaml"):
        # this lets us requeue runs without worrying if we changed our local config since then
        logging.info("loading pre-existing configuration, we're continuing a previous run")
        new_flags = OmegaConf.load("config.yaml")
        cli_conf = OmegaConf.from_cli()
        # however, you can override parameters from the cli still
        # this is useful e.g. if you did total_epochs=N before and want to increase it
        flags = OmegaConf.merge(new_flags, cli_conf)

    # log config + save it to local directory
    log.info(OmegaConf.to_yaml(flags))
    OmegaConf.save(flags, "config.yaml")

    if flags.wandb_log:
        hyperparams = OmegaConf.to_object(flags)
        # intentionally remove unimportant flags from the log
        del hyperparams["wandb_log"]
        del hyperparams["wbentity"]
        del hyperparams["wbproject"]
        del hyperparams["group"]
        del hyperparams["log_freq"]
        del hyperparams["datadir"]
        wandb.init(project=flags.wbproject,
                   entity=flags.wbentity,
                   group=flags.group,
                   config=hyperparams,
                   dir=os.getcwd(),
                   resume="never")
    
    main(flags)
    
    if flags.wandb_log:
        wandb.finish()


if __name__ == "__main__":
    setup()