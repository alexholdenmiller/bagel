
```
# for config and plotting
pip install --upgrade hydra-core hydra_colorlog plotext
python main.py

# for logging
pip install wandb
wandb login
python main.py wandb_log=True
```