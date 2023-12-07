
```
# for config and plotting
pip install --upgrade hydra-core hydra_colorlog plotext
python main.py

# for logging
pip install wandb
wandb login
python main.py wandb_log=True
```

for set up on greene check out this:
https://github.com/alexholdenmiller/nyu_cluster/edit/main/README.md

```
# to run a single job on greene
# srun...
python main.py

# to run a sweep (don't srun first)
python main.py -m lr=0.1,0.01 batch_size=128,32
```
