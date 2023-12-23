
## basics
```
# for config and plotting
pip installd -r requirements.txt
python main.py

# for logging
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

## reproing patch embedding, convolutions, and vector quantization

```
python main.py model=vit
python main.py model=convt
python main.py model=vqt
```

## for running the vqvae experiments
```
python vqvae.py # to train the vqvae model

python vit_vqvae.py # to train vit with vqvae encoding (specify modes: conv, patch, vqvae for comparisons)
```
