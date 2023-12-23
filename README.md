
## basics
```
# for config and plotting
pip installd -r requirements.txt
python main.py

# for logging
wandb login
python main.py wandb_log=True
```

For set up on greene check out this:
https://github.com/alexholdenmiller/nyu_cluster/edit/main/README.md

```
# to run a single job on greene
# srun...
python main.py

# to run a sweep (don't srun first)
python main.py -m lr=0.1,0.01 batch_size=128,32
```

## Reproducing patch embedding, convolutions, and vector quantization

```
python main.py model=vit
python main.py model=convt
python main.py model=vqt
```

## For reproducing the vq_vae encoding + vq + vit, patch vit  and conv + vq + vit experiments
```
python vqvae.py # to train the vqvae model

python vit_vqvae.py # to train vit with vqvae encoding (specify modes: conv, patch, vqvae for comparisons)
```

We provide 2 notebook demos: vit-cifar-10.ipynb and vqvae-.ipynb to deonstrate base vit and vqvae vit runs. 
