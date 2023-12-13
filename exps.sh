
python main.py -m wandb_log=true group=sweep_code model=vqt vq_vars=320 vq_groups=1,2,4,8,16,32 vq_dim=128 &
python main.py -m wandb_log=true group=sweep_code model=vqt vq_vars=160 vq_groups=1,2,4,8,16,32 vq_dim=256 &
python main.py -m wandb_log=true group=sweep_code model=vqt vq_vars=80 vq_groups=1,2,4,8,16,32 vq_dim=512 &
python main.py -m wandb_log=true group=sweep_code model=vqt vq_vars=40 vq_groups=1,2,4,8,16,32 vq_dim=1024 &
