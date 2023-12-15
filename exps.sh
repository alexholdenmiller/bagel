
python main.py -m wandb_log=true group=sweep_dadapt model=vit optim=adamw lr=0.006,0.0006,0.00006 scheduler=onecycle,cos,none weight_decay=1,0.1,0.01 &
sleep 60

python main.py -m wandb_log=true group=sweep_dadapt model=vit optim=dadamw lr=10,4,1,0.25,0.1 scheduler=onecycle,cos,none weight_decay=1,0.1,0.01 &
sleep 60

python main.py -m wandb_log=true group=sweep_dadapt model=vit optim=padamw lr=10,4,1,0.25,0.1 scheduler=onecycle,cos,none weight_decay=1,0.1,0.01 &
