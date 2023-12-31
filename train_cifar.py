import logging
import torch
import wandb

import plotext as plt
import torch.nn as nn

from dadaptation import DAdaptAdam
from omegaconf import DictConfig
from prodigyopt import Prodigy
from time import time
from torchvision import datasets, transforms
from tqdm import tqdm

from models import VisionTransformer, VitWithConvs, VitWithVQ

log = logging.getLogger(__name__)


def set_device(cuda : bool):
    device = "cpu"
    if cuda:
        if torch.cuda.is_available():
            device = "cuda"
            log.info("found gpu, using it")
        else:
            log.info("no gpu found, defaulting to cpu")
    return torch.device(device)

def get_data(flags):
    # Load the CIFAR-10 dataset


    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.Resize(flags.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(flags.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR10(root=flags.datadir, train=True, download=True, transform=transform_train)
    validset = datasets.CIFAR10(root=flags.datadir, train=False, download=True, transform=transform_test)
    return trainset, validset

def main(flags : DictConfig):
    device = set_device(flags.cuda)
    torch.manual_seed(flags.random_seed)

    trainset, validset = get_data(flags)

    # constants
    patch_size = 4
    in_channels = 3
    num_classes = 10

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=flags.batch_size, shuffle=True, num_workers=0, pin_memory=(device == "cuda"))
    validloader = torch.utils.data.DataLoader(validset, batch_size=flags.batch_size, shuffle=False, num_workers=0, pin_memory=(device == "cuda"))

    if flags.model == "vit":
        model = VisionTransformer(flags, in_channels, num_classes)
    elif flags.model == "convt":
        model = VitWithConvs(flags, in_channels, num_classes)
    elif flags.model == "vqt":
        model = VitWithVQ(flags, in_channels, num_classes)
    else:
        raise RuntimeError(f"don't recognize model={flags.model}")
    
    total_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    log.info(f"model has {total_params} trainable parameters")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    if flags.optim == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(),
                                    lr=flags.lr,
                                    weight_decay=flags.weight_decay)
    elif flags.optim == "adam":
        optimizer = torch.optim.Adam(model.parameters(),
                                    lr=flags.lr,
                                    weight_decay=flags.weight_decay)
    elif flags.optim.startswith("dadam"):
        if flags.lr != 1.0:
            log.warning("lr != 1.0 for dadapt")
        optimizer = DAdaptAdam(model.parameters(),
                               lr=flags.lr,
                               weight_decay=flags.weight_decay,
                               decouple=flags.optim.endswith("w"))
    elif flags.optim.startswith("padam"):
        if flags.lr != 1.0:
            log.warning("lr != 1.0 for prodigy")
        optimizer = Prodigy(model.parameters(),
                            lr=flags.lr,
                            weight_decay=flags.weight_decay,
                            decouple=flags.optim.endswith("w"))
    else:
        raise RuntimeError(f"don't recognize optim={flags.optim}")
    
    scheduler = None
    if flags.scheduler == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=flags.lr, steps_per_epoch=len(trainloader), epochs=flags.num_epochs)
    elif flags.scheduler == "cos":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(trainloader) * flags.num_epochs)
    elif flags.scheduler != "none":
        raise RuntimeError(f"don't recognize scheduler={flags.scheduler}")

    best_val_acc = 0
    train_accs = [0.0]
    val_accs = [0.0]
    epochs_no_improve = 0
    max_patience = 5

    plt.plot_size(60, 10)

    prev_time = time()
    total_steps = 0
    for epoch in range(1, flags.num_epochs + 1):
        running_accuracy = 0.0
        running_loss = 0.0
        running_codeppl = 0.0
        model.train()
        for batch in tqdm(trainloader, desc=f"Training Epoch {epoch}: "):
            inputs, labels = batch
            total_steps += inputs.size(0)
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, logs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            acc = (outputs.argmax(dim=1) == labels).float().mean()
            running_accuracy += acc.item()
            running_loss += loss.item()

            if flags.model == "vqt":
                model.vq.set_num_updates(total_steps)
                running_codeppl += logs["code_perplexity"].item() / inputs.size(0)

        running_accuracy /= len(trainloader) / 100
        # running_accuracy *= 100
        running_loss /= len(trainloader)
        train_accs.append(running_accuracy)
        if running_codeppl > 0.0:
            running_codeppl /= len(trainloader)

        # Validate the model
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for data in validloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs, logs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_acc = 100 * correct / total
        val_accs.append(val_acc)

        print()
        plt.clear_data()
        plt.plot(train_accs, label="train")
        plt.plot(val_accs, label="valid")
        plt.show()

        curr_time = time()
        log_stats = {
            "Epoch": epoch,
            "Time Taken": round(curr_time - prev_time, 1),
            "Train Accuracy": round(running_accuracy, 1),
            "Training Loss": round(running_loss, 2),
            "Validation Accuracy": round(val_acc, 1),
        }
        if running_codeppl > 0.0:
            log_stats["code_ppl"] = round(running_codeppl, 2)
        log.info(log_stats)
        prev_time = curr_time

        if flags.wandb_log:
            wb_stats = {
                "train_acc": running_accuracy,
                "train_loss": running_loss,
                "valid_acc": val_acc,
                "runtime": curr_time - prev_time,
            }
            if running_codeppl > 0.0:
                wb_stats["code_ppl"] = running_codeppl / len(trainloader)
            wandb.log(wb_stats)

        if val_acc > best_val_acc:
            epochs_no_improve = 0
            best_val_acc = val_acc
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= max_patience:
            print('Early stopping!')
            break
        else:
            continue

    if flags.wandb_log:
        wandb.run.summary["best_valid_acc"] = best_val_acc
        wandb.run.summary["model_size"] = total_params

if __name__ == "__main__":
    print("use main.py to launch training - it sets the config params")