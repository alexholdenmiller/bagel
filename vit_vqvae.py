import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import wandb

from abc import abstractmethod
from torch import nn
from torch.nn import functional as F
from torchvision import datasets, transforms
from tqdm import tqdm

from vqvae import *


class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim):
      # TODO

        super(PatchEmbedding, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.proj = nn.Conv2d(self.in_channels, 
                            self.embed_dim, 
                            kernel_size=self.patch_size, 
                            stride=self.patch_size
                           )
        self.norm = nn.LayerNorm(self.embed_dim)

    def forward(self, x):
        # TODO
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        # TODO
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(self.embed_dim, self.embed_dim * 3)
        self.qk_norm = False
        self.use_activation = False
        self.activation = nn.ReLU() if self.use_activation else nn.Identity()
        self.q_norm = nn.LayerNorm(self.head_dim) if self.qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if self.qk_norm else nn.Identity()
        self.attn_dropout = nn.Dropout(0.1)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.proj_dropout = nn.Dropout(0)

    def forward(self, x):
        # TODO
        batch_si, seq_len, emb_dim = x.shape
        qkv = self.qkv(x).reshape(batch_si, seq_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attention = q @ k.transpose(-2, -1)
        attention = attention.softmax(dim=-1)
        attention = self.attn_dropout(attention)

        z = attention @ v
        z = z.transpose(1, 2).reshape(batch_si, seq_len, emb_dim)
        z = self.proj(z)
#         z = self.proj_dropout(z)
        return z


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout):
        # TODO
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.attention_norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(nn.Linear(embed_dim, mlp_dim),
                                  nn.GELU(),
                                  nn.Dropout(dropout),
                                  nn.Linear(mlp_dim, embed_dim)
                                  # nn.Dropout(dropout)
        )
        self.mlp_norm = nn.LayerNorm(embed_dim)


    def forward(self, x):
        # TODO
        res = x
        x = self.attention_norm(x)
        x = self.attention(x)
        x = x + res # residual connection
        res = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = x + res
        return x


class VitWithVQVAE(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim, num_heads, mlp_dim, num_layers, num_classes, dropout, mode, vqvae_model, vq_dim=None):
        # TODO
        super(VitWithVQVAE, self).__init__()
        self.mode = mode
        if self.mode == 'patch':
            self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
            self.embed_len = self.patch_embed.num_patches + 1
        if self.mode == 'vqvae':
            self.conv_dilate = 1
            self.conv_kernel = 4
            self.conv_stride = 4
            self.vqvae = vqvae_model
            self.vq_dim = vq_dim
            self.project_vs = nn.Linear(vq_dim, embed_dim)
            self.embed_len = int(((image_size - self.conv_dilate * (self.conv_kernel - 1) - 1) / self.conv_stride + 1) ** 2) + 1  # max seqlen
        if self.mode == "conv":
            self.conv_dilate = 1
            self.conv_kernel = 5
            self.conv_stride = 3
            self.conv2d = nn.Conv2d(in_channels,
                                    embed_dim * 1,
                                    5,
                                    groups=1,
                                    stride=3,
                                    dilation=1)
            self.proj_feats = nn.Linear(embed_dim * 1, embed_dim)
            self.feat_norm = nn.LayerNorm(embed_dim)
            self.embed_len = int(((image_size - self.conv_dilate * (self.conv_kernel - 1) - 1) / self.conv_stride + 1) ** 2) + 1  # max seqlen

        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.embed_len, embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_dim, dropout) for i in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.cls_head = nn.Sequential(nn.Linear(embed_dim, embed_dim//2),
                                nn.GELU(),
                                nn.Dropout(dropout),
                                nn.Linear(embed_dim // 2, num_classes),
                                # nn.Dropout(dropout)     
                                )                           

    def forward(self, x):
        # TODO
        if self.mode == "patch":
            x = self.patch_embed(x)
#             print("patch_embed",x.shape)
        if self.mode == "vqvae":
            self.vqvae.eval()
            with torch.no_grad():
                z = self.vqvae.encode(x)[0]
                _, _, quantized_z, _ = vqvae_model.vq(x)
#                 print(quantized_z.shape)
                flattened_z = quantized_z.flatten(2).transpose(1, 2)
#                 print(flattened_z.shape)
                x = self.project_vs(flattened_z)
#                 x = self.dropout(x)
#                 print("proj vs shape", x.shape)

        if self.mode == "conv":
            x = self.conv2d(x)
#             print(x.shape)
            vqvae_model.eval()
            with torch.no_grad():
                _, _, quantized_z, _ = vqvae_model.vq(x)
#             print(quantized_z.shape)
            flattened_z = quantized_z.flatten(2).transpose(1, 2)
#             print(flattened_z.shape)
            x = self.proj_feats(flattened_z)
#             print(x.shape)
#             x = self.dropout(x)
#             x = self.feat_norm(x)

        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
#         print("concat cls tok", x.shape)
#         print(self.embed_len)
        x = x + self.pos_embed[:, -x.size(1):, :]
#         print("add pos embed", x.shape)
        # x = self.dropout(x)
        for block in self.transformer_blocks:
            x = block(x)
#             print("trans block", x.shape)
        # x = self.norm(x)
        logits = self.cls_head(x[:, 0])
        
#         print("logits shape", logits.shape)

        return logits


image_size = 32
patch_size = 4
in_channels = 3
embed_dim = 256 # 512
num_heads = 4
mlp_dim = 1024
num_layers = 6 # 4
num_classes = 10
dropout = 0.1
batch_size = 128 # 256
mode = "patch"

vq_dim = 64

# Load the CIFAR-10 dataset
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.Resize(image_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)


device = "cuda" if torch.cuda.is_available() else "cpu"


path = 'vqvae_models/vq_vae_model.pth'
vqvae_model = VQVAE(in_channels=3, num_embeddings=512, embedding_dim=64, use_ema=False).to(device)
vqvae_state_dict = torch.load(path)['state_dict']
vqvae_model.load_state_dict(vqvae_state_dict)
vqvae_total_params = sum(param.numel() for param in vqvae_model.parameters())
print(vqvae_total_params)

model = VitWithVQVAE(image_size, patch_size, in_channels, embed_dim, num_heads, mlp_dim, num_layers, num_classes, dropout, "vqvae", vqvae_model, vq_dim).to(device)
# input_tensor = torch.randn(1, in_channels, image_size, image_size).to(device)
# output = model(input_tensor)
# print(output.shape)

vit_total_params = sum(param.numel() for param in model.parameters())
print(vit_total_params)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
lr = 0.0006 # 0.003
weight_decay =  0 # 0.0001
num_epochs = 50 # 150
optimizer = torch.optim.Adam(model.parameters(),
                                lr=lr,
                                weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(trainloader), epochs=num_epochs)

cfg = {'epoch': num_epochs, 
           'lr':lr,
        'image_size': image_size,
        'patch_size': patch_size,
        'in_channels':in_channels,
        'embed_dim':embed_dim,
        'num_heads':num_heads,
        'mlp_dim': mlp_dim,
        'num_layers':num_layers,
        'num_classes':num_classes,
        'batch_size':batch_size,
        'mode':mode,
        'weight_decay':weight_decay,
        'optimizer':'adam',
        'scheduler':'onecyclelr'
      }

wandb.init(project="bagel", group='vitvq', config=cfg)

# Train the model
best_val_acc = 0
train_accs = []
test_accs = []
epochs_no_improve = 0
max_patience = 20
early_stop = False
pbar=tqdm(range(num_epochs))
for epoch in pbar:
    # if not load_pretrained:
    running_accuracy = 0.0
    running_loss = 0.0
    model.train()
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        running_accuracy += acc / len(trainloader)
        running_loss += loss.item() / len(trainloader)
    
    train_accs.append(running_accuracy)

    wandb.log({'train_acc':running_accuracy, 'train_loss':loss})

    # Validate the model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_acc = 100 * correct / total
    wandb.log({'val_acc':val_acc, 'val_loss':val_loss})
    pbar.set_postfix({"Epoch": epoch+1, "Train Accuracy": running_accuracy*100, "Training Loss": running_loss, "Validation Accuracy": val_acc})

    # Save the best model

    if val_acc > best_val_acc:
        epochs_no_improve = 0
        best_val_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer,
            'scheduler' : scheduler,
            'train_acc': train_accs,
            'test_acc': val_acc
        },  'best_model.pth')

    else:
        epochs_no_improve += 1

    if epoch > 100 and epochs_no_improve >= max_patience:
        print('Early stopping!')
        early_stop = True
        break
    else:
        continue


