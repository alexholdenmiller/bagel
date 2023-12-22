import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from abc import abstractmethod
from torch import nn
from torch.nn import functional as F
from typing import List, Callable, Union, Any, TypeVar, Tuple
from tqdm import tqdm

import wandb

# from torch import tensor as Tensor
Tensor = TypeVar('torch.tensor')

class BaseVAE(nn.Module):

    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size:int, current_device: int, **kwargs) -> Tensor:
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass

class VectorQuantizer(nn.Module):
    """
    Reference:
    [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 beta: float = 0.25,
                 use_ema = False,
                 decay=0.99,
                 epsilon=1e-5):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        
        self.use_ema = use_ema
        
        if self.use_ema:
            self.embedding.weight.data.normal_()
            self.register_buffer('_ema_cluster_size', torch.zeros(self.num_embeddings))
            self.ema_w = nn.Parameter(torch.Tensor(self.num_embeddings, self.embedding_dim))
            self.ema_w.data.normal_()

            self.decay = decay
            self.epsilon = epsilon
        else:
            self.embedding.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)
        

    def forward(self, latents: Tensor) -> Tensor:
        latents = latents.permute(0, 2, 3, 1).contiguous()  # [B x D x H x W] -> [B x H x W x D]
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.embedding_dim)  # [BHW x D]

        # Compute L2 distance between latents and embedding weights
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
               torch.sum(self.embedding.weight ** 2, dim=1) - \
               2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [BHW x K]

        # Get the encoding that has the min distance
        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [BHW, 1]

        # Convert to one-hot encodings
        device = latents.device
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.num_embeddings, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BHW x K]
        
        # Use EMA to update the embedding vectors
        if self.use_ema:
            self._ema_cluster_size = self._ema_cluster_size * self.decay + \
                (1 - self.decay) * torch.sum(encoding_one_hot, 0)
    
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self.epsilon)
                / (n + self.num_embeddings * self.epsilon) * n
            )
            
            dw = torch.matmul(encoding_one_hot.t(), flat_latents)
            self.ema_w = nn.Parameter(self.ema_w * self.decay + (1 - self.decay) * dw)
        
            self.embedding.weight = nn.Parameter(self.ema_w / self._ema_cluster_size.unsqueeze(1))
        
        # Quantize the latents
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [BHW, D]
        quantized_latents = quantized_latents.view(latents_shape)  # [B x H x W x D]

        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        
        if self.use_ema:
            vq_loss = commitment_loss * self.beta
        else:
            embedding_loss = F.mse_loss(quantized_latents, latents.detach())
            vq_loss = commitment_loss * self.beta + embedding_loss

        # Add the residue back to the latents
        quantized_latents = latents + (quantized_latents - latents).detach()

        avg_probs = torch.mean(encoding_one_hot, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return vq_loss, perplexity, quantized_latents.permute(0,3,1,2).contiguous(), encoding_one_hot

class ResidualLayer(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        super(ResidualLayer, self).__init__()
        self.resblock = nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                                kernel_size=3, padding=1, bias=False),
                                      nn.ReLU(True),
                                      nn.Conv2d(out_channels, out_channels,
                                                kernel_size=1, bias=False))

    def forward(self, input: Tensor) -> Tensor:
        return input + self.resblock(input)

class VQVAE(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 embedding_dim: int,
                 num_embeddings: int,
                 use_ema: True,
                 hidden_dims: List = None,
                 beta: float = 0.25,
                 img_size: int = 32,
                 **kwargs) -> None:
        super(VQVAE, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.img_size = img_size
        self.beta = beta

        modules = []
        if hidden_dims is None:
            hidden_dims = [128, 256]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels,
                          kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU())
        )

        for _ in range(6):
            modules.append(ResidualLayer(in_channels, in_channels))
        modules.append(nn.LeakyReLU())

        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, embedding_dim,
                          kernel_size=1, stride=1),
                nn.LeakyReLU())
        )

        self.encoder = nn.Sequential(*modules)

        self.vq_layer = VectorQuantizer(num_embeddings,
                                        embedding_dim,
                                        self.beta,
                                        use_ema)

        # Build Decoder
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(embedding_dim,
                          hidden_dims[-1],
                          kernel_size=3,
                          stride=1,
                          padding=1),
                nn.LeakyReLU())
        )

        for _ in range(6):
            modules.append(ResidualLayer(hidden_dims[-1], hidden_dims[-1]))

        modules.append(nn.LeakyReLU())

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=4,
                                       stride=2,
                                       padding=1),
                    nn.LeakyReLU())
            )

        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[-1],
                                   out_channels=3,
                                   kernel_size=4,
                                   stride=2, padding=1),
                nn.Tanh()))

        self.decoder = nn.Sequential(*modules)

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        return [result]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        result = self.decoder(z)
        return result
    
    def vq(self, encoding: List[Tensor]): 
        return self.vq_layer(encoding)

    def forward(self, inputs: Tensor, **kwargs) -> List[Tensor]:
        encoding = self.encode(inputs)[0]
        vqloss, perplexity, quantized_inputs, encoding_one_hot = self.vq(encoding)
        reconstructed = self.decode(quantized_inputs)
        return reconstructed, vqloss, perplexity

    def sample(self,
               num_samples: int,
               current_device: Union[int, str], **kwargs) -> Tensor:
        raise Warning('VQVAE sampler is not implemented.')

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]

# Function to unnormalize and display an image
def imshow(img):
    img = img / 2 + 0.5  # Unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

image_size = 32
batch_size = 64

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

# Download the training set
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)

# Create a DataLoader for the training set
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

# Download the test set
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

# Create a DataLoader for the test set
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)


# Define a transform to reverse the normalization
reverse_transform = transforms.Compose([
    transforms.Normalize((-0.5, -0.5, -0.5), (1.0, 1.0, 1.0)),
    transforms.ToPILImage()
])

# Get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# Show images
imshow(torchvision.utils.make_grid(images))


device = "cuda" if torch.cuda.is_available() else "cpu"

vqvae_cfg = {
    'epochs':25,
    'in_channels':3,
    'num_embed':512,
    'embed_dim':64,
    'lr':3e-4,
    'use_ema':True
}

wandb.init(project='bagel',group='vqvae',config=vqvae_cfg)

# Training loop

# Instantiate the VQ-VAE model
model = VQVAE(in_channels=3, num_embeddings=512, embedding_dim=64, use_ema=True).to(device)
model.train()

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=3e-4)

train_losses = []
recon_losses = []
vq_losses = []
ppls = []
LOSS = float('inf')

# Training loop
num_epochs = 25

for epoch in tqdm(range(num_epochs)):
    for data in trainloader:
        inputs, _ = data
        inputs = inputs.to(device)
        optimizer.zero_grad()

        # Forward pass
        reconstructed, vq_loss, _ = model(inputs)

        # Compute the loss
        recon_loss = criterion(reconstructed, inputs) 
        loss = recon_loss + vq_loss

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
    train_losses.append(loss.item())
    recon_losses.append(recon_loss.item())
    vq_losses.append(vq_loss.item())
    
    # Save the trained model
    if loss < LOSS:
        torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 'loss': loss}, 'vq_vae_model_ema.pth')
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')
    
    wandb.log({'vqvae_loss':loss.item(), 'epoch':epoch})
