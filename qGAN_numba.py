import numba
from numba import jit, prange
import torch
from torch.utils.data import Subset
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import time
import torchvision
import torchvision.transforms as transforms
from Generator import QuantumGenerator
from Discriminator import Discriminator

@jit(nopython=True, fastmath=True, parallel=True, cache=True)
def normalize_tensors_numba(tensor_data: np.ndarray) -> np.ndarray:
    """JIT-compiled tensor normalization - optimized for batch processing"""
    batch_size, channels, height, width = tensor_data.shape
    normalized = np.empty_like(tensor_data)

    for i in prange(batch_size):  # Parallel loop
        flat = tensor_data[i].flatten()
        _sum = 0.0
        # Compute sum
        for j in range(flat.shape[0]):
            _sum += flat[j]

        # Normalize
        if _sum > 1e-12:
            for j in range(channels):
                for k in range(height):
                    for l in range(width):
                        normalized[i, j, k, l] = tensor_data[i, j, k, l] / _sum
        else:
            for j in range(channels):
                for k in range(height):
                    for l in range(width):
                        normalized[i, j, k, l] = tensor_data[i, j, k, l]
    return normalized

@jit(nopython=True, fastmath=True, cache=True)
def compute_entropy_numba(gen_dist: np.ndarray, target: np.ndarray) -> float:
    """JIT-compiled entropy calculation"""
    batch_size, channels, height, width = gen_dist.shape
    total_elements = channels * height * width

    # Average over batch
    gen_flat = np.zeros(total_elements, dtype=np.float32)
    for i in range(batch_size):
        idx = 0
        for j in range(channels):
            for k in range(height):
                for l in range(width):
                    gen_flat[idx] += gen_dist[i, j, k, l] / batch_size
                    idx += 1

    target_flat = target.flatten()

    # Add epsilon and normalize
    gen_sum, target_sum = 0.0, 0.0
    for i in range(total_elements):
        gen_flat[i] += 1e-12
        target_flat[i] += 1e-12
        gen_sum += gen_flat[i]
        target_sum += target_flat[i]

    for i in range(total_elements):
        gen_flat[i] /= gen_sum
        target_flat[i] /= target_sum

    # KL divergence
    kl = 0.0
    for i in range(total_elements):
        if target_flat[i] > 1e-12 and gen_flat[i] > 1e-12:
            kl += target_flat[i] * np.log(target_flat[i] / gen_flat[i])
    return kl


class QuantumGANTrainer:
    def __init__(self, device, use_numba=True, use_amp=True):
        self.device = device
        self.use_numba = use_numba and numba is not None
        self.use_amp = use_amp and device.type == 'cuda'

        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')

        print(f"Optimizations: Numba={self.use_numba}, AMP={self.use_amp}")

    def create_generator_input(self, batch_size: int) -> torch.Tensor:
        """Create input for quantum generator"""
        return torch.empty(batch_size, 0, device=self.device)

    def normalize_batch(self, tensor: torch.Tensor) -> torch.Tensor:
        """Optimized batch normalization"""
        if self.use_numba and tensor.size(0) > 4:  # Use numba for larger batches
            # For large batches, CPU normalization + transfer can be faster
            tensor_np = tensor.cpu().numpy()
            normalized_np = normalize_tensors_numba(tensor_np)
            return torch.from_numpy(normalized_np).to(self.device)
        else:
            # For small batches, use PyTorch on GPU
            batch_size = tensor.size(0)
            tensor_flat = tensor.view(batch_size, -1)
            norms = torch.norm(tensor_flat, p=2, dim=1, keepdim=True)
            norms = torch.clamp(norms, min=1e-8)
            normalized_flat = tensor_flat / norms.unsqueeze(-1)
            return normalized_flat.view_as(tensor)


    def compute_entropy(self, gen_dist: torch.Tensor, target: torch.Tensor) -> float:
        """entropy computation"""
        if self.use_numba:
            gen_np = gen_dist.detach().cpu().numpy()
            target_np = target.cpu().numpy()
            return compute_entropy_numba(gen_np, target_np)
        else:
            # PyTorch fallback
            batch_size = gen_dist.size(0)
            gen_flat = gen_dist.view(batch_size, -1).mean(dim=0)
            target_flat = target.flatten()

            gen_flat = gen_flat + 1e-10
            target_flat = target_flat + 1e-10

            gen_flat = gen_flat / gen_flat.sum()
            target_flat = target_flat / target_flat.sum()

            kl = torch.sum(target_flat * torch.log(target_flat / gen_flat))
            return kl.item()

    def training_step(self, real_samples: torch.Tensor, generator, discriminator,
                              generator_optimizer, discriminator_optimizer, loss_function):
        """Complete optimized training step"""
        batch_size = real_samples.size(0)

        # Move to device if needed
        if real_samples.device != self.device:
            real_samples = real_samples.to(self.device)

        # Normalize real samples
        real_dist = self.normalize_batch(real_samples)

        # Create labels
        real_labels = torch.rand((batch_size, 1), dtype=torch.float, device=self.device)*0.2 + 0.8
        fake_labels = torch.rand((batch_size, 1), dtype=torch.float, device=self.device)*0.2

        # Train Discriminator
        discriminator_optimizer.zero_grad()

        # Generate fake samples
        generator_input = self.create_generator_input(batch_size)
        with torch.no_grad():  # No generator gradients for discriminator training
            gen_dist = generator(generator_input).reshape(batch_size, 1, 8, 8)

        # Combine real and fake samples
        all_samples = torch.cat((real_dist, gen_dist))
        all_labels = torch.cat((real_labels, fake_labels))
        all_labels = all_labels.clone()
        flip_idx = torch.rand(all_labels.size()) < 0.1
        all_labels[flip_idx] = 1.0 - all_labels[flip_idx]

        # Discriminator forward pass with AMP
        if self.use_amp:
            with torch.amp.autocast('cuda'):
                disc_output = discriminator(all_samples)
                discriminator_loss = loss_function(disc_output, all_labels)

            self.scaler.scale(discriminator_loss).backward()
            self.scaler.step(discriminator_optimizer)
            self.scaler.update()
        else:
            disc_output = discriminator(all_samples)
            discriminator_loss = loss_function(disc_output, all_labels)
            discriminator_loss.backward()
            discriminator_optimizer.step()

        # Train Generator
        generator_optimizer.zero_grad()

        # Generate new samples
        generator_input = self.create_generator_input(batch_size)
        gen_dist = generator(generator_input).reshape(batch_size, 1, 8, 8)

        # Generator forward pass with AMP
        if self.use_amp:
            with torch.amp.autocast('cuda'):
                disc_output_fake = discriminator(gen_dist)
                generator_loss = loss_function(disc_output_fake, real_labels)

            self.scaler.scale(generator_loss).backward()
            self.scaler.step(generator_optimizer)
            self.scaler.update()
        else:
            disc_output_fake = discriminator(gen_dist)
            generator_loss = loss_function(disc_output_fake, real_labels)
            generator_loss.backward()
            generator_optimizer.step()

        # Compute metrics
        current_entropy = 0.0
        for i in range(batch_size):
            current_entropy += self.compute_entropy(gen_dist[i:i+1], real_dist[i:i+1])
        current_entropy /= batch_size

        return {
            'generator_loss': generator_loss.item(),
            'discriminator_loss': discriminator_loss.item(),
            'entropy': current_entropy,
            'gen_dist': gen_dist.detach()
        }

# Complete training loop
def run_training(generator, discriminator, train_loader, n_epochs=200):
    """Run complete training with optimizations"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move models to device
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    # Optimizers
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.005)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001, weight_decay=0.0005)

    # Loss function
    loss_function = nn.BCEWithLogitsLoss()

    # Initialize trainer
    trainer = QuantumGANTrainer(device)

    # Training metrics
    metrics = {
        'generator_loss': [],
        'discriminator_loss': [],
        'entropy': [],
        'epoch_times': []
    }

    print("Starting training...")

    for epoch in range(n_epochs):
        epoch_start = time.time()
        epoch_g_loss, epoch_d_loss, epoch_entropy = 0.0, 0.0, 0.0
        num_batches = 0

        # Training loop with progress bar
        for batch_idx, (real_samples, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}")):

            # Perform training step
            step_metrics = trainer.training_step(
                real_samples, generator, discriminator,
                generator_optimizer, discriminator_optimizer, loss_function
            )

            # Accumulate metrics
            epoch_g_loss += step_metrics['generator_loss']
            epoch_d_loss += step_metrics['discriminator_loss']
            epoch_entropy += step_metrics['entropy']
            num_batches += 1

        # Compute epoch averages
        avg_g_loss = epoch_g_loss / num_batches
        avg_d_loss = epoch_d_loss / num_batches
        avg_entropy = epoch_entropy / num_batches
        epoch_time = time.time() - epoch_start

        # Store metrics
        metrics['generator_loss'].append(avg_g_loss)
        metrics['discriminator_loss'].append(avg_d_loss)
        metrics['entropy'].append(avg_entropy)
        metrics['epoch_times'].append(epoch_time)

        print(f"Epoch {epoch+1}/{n_epochs} | "
              f"G Loss: {avg_g_loss:.4f} | D Loss: {avg_d_loss:.4f} | "
              f"Entropy: {avg_entropy:.4f} | Time: {epoch_time:.2f}s")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'generator_optimizer_state_dict': generator_optimizer.state_dict(),
                'discriminator_optimizer_state_dict': discriminator_optimizer.state_dict(),
                'metrics': metrics,
            }
            torch.save(checkpoint, f'checkpoint_epoch_{epoch+1}.pth')
            print(f"Checkpoint saved at epoch {epoch+1}")

    print("Training completed!")
    return metrics, generator, discriminator


if __name__ == "__main__":
    import argparse
    # -----------------------------
    # Command-line arguments
    # -----------------------------
    parser = argparse.ArgumentParser(description="Quantum GAN on MNIST digit 0")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=500, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--seed", type=int, default=35, help="Random seed")
   
    args, unknown = parser.parse_known_args()


    generator = QuantumGenerator(num_qubits=6)
    discriminator = Discriminator()

    batch_size = args.batch_size

    transform = transforms.Compose([
    transforms.Resize((8,8)),
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,))
])
    train_set = torchvision.datasets.MNIST(root=".", train=True, download=True, transform=transform)
    target_label = 0
    label_indices = [i for i, t in enumerate(train_set.targets) if t == target_label]
    label_indices_reduced = label_indices[:1]
    train_set = Subset(train_set, indices=label_indices_reduced)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=0)

    n_epochs = args.epochs
    metrics, trained_generator, trained_discriminator = run_training(
        generator, discriminator, train_loader, n_epochs=n_epochs
    )

    # Plot results
    plt.rcParams['lines.linewidth'] = 0.75   # line thickness
    plt.rcParams['lines.linestyle'] = '--'   # default line style
    plt.rcParams['lines.marker'] = 'o'       # default marker
    plt.rcParams['lines.markersize'] = 3     # marker size
    plt.rcParams['font.size'] = 11           # bigger text
    plt.rcParams['font.weight'] = 'light'

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(metrics['generator_loss'], label='Generator', color = 'royalblue')
    plt.plot(metrics['discriminator_loss'], label='Discriminator', color='magenta')
    plt.title('Training Loss')
    plt.legend()
    plt.grid()

    plt.subplot(1, 3, 2)
    plt.plot(metrics['entropy'])
    plt.title('Relative Entropy')
    plt.grid()

    plt.subplot(1, 3, 3)
    plt.plot(metrics['epoch_times'])
    plt.title('Epoch Times (s)')
    plt.grid()

    plt.tight_layout()
    plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Generate a sample
    with torch.no_grad():
        generator.eval()  # Set to evaluation mode

        # Create input for one sample
        trainer = QuantumGANTrainer(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        generator_input = trainer.create_generator_input(batch_size=1)

        # Generate the sample
        generated_tensor = generator(generator_input)

        # Convert to numpy
        generated_sample = generated_tensor.cpu().numpy()[0]  # Take first sample

    print(f"Sample shape: {generated_sample.shape}")
    print(f"Sample min: {generated_sample.min():.3f}, max: {generated_sample.max():.3f}")
    print(f"Sample norm (should be ~1.0): {np.sum(generated_sample.reshape(1, -1)):.3f}")

    
    ax = plt.subplot(1, 1, 1)
    plt.imshow(generated_sample.reshape(8, 8))
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f'0_{n_epochs}_epochs.png', dpi=300, bbox_inches='tight')
    plt.close()