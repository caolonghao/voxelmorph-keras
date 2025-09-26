#!/usr/bin/env python

"""
Atlas/Template Construction with VoxelMorph (PyTorch)

This script learns a deformable atlas from a collection of images.
The atlas is optimized jointly with the deformation fields that align each image to it.

If you use this code, please cite:

    Learning Conditional Deformable Templates with Convolutional Networks
    Adrian V. Dalca, Marianne Rakic, John Guttag, Mert R. Sabuncu
    NeurIPS 2019. eprint arXiv:1908.02738
"""

import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm
import neurite as ne

class AtlasModel(nn.Module):
    """
    Atlas construction model.

    Learns a template/atlas image and deformation fields that warp input images to the atlas.
    """

    def __init__(self, inshape, nb_features=None, int_steps=7, use_probs=False):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_features: Encoder/decoder features. Default: [16, 32, 32, 32]
            int_steps: Number of flow integration steps. Default: 7
            use_probs: Use probabilistic model. Default: False
        """
        super().__init__()

        ndims = len(inshape)
        assert ndims in [2, 3], 'ndims should be 2 or 3. found: %d' % ndims

        if nb_features is None:
            nb_features = [16, 32, 32, 32]

        self.inshape = inshape
        self.int_steps = int_steps
        self.use_probs = use_probs

        # Initialize atlas as a learnable parameter
        self.atlas = nn.Parameter(torch.zeros(1, 1, *inshape))
        nn.init.normal_(self.atlas, mean=0.5, std=0.1)

        # Build UNet for predicting deformation field
        self.unet = vxm.nn.Unet(
            inshape=inshape,
            infeats=2,  # concatenated atlas and image
            nb_features=nb_features
        )

        # Velocity field output
        conv_kwargs = {'kernel_size': 3, 'padding': 1}
        if use_probs:
            # Output mean and log variance
            self.flow = nn.Conv2d(nb_features[0], ndims * 2, **conv_kwargs) if ndims == 2 \
                else nn.Conv3d(nb_features[0], ndims * 2, **conv_kwargs)
        else:
            self.flow = nn.Conv2d(nb_features[0], ndims, **conv_kwargs) if ndims == 2 \
                else nn.Conv3d(nb_features[0], ndims, **conv_kwargs)

        # Initialize flow layer with small weights
        self.flow.weight.data.normal_(0, 1e-5)
        self.flow.bias.data.zero_()

        # Integration layer
        if int_steps > 0:
            self.integrate = vxm.nn.layers.VecInt(inshape, int_steps)

        # Spatial transformer
        self.transformer = vxm.nn.layers.SpatialTransformer(inshape)

    def forward(self, src):
        """
        Forward pass.

        Parameters:
            src: Source image [B, 1, *inshape]

        Returns:
            warped: Warped source image
            flow: Deformation field
            atlas: Current atlas
        """
        # Expand atlas to match batch size
        batch_size = src.shape[0]
        atlas_batch = self.atlas.expand(batch_size, -1, *[-1] * len(self.inshape))

        # Concatenate atlas and source
        x = torch.cat([atlas_batch, src], dim=1)

        # Predict velocity field
        x = self.unet(x)
        flow = self.flow(x)

        if self.use_probs:
            # Split into mean and log variance
            flow_mean = flow[:, :len(self.inshape)]
            flow_logvar = flow[:, len(self.inshape):]

            # Sample from distribution
            flow = flow_mean + torch.exp(0.5 * flow_logvar) * torch.randn_like(flow_mean)

        # Integrate to get deformation field
        if self.int_steps > 0:
            flow = self.integrate(flow)

        # Warp source with flow
        warped = self.transformer(src, flow)

        return warped, flow, atlas_batch


def train(model, train_generator, optimizer, losses, epochs, steps_per_epoch, model_dir):
    """
    Train the atlas model.
    """

    model.train()

    for epoch in range(epochs):
        epoch_loss = []
        epoch_start = time.time()

        for step in range(steps_per_epoch):
            # Get batch
            batch = next(train_generator)
            if isinstance(batch, (list, tuple)):
                inputs = batch[0]
            else:
                inputs = batch

            # Move to GPU if available
            if torch.cuda.is_available():
                inputs = inputs.cuda()

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            warped, flow, atlas = model(inputs)

            # Calculate losses
            total_loss = 0
            loss_dict = {}

            for loss_name, loss_func in losses.items():
                if loss_name == 'sim':
                    # Similarity loss between warped image and atlas
                    loss_val = loss_func(warped, atlas)
                elif loss_name == 'grad':
                    # Gradient/smoothness loss on flow field
                    loss_val = loss_func(flow, flow)  # vxm losses expect y_pred, y_true format
                else:
                    loss_val = loss_func(warped, atlas)

                total_loss += loss_val
                loss_dict[loss_name] = loss_val.item()

            # Backward pass
            total_loss.backward()
            optimizer.step()

            epoch_loss.append(total_loss.item())

            if step % 20 == 0:
                print(f'  Step {step}/{steps_per_epoch} - Loss: {total_loss.item():.4f}')

        # Print epoch summary
        mean_loss = np.mean(epoch_loss)
        epoch_time = time.time() - epoch_start
        print(f'Epoch {epoch + 1}/{epochs} - Loss: {mean_loss:.4f} - Time: {epoch_time:.1f}s')

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(model_dir, f'atlas_epoch_{epoch + 1:04d}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': mean_loss,
            }, checkpoint_path)
            print(f'  Saved checkpoint: {checkpoint_path}')

    # Save final model
    final_path = os.path.join(model_dir, 'atlas_final.pt')
    torch.save(model.state_dict(), final_path)
    print(f'Saved final model: {final_path}')

    # Save atlas as numpy file
    atlas_np = model.atlas.detach().cpu().numpy().squeeze()
    atlas_path = os.path.join(model_dir, 'atlas.npy')
    np.save(atlas_path, atlas_np)
    print(f'Saved atlas: {atlas_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data parameters
    parser.add_argument('--img-list', required=True, help='path to text file listing image files')
    parser.add_argument('--img-prefix', help='optional image file prefix')
    parser.add_argument('--img-suffix', help='optional image file suffix')

    # Model parameters
    parser.add_argument('--model-dir', default='models',
                        help='model output directory (default: models)')
    parser.add_argument('--int-steps', type=int, default=7,
                        help='number of integration steps (default: 7)')
    parser.add_argument('--use-probs', action='store_true',
                        help='use probabilistic model')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of training epochs (default: 500)')
    parser.add_argument('--steps-per-epoch', type=int, default=100,
                        help='steps per epoch (default: 100)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='batch size (default: 1)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate (default: 1e-4)')

    # Loss parameters
    parser.add_argument('--sim-loss', default='mse',
                        help='similarity loss: mse or ncc (default: mse)')
    parser.add_argument('--sim-weight', type=float, default=1.0,
                        help='weight for similarity loss (default: 1.0)')
    parser.add_argument('--grad-weight', type=float, default=0.01,
                        help='weight for gradient loss (default: 0.01)')

    # Other
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID (default: 0)')
    parser.add_argument('--multichannel', action='store_true',
                        help='use multichannel input')

    args = parser.parse_args()

    # Device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    # Load data
    train_files = vxm.py.utils.read_file_list(args.img_list,
                                               prefix=args.img_prefix,
                                               suffix=args.img_suffix)
    print(f'Found {len(train_files)} training images')

    # Get image shape from first file
    sample_img, sample_shape = vxm.py.utils.load_volfile(train_files[0],
                                                          ret_shape=True,
                                                          add_batch_axis=True,
                                                          add_feat_axis=not args.multichannel)
    inshape = sample_shape[2:]  # Remove batch and channel dims
    print(f'Image shape: {inshape}')

    # Create data generator
    add_feat_axis = not args.multichannel
    generator = vxm.py.generators.volgen(train_files,
                                          batch_size=args.batch_size,
                                          add_feat_axis=add_feat_axis)

    # Build model
    model = AtlasModel(
        inshape=inshape,
        int_steps=args.int_steps,
        use_probs=args.use_probs
    )

    if torch.cuda.is_available():
        model = model.cuda()

    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Setup losses
    losses = {}

    # Similarity loss
    if args.sim_loss == 'mse':
        losses['sim'] = vxm.nn.losses.MSE().loss * args.sim_weight
    elif args.sim_loss == 'ncc':
        losses['sim'] = vxm.nn.losses.NCC().loss * args.sim_weight
    else:
        raise ValueError(f'Unknown similarity loss: {args.sim_loss}')

    # Gradient loss for regularization
    losses['grad'] = vxm.nn.losses.Grad('l2').loss * args.grad_weight

    # Create model directory
    os.makedirs(args.model_dir, exist_ok=True)

    # Train
    print('Starting atlas training...')
    train(model, generator, optimizer, losses, args.epochs, args.steps_per_epoch, args.model_dir)