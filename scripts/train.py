#!/usr/bin/env python

"""Torch-backed training script for the VoxelMorph dense registration model."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Set Keras backend to PyTorch
os.environ['KERAS_BACKEND'] = 'torch'

import numpy as np
import voxelmorph as vxm

from keras import callbacks, optimizers

from . import _torch_utils as cli


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # data organisation parameters
    parser.add_argument('--img-list', required=True, help='line-separated list of training files')
    parser.add_argument('--img-prefix', help='optional input image file prefix')
    parser.add_argument('--img-suffix', help='optional input image file suffix')
    parser.add_argument('--atlas', help='optional atlas filename (enables scan-to-atlas training)')
    parser.add_argument('--model-dir', default='models', help='model output directory (default: models)')
    parser.add_argument('--multichannel', action='store_true', help='indicate that data has multiple channels')

    # training parameters
    parser.add_argument('--gpu', help='GPU ID numbers (default: use all available)')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size (default: 1)')
    parser.add_argument('--epochs', type=int, default=1500, help='number of training epochs (default: 1500)')
    parser.add_argument('--steps-per-epoch', type=int, default=100, help='steps per epoch (default: 100)')
    parser.add_argument('--load-weights', help='optional weights file to initialise with')
    parser.add_argument('--initial-epoch', type=int, default=0, help='initial epoch number (default: 0)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')

    # network architecture parameters
    parser.add_argument('--enc', type=int, nargs='+', help='list of unet encoder filters (default: 16 32 32 32)')
    parser.add_argument('--dec', type=int, nargs='+', help='list of unet decoder filters (default: 32 32 32 32 32 16 16)')
    parser.add_argument('--int-steps', type=int, default=7, help='number of integration steps (default: 7)')
    parser.add_argument('--int-downsize', type=int, default=2, help='flow downsample factor for integration (default: 2)')
    parser.add_argument('--use-probs', action='store_true', help='enable probabilistic velocity field')
    parser.add_argument('--bidir', action='store_true', help='enable bidirectional cost function')

    # loss hyperparameters
    parser.add_argument('--image-loss', default='mse', choices=['mse', 'ncc'], help='image reconstruction loss (default: mse)')
    parser.add_argument('--lambda', type=float, dest='lambda_weight', default=0.01, help='weight of gradient or KL loss (default: 0.01)')
    parser.add_argument('--kl-lambda', type=float, default=10.0, help='prior lambda regularisation for KL loss (default: 10)')
    parser.add_argument('--legacy-image-sigma', dest='image_sigma', type=float, default=1.0,
                        help='image noise parameter for MICCAI 2018 network (default: 1.0)')

    return parser


def prepare_data(args):
    train_files = vxm.py.utils.read_file_list(args.img_list, prefix=args.img_prefix, suffix=args.img_suffix)
    if not train_files:
        raise ValueError('Could not find any training data.')

    add_feat_axis = not args.multichannel

    if args.atlas:
        atlas = vxm.py.utils.load_volfile(args.atlas, np_var='vol', add_batch_axis=True, add_feat_axis=add_feat_axis)

        def make_generator():
            return vxm.generators.scan_to_atlas(
                train_files,
                atlas,
                batch_size=args.batch_size,
                bidir=args.bidir,
                add_feat_axis=add_feat_axis,
            )

    else:

        def make_generator():
            return vxm.generators.scan_to_scan(
                train_files,
                batch_size=args.batch_size,
                bidir=args.bidir,
                add_feat_axis=add_feat_axis,
            )

    sample = next(make_generator())
    source_sample = sample[0][0]
    inshape = source_sample.shape[1:-1]
    nfeats = source_sample.shape[-1]

    return make_generator, inshape, nfeats


def build_model(args, inshape, nfeats):
    enc_nf = args.enc if args.enc else [16, 32, 32, 32]
    dec_nf = args.dec if args.dec else [32, 32, 32, 32, 32, 16, 16]

    model = vxm.networks.VxmDense(
        inshape=inshape,
        nb_unet_features=[enc_nf, dec_nf],
        bidir=args.bidir,
        use_probs=args.use_probs,
        int_steps=args.int_steps,
        int_resolution=args.int_downsize,
        src_feats=nfeats,
        trg_feats=nfeats,
    )
    return model


def compile_model(model, args):
    if args.image_loss == 'ncc':
        image_loss = vxm.losses.NCC().loss
    elif args.image_loss == 'mse':
        image_loss = vxm.losses.MSE(args.image_sigma).loss
    else:
        raise ValueError(f'Unsupported image loss: {args.image_loss}')

    losses = [image_loss]
    weights = [1.0]
    if args.bidir:
        losses.append(image_loss)
        weights.append(1.0)

    if args.use_probs:
        flow_shape = tuple(model.outputs[-1].shape[1:-1])
        losses.append(vxm.losses.KL(args.kl_lambda, flow_shape).loss)
    else:
        losses.append(vxm.losses.Grad('l2', loss_mult=args.int_downsize).loss)
    weights.append(args.lambda_weight)

    model.compile(optimizer=optimizers.Adam(learning_rate=args.lr), loss=losses, loss_weights=weights)


def main():
    parser = build_argument_parser()
    args = parser.parse_args()

    device, _ = cli.setup_device(args.gpu)
    if device.startswith('cpu'):
        print('[voxelmorph] Using CPU backend')
    else:
        print(f'[voxelmorph] Using device {device}')

    generator_factory, inshape, nfeats = prepare_data(args)

    cli.ensure_directory(args.model_dir)
    save_path = os.path.join(args.model_dir, 'weights_epoch_{epoch:04d}.weights.h5')

    model = build_model(args, inshape, nfeats)
    cli.load_weights_if_available(model, args.load_weights)

    compile_model(model, args)

    checkpoint_cb = callbacks.ModelCheckpoint(
        filepath=save_path,
        save_weights_only=True,
        save_freq='epoch',
    )

    if args.initial_epoch == 0:
        cli.save_weights(model, save_path.format(epoch=0))

    generator = generator_factory()

    model.fit(
        generator,
        initial_epoch=args.initial_epoch,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        callbacks=[checkpoint_cb],
        verbose=1,
    )


if __name__ == '__main__':
    main()
