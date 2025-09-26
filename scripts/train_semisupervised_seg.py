#!/usr/bin/env python

"""Train a semi-supervised Voxelmorph registration model with segmentation supervision."""

from __future__ import annotations

import argparse
import os
import numpy as np
import voxelmorph as vxm

from keras import callbacks, optimizers

from . import _torch_utils as cli


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument('--img-list', required=True, help='line-separated list of training files')
    parser.add_argument('--img-suffix', help='input image file suffix')
    parser.add_argument('--seg-suffix', help='input segmentation file suffix')
    parser.add_argument('--img-prefix', help='input image file prefix')
    parser.add_argument('--seg-prefix', help='input segmentation file prefix')
    parser.add_argument('--labels', required=True, help='npy file with label list for dice computation')
    parser.add_argument('--model-dir', default='models', help='model output directory (default: models)')
    parser.add_argument('--atlas', help='optional atlas filename for scan-to-atlas training')

    parser.add_argument('--gpu', help='GPU ID numbers (default: all available)')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size (default: 1)')
    parser.add_argument('--epochs', type=int, default=1500, help='number of training epochs (default: 1500)')
    parser.add_argument('--steps-per-epoch', type=int, default=100, help='steps per epoch (default: 100)')
    parser.add_argument('--load-weights', help='optional weights file to initialise with')
    parser.add_argument('--initial-epoch', type=int, default=0, help='initial epoch number (default: 0)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')

    parser.add_argument('--enc', type=int, nargs='+', help='unet encoder filters (default: 16 32 32 32)')
    parser.add_argument('--dec', type=int, nargs='+', help='unet decoder filters (default: 32 32 32 32 32 16 16)')
    parser.add_argument('--int-steps', type=int, default=7, help='number of integration steps (default: 7)')
    parser.add_argument('--int-downsize', type=int, default=2, help='flow downsample factor for integration (default: 2)')

    parser.add_argument('--image-loss', default='mse', choices=['mse', 'ncc'], help='image reconstruction loss (default: mse)')
    parser.add_argument('--grad-loss-weight', type=float, default=0.01, help='weight of gradient loss (default: 0.01)')
    parser.add_argument('--dice-loss-weight', type=float, default=0.01, help='weight of dice loss (default: 0.01)')

    return parser


def prepare_generators(args, train_imgs, train_segs, labels):
    def make_generator():
        return vxm.generators.semisupervised(
            train_imgs,
            train_segs,
            labels=labels,
            atlas_file=args.atlas,
            batch_size=args.batch_size,
        )

    sample = next(make_generator())
    inshape = sample[0][0].shape[1:-1]
    return make_generator, inshape


def build_model(args, inshape, nb_labels):
    enc_nf = args.enc if args.enc else [16, 32, 32, 32]
    dec_nf = args.dec if args.dec else [32, 32, 32, 32, 32, 16, 16]

    model = vxm.networks.VxmDenseSemiSupervisedSeg(
        inshape=inshape,
        nb_unet_features=[enc_nf, dec_nf],
        nb_labels=nb_labels,
        int_steps=args.int_steps,
        int_resolution=args.int_downsize,
        bidir=False,
        bidir_labels=False,
    )
    return model


def compile_model(model, args, labels):
    if args.image_loss == 'ncc':
        image_loss = vxm.losses.NCC().loss
    else:
        image_loss = vxm.losses.MSE().loss

    grad_loss = vxm.losses.Grad('l2', loss_mult=args.int_downsize).loss
    dice_loss = vxm.losses.Dice().loss

    model.compile(
        optimizer=optimizers.Adam(learning_rate=args.lr),
        loss=[image_loss, grad_loss, dice_loss],
        loss_weights=[1.0, args.grad_loss_weight, args.dice_loss_weight],
    )


def main():
    args = build_parser().parse_args()
    cli.setup_device(args.gpu)

    if args.img_prefix == args.seg_prefix and args.img_suffix == args.seg_suffix:
        raise ValueError('Image and segmentation suffix/prefix must differ.')

    train_imgs = vxm.py.utils.read_file_list(args.img_list, prefix=args.img_prefix, suffix=args.img_suffix)
    train_segs = vxm.py.utils.read_file_list(args.img_list, prefix=args.seg_prefix, suffix=args.seg_suffix)
    if not train_imgs:
        raise ValueError('Could not find any training data.')

    labels = np.load(args.labels)
    generator_factory, inshape = prepare_generators(args, train_imgs, train_segs, labels)

    model = build_model(args, inshape, len(labels))
    cli.load_weights_if_available(model, args.load_weights)
    compile_model(model, args, labels)

    save_pattern = os.path.join(args.model_dir, 'weights_epoch_{epoch:04d}.weights.h5')
    cli.ensure_directory(args.model_dir)
    checkpoint_cb = callbacks.ModelCheckpoint(filepath=save_pattern, save_weights_only=True, save_freq='epoch')

    if args.initial_epoch == 0:
        cli.save_weights(model, save_pattern.format(epoch=0))

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

