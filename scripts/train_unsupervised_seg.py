#!/usr/bin/env python

"""Train probabilistic atlas segmentation with the torch-backed Voxelmorph stack."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import voxelmorph as vxm

from keras import callbacks, optimizers

from . import _torch_utils as cli


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument('--img-list', required=True, help='line-separated list of training files')
    parser.add_argument('--img-prefix', help='optional input image file prefix')
    parser.add_argument('--img-suffix', help='optional input image file suffix')
    parser.add_argument('--atlas', required=True, help='probabilistic atlas npz file')
    parser.add_argument('--mapping', help='atlas label mapping npz file')
    parser.add_argument('--init-stat', help='npz file with init_mu and init_sigma arrays')
    parser.add_argument('--model-dir', default='models', help='model output directory (default: models)')

    parser.add_argument('--gpu', help='GPU ID numbers')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size (default: 1)')
    parser.add_argument('--epochs', type=int, default=1500, help='number of training epochs (default: 1500)')
    parser.add_argument('--steps-per-epoch', type=int, default=100, help='steps per epoch (default: 100)')
    parser.add_argument('--load-weights', help='optional weights file to initialise model')
    parser.add_argument('--initial-epoch', type=int, default=0, help='initial epoch number (default: 0)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')

    parser.add_argument('--enc', type=int, nargs='+', help='list of unet encoder filters (default: 16 32 32 32)')
    parser.add_argument('--dec', type=int, nargs='+', help='list of unet decoder filters (default: 32 32 32 32 32 16 16)')
    parser.add_argument('--no-warp-atlas', action='store_true', help='disable atlas warping within the network')
    parser.add_argument('--stat-pre-warp', action='store_true', help='compute gaussian stats before warping the atlas')
    parser.add_argument('--grad-loss-weight', type=float, default=10.0, help='weight of gradient regulariser (default: 10.0)')

    return parser


def load_atlas(atlas_path: str, mapping_path: str | None):
    atlas_full = vxm.py.utils.load_volfile(atlas_path, add_batch_axis=True)
    if mapping_path:
        mapping = np.load(mapping_path)['mapping'].astype('int').flatten()
        if len(mapping) != atlas_full.shape[-1]:
            raise ValueError('Mapping shape mismatched with atlas channels.')
        nb_labels = int(1 + mapping.max())
        atlas = np.zeros([*atlas_full.shape[:-1], nb_labels])
        for idx, label in enumerate(mapping):
            atlas[0, ..., label] += atlas_full[0, ..., idx]
    else:
        atlas = atlas_full
        nb_labels = atlas.shape[-1]
        mapping = None
    return atlas, atlas_full, mapping


def prepare_generator(train_files, atlas, batch_size):
    def make_generator():
        return vxm.generators.scan_to_atlas(train_files, atlas, batch_size=batch_size, add_feat_axis=True)

    sample = next(make_generator())
    inshape = sample[0][0].shape[1:-1]
    return make_generator, inshape


def build_model(args, inshape, nb_labels, init_mu, init_sigma):
    enc_nf = args.enc if args.enc else [16, 32, 32, 32]
    dec_nf = args.dec if args.dec else [32, 32, 32, 32, 32, 16, 16]

    model = vxm.networks.ProbAtlasSegmentation(
        inshape,
        nb_labels=nb_labels,
        nb_unet_features=[enc_nf, dec_nf],
        stat_post_warp=(not args.stat_pre_warp),
        warp_atlas=(not args.no_warp_atlas),
        init_mu=init_mu,
        init_sigma=init_sigma,
    )
    return model


def compile_model(model, args):
    def loss(_, yp):
        atlas_indicator = model.inputs[1] > 0
        atlas_indicator = ops.cast(atlas_indicator, 'float32')
        yp = ops.cast(yp, 'float32')
        return -ops.sum(yp * atlas_indicator) / ops.maximum(ops.sum(atlas_indicator), 1e-6)

    grad_weight = args.grad_loss_weight if not args.no_warp_atlas else 0.0
    losses = [loss, vxm.losses.Grad('l2', loss_mult=2).loss]
    weights = [1.0, grad_weight]
    model.compile(optimizer=optimizers.Adam(learning_rate=args.lr), loss=losses, loss_weights=weights)


def main():
    parser = build_parser()
    args = parser.parse_args()

    cli.ensure_directory(args.model_dir)

    atlas, atlas_full, mapping = load_atlas(args.atlas, args.mapping)
    init_mu = np.load(args.init_stat)['init_mu'] if args.init_stat else None
    init_sigma = np.load(args.init_stat)['init_sigma'] if args.init_stat else None

    train_files = vxm.py.utils.read_file_list(args.img_list, prefix=args.img_prefix, suffix=args.img_suffix)
    if not train_files:
        raise ValueError('Could not find any training data.')

    generator_factory, inshape = prepare_generator(train_files, atlas, args.batch_size)
    nb_labels = atlas.shape[-1]

    model = build_model(args, inshape, nb_labels, init_mu, init_sigma)
    cli.load_weights_if_available(model, args.load_weights)
    compile_model(model, args)

    checkpoint_cb = callbacks.ModelCheckpoint(
        filepath=os.path.join(args.model_dir, 'weights_epoch_{epoch:04d}.weights.h5'),
        save_weights_only=True,
        save_freq='epoch',
    )

    if args.initial_epoch == 0:
        cli.save_weights(model, os.path.join(args.model_dir, 'weights_epoch_0000.weights.h5'))

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

