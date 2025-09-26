#!/usr/bin/env python

"""Train an unconditional atlas using the torch-backed Voxelmorph stack."""

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
    parser.add_argument('--img-prefix', help='optional input image file prefix')
    parser.add_argument('--img-suffix', help='optional input image file suffix')
    parser.add_argument('--init-template', help='initial template image')
    parser.add_argument('--model-dir', default='models', help='model output directory (default: models)')
    parser.add_argument('--multichannel', action='store_true', help='specify that data has multiple channels')

    parser.add_argument('--gpu', help='GPU number(s) - if not supplied, CPU is used')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size (default: 1)')
    parser.add_argument('--epochs', type=int, default=1500, help='number of training epochs (default: 1500)')
    parser.add_argument('--steps-per-epoch', type=int, default=100, help='steps per epoch (default: 100)')
    parser.add_argument('--load-weights', help='optional weights file to initialise with')
    parser.add_argument('--initial-epoch', type=int, default=0, help='initial epoch number (default: 0)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')

    parser.add_argument('--enc', type=int, nargs='+', help='list of unet encoder filters (default: 16 32 32 32)')
    parser.add_argument('--dec', type=int, nargs='+', help='list of unet decoder filters (default: 32 32 32 32 32 16 16)')

    parser.add_argument('--image-loss', default='ncc', choices=['ncc', 'mse'], help='image reconstruction loss (default: ncc)')
    parser.add_argument('--image-loss-weight', type=float, default=1.0, help='weight of reconstructed image loss (default: 1.0)')
    parser.add_argument('--mean-loss-weight', type=float, default=1.0, help='weight encouraging the atlas towards the data mean (default: 1.0)')
    parser.add_argument('--grad-loss-weight', type=float, default=1.0, help='weight of deformation smoothness loss (default: 1.0)')

    return parser


def load_training_data(args):
    train_files = vxm.py.utils.read_file_list(args.img_list, prefix=args.img_prefix, suffix=args.img_suffix)
    if not train_files:
        raise ValueError('Could not find any training data.')
    return train_files


def compute_initial_template(args, train_files):
    add_feat_axis = not args.multichannel
    if args.init_template:
        template = vxm.py.utils.load_volfile(args.init_template, add_batch_axis=True, add_feat_axis=add_feat_axis)
    else:
        navgs = min(100, len(train_files))
        print(f'[voxelmorph] Averaging first {navgs} scans to initialise the template.')
        template = 0
        for scan in train_files[:navgs]:
            template += vxm.py.utils.load_volfile(scan, add_batch_axis=True, add_feat_axis=add_feat_axis)
        template /= navgs
    return template


def make_generator(args, train_files, add_feat_axis):
    base = vxm.generators.template_creation(
        train_files,
        bidir=False,
        batch_size=args.batch_size,
        add_feat_axis=add_feat_axis,
    )

    while True:
        inputs, outputs = next(base)
        scan = outputs[0]
        zeros = outputs[-1]
        yield inputs, [scan, scan, zeros]


def set_initial_atlas(model, atlas):
    layer = model.get_layer(f'{model.name}_atlas_param')
    layer.set_weights([atlas.squeeze(axis=0)])


def extract_atlas(model):
    layer = model.get_layer(f'{model.name}_atlas_param')
    return layer.get_weights()[0]


def build_model(args, inshape, nfeats):
    enc_nf = args.enc if args.enc else [16, 32, 32, 32]
    dec_nf = args.dec if args.dec else [32, 32, 32, 32, 32, 16, 16]

    return vxm.networks.TemplateCreation(
        inshape=inshape,
        nb_unet_features=[enc_nf, dec_nf],
        src_feats=nfeats,
        atlas_feats=nfeats,
    )


def compile_model(model, args):
    if args.image_loss == 'ncc':
        image_loss = vxm.losses.NCC().loss
    else:
        image_loss = vxm.losses.MSE().loss

    mean_loss = vxm.losses.MSE().loss
    grad_loss = vxm.losses.Grad('l2').loss

    model.compile(
        optimizer=optimizers.Adam(learning_rate=args.lr),
        loss=[image_loss, mean_loss, grad_loss],
        loss_weights=[args.image_loss_weight, args.mean_loss_weight, args.grad_loss_weight],
    )


def main():
    args = build_parser().parse_args()
    cli.setup_device(args.gpu)

    train_files = load_training_data(args)
    cli.ensure_directory(args.model_dir)

    add_feat_axis = not args.multichannel
    template = compute_initial_template(args, train_files)
    vxm.py.utils.save_volfile(template.squeeze(), os.path.join(args.model_dir, 'init_template.nii.gz'))

    generator = make_generator(args, train_files, add_feat_axis)
    sample_inputs, _ = next(generator)
    inshape = sample_inputs[0].shape[1:-1]
    nfeats = sample_inputs[0].shape[-1]

    model = build_model(args, inshape, nfeats)
    set_initial_atlas(model, template.astype('float32'))
    if args.load_weights:
        model.load_weights(args.load_weights)
    compile_model(model, args)

    checkpoint_pattern = os.path.join(args.model_dir, 'weights_epoch_{epoch:04d}.weights.h5')
    checkpoint_cb = callbacks.ModelCheckpoint(filepath=checkpoint_pattern, save_weights_only=True, save_freq='epoch')

    if args.initial_epoch == 0:
        cli.save_weights(model, checkpoint_pattern.format(epoch=0))

    model.fit(
        generator,
        initial_epoch=args.initial_epoch,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        callbacks=[checkpoint_cb],
        verbose=1,
    )

    atlas = extract_atlas(model)
    vxm.py.utils.save_volfile(atlas.squeeze(), os.path.join(args.model_dir, 'template.nii.gz'))


if __name__ == '__main__':
    main()
