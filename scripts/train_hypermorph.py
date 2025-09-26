#!/usr/bin/env python

"""Train a HyperMorph model with the torch-backed Voxelmorph stack."""

import os
# Set Keras backend to PyTorch
os.environ['KERAS_BACKEND'] = 'torch'

from __future__ import annotations

import argparse
import numpy as np
import voxelmorph as vxm

from keras import callbacks, optimizers, ops

from . import _torch_utils as cli


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument('--img-list', required=True, help='line-separated list of training files')
    parser.add_argument('--img-prefix', help='optional input image file prefix')
    parser.add_argument('--img-suffix', help='optional input image file suffix')
    parser.add_argument('--atlas', help='atlas filename for scan-to-atlas training')
    parser.add_argument('--model-dir', default='models', help='model output directory (default: models)')
    parser.add_argument('--multichannel', action='store_true', help='specify that data has multiple channels')
    parser.add_argument('--test-reg', nargs=3, help='moving, fixed, and output filenames to sweep the trained model')

    parser.add_argument('--gpu', help='GPU ID numbers (default: all available)')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size (default: 1)')
    parser.add_argument('--epochs', type=int, default=6000, help='number of training epochs (default: 6000)')
    parser.add_argument('--steps-per-epoch', type=int, default=100, help='steps per epoch (default: 100)')
    parser.add_argument('--load-weights', help='optional weights file to initialise with')
    parser.add_argument('--initial-epoch', type=int, default=0, help='initial epoch number (default: 0)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')

    parser.add_argument('--enc', type=int, nargs='+', help='list of U-Net encoder filters (default: 16 32 32 32)')
    parser.add_argument('--dec', type=int, nargs='+', help='list of U-Net decoder filters (default: 32 32 32 32 32 16 16)')
    parser.add_argument('--int-steps', type=int, default=7, help='number of integration steps (default: 7)')
    parser.add_argument('--int-downsize', type=int, default=2, help='flow downsample factor for integration (default: 2)')

    parser.add_argument('--image-loss', default='mse', choices=['mse', 'ncc'], help='image reconstruction loss (default: mse)')
    parser.add_argument('--image-sigma', type=float, default=0.05, help='estimated image noise for mse scaling (default: 0.05)')
    parser.add_argument('--oversample-rate', type=float, default=0.2, help='hyperparameter oversample rate (default: 0.2)')

    return parser


def load_training_files(args):
    files = vxm.py.utils.read_file_list(args.img_list, prefix=args.img_prefix, suffix=args.img_suffix)
    if not files:
        raise ValueError('Could not find any training data.')
    return files


def make_base_generator(args, train_files, add_feat_axis):
    if args.atlas:
        atlas = vxm.py.utils.load_volfile(args.atlas, np_var='vol', add_batch_axis=True, add_feat_axis=add_feat_axis)
        return vxm.generators.scan_to_atlas(
            train_files,
            atlas,
            batch_size=args.batch_size,
            add_feat_axis=add_feat_axis,
        )

    return vxm.generators.scan_to_scan(
        train_files,
        batch_size=args.batch_size,
        add_feat_axis=add_feat_axis,
    )


def random_hyperparam(args):
    if np.random.rand() < args.oversample_rate:
        return np.random.choice([0.0, 1.0])
    return np.random.rand()


def make_generator(args, base_generator):
    while True:
        inputs, outputs = next(base_generator)
        hyp = np.array([random_hyperparam(args) for _ in range(args.batch_size)], dtype='float32')
        hyp = np.expand_dims(hyp, axis=-1)
        yield inputs + [hyp], outputs


def build_model(args, inshape, nfeats):
    enc_nf = args.enc if args.enc else [16, 32, 32, 32]
    dec_nf = args.dec if args.dec else [32, 32, 32, 32, 32, 16, 16]

    return vxm.networks.HyperVxmDense(
        inshape=inshape,
        nb_unet_features=[enc_nf, dec_nf],
        int_steps=args.int_steps,
        int_resolution=args.int_downsize,
        src_feats=nfeats,
        trg_feats=nfeats,
    )


def compile_model(model, args, image_loss_func):
    hyper_input = model.inputs[-1]
    hyp_scalar = ops.squeeze(hyper_input, axis=-1)

    def image_loss(y_true, y_pred):
        weight = 1.0 - hyp_scalar
        return weight * image_loss_func(y_true, y_pred)

    grad_loss = vxm.losses.Grad('l2', loss_mult=args.int_downsize).loss

    def reg_loss(y_true, y_pred):
        return hyp_scalar * grad_loss(y_true, y_pred)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=args.lr),
        loss=[image_loss, reg_loss],
    )


def evaluate_sweep(model, args, add_feat_axis):
    if not args.test_reg:
        return

    moving = vxm.py.utils.load_volfile(args.test_reg[0], add_batch_axis=True, add_feat_axis=add_feat_axis)
    fixed, fixed_affine = vxm.py.utils.load_volfile(
        args.test_reg[1], add_batch_axis=True, add_feat_axis=add_feat_axis, ret_affine=True,
    )

    warps = []
    for hyp in np.linspace(0, 1, 20, dtype='float32'):
        hyp_tensor = np.array([[hyp]], dtype='float32')
        warped = model.predict([moving, fixed, hyp_tensor], verbose=0)[0]
        warps.append(warped.squeeze())

    stacked = np.stack(warps, axis=-1)
    if stacked.ndim == 3:
        stacked = np.expand_dims(stacked, axis=-2)
    vxm.py.utils.save_volfile(stacked, args.test_reg[2], fixed_affine)


def main():
    args = build_parser().parse_args()
    cli.setup_device(args.gpu)

    train_files = load_training_files(args)
    cli.ensure_directory(args.model_dir)

    add_feat_axis = not args.multichannel
    base_generator = make_base_generator(args, train_files, add_feat_axis)
    generator = make_generator(args, base_generator)
    sample_inputs, sample_outputs = next(generator)

    inshape = sample_inputs[0].shape[1:-1]
    nfeats = sample_inputs[0].shape[-1]

    model = build_model(args, inshape, nfeats)
    cli.load_weights_if_available(model, args.load_weights)

    if args.image_loss == 'ncc':
        image_loss_func = vxm.losses.NCC().loss
    else:
        image_loss_func = vxm.losses.MSE(args.image_sigma).loss

    compile_model(model, args, image_loss_func)

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

    evaluate_sweep(model, args, add_feat_axis)


if __name__ == '__main__':
    main()
