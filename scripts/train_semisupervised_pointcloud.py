#!/usr/bin/env python

"""Train semi-supervised registration with surface point-cloud guidance."""

from __future__ import annotations

import argparse
import os

# Set Keras backend to PyTorch
os.environ['KERAS_BACKEND'] = 'torch'

import numpy as np
import voxelmorph as vxm

from keras import callbacks, optimizers

from . import _torch_utils as cli


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument('--img-list', required=True, help='line-separated list of training files')
    parser.add_argument('--img-prefix', help='optional input image file prefix')
    parser.add_argument('--img-suffix', help='optional input image file suffix')
    parser.add_argument('--atlas', required=True, help='atlas filename (NPZ with vol/seg)')
    parser.add_argument('--model-dir', default='models', help='model output directory (default: models)')
    parser.add_argument('--multichannel', action='store_true', help='specify that data has multiple channels')
    parser.add_argument('--smooth-seg', type=float, default=0.1, help='segmentation smoothness sigma (default: 0.1)')
    parser.add_argument('--labels', type=int, nargs='+', help='labels to include (default: all atlas labels)')
    parser.add_argument('--align-segs', action='store_true', help='align segmentations instead of images (single label only)')

    parser.add_argument('--gpu', help='GPU ID numbers (default: all available)')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size (default: 1)')
    parser.add_argument('--epochs', type=int, default=1500, help='number of training epochs (default: 1500)')
    parser.add_argument('--steps-per-epoch', type=int, default=100, help='steps per epoch (default: 100)')
    parser.add_argument('--load-weights', help='optional weights file to initialise with')
    parser.add_argument('--initial-epoch', type=int, default=0, help='initial epoch number (default: 0)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')

    parser.add_argument('--enc', type=int, nargs='+', help='list of U-Net encoder filters (default: 16 32 32 32)')
    parser.add_argument('--dec', type=int, nargs='+', help='list of U-Net decoder filters (default: 32 32 32 32 32 16 16)')
    parser.add_argument('--int-steps', type=int, default=7, help='number of integration steps (default: 7)')
    parser.add_argument('--int-downsize', type=int, default=2, help='flow downsample factor for integration (default: 2)')
    parser.add_argument('--use-probs', action='store_true', help='enable probabilistic velocity field')
    parser.add_argument('--bidir', action='store_true', help='enable bidirectional image loss')
    parser.add_argument('--surf-points', type=int, default=5000, help='number of surface points to warp (default: 5000)')
    parser.add_argument('--sdt-resize', type=float, default=1.0, help='resize factor for distance transform (default: 1.0)')
    parser.add_argument('--num-labels', type=int, help='number of labels to sample per batch (default: all)')

    parser.add_argument('--image-loss', default='mse', choices=['mse', 'ncc'], help='image reconstruction loss (default: mse)')
    parser.add_argument('--lambda', dest='lambda_weight', type=float, default=0.01, help='weight of gradient/KL loss (default: 0.01)')
    parser.add_argument('--dt-sigma', type=float, default=1.0, help='surface noise parameter (default: 1.0)')
    parser.add_argument('--kl-lambda', type=float, default=10.0, help='prior lambda regularisation for KL loss (default: 10)')

    return parser


def load_training_files(args):
    files = vxm.py.utils.read_file_list(args.img_list, prefix=args.img_prefix, suffix=args.img_suffix)
    if not files:
        raise ValueError('Could not find any training data.')
    return files


def load_atlas(args):
    atlas_vol = vxm.py.utils.load_volfile(args.atlas, np_var='vol', add_batch_axis=False, add_feat_axis=False)
    atlas_seg = vxm.py.utils.load_volfile(args.atlas, np_var='seg', add_batch_axis=False, add_feat_axis=False)
    return atlas_vol, atlas_seg


def resolve_labels(args, atlas_seg):
    if args.labels:
        labels = np.array(args.labels)
    else:
        labels = np.sort(np.unique(atlas_seg))[1:]
    if args.align_segs and len(labels) != 1:
        raise ValueError('align_segs requires exactly one label.')
    nb_labels_sample = args.num_labels if args.num_labels is not None else len(labels)
    return labels, nb_labels_sample


def make_generator(args, train_files, atlas_vol, atlas_seg, labels, nb_labels_sample, add_feat_axis):
    base = vxm.generators.surf_semisupervised(
        train_files,
        atlas_vol,
        atlas_seg,
        nb_surface_pts=args.surf_points,
        labels=labels,
        batch_size=args.batch_size,
        surf_bidir=True,
        smooth_seg_std=args.smooth_seg,
        nb_labels_sample=nb_labels_sample,
        sdt_vol_resize=args.sdt_resize,
        align_segs=args.align_segs,
        add_feat_axis=add_feat_axis,
    )

    while True:
        inputs, outputs = next(base)
        source = inputs[0].astype('float32')
        target = inputs[1].astype('float32')
        subj_surface = inputs[4][..., :-1].astype('float32')
        atlas_surface = inputs[5][..., :-1].astype('float32')
        zeros = outputs[2].astype('float32')

        y_trues = [target]
        if args.bidir:
            y_trues.append(source)
        y_trues.append(atlas_surface)
        y_trues.append(zeros)

        yield [source, target, subj_surface], y_trues


def build_model(args, inshape, nfeats, nb_surface_points):
    enc_nf = args.enc if args.enc else [16, 32, 32, 32]
    dec_nf = args.dec if args.dec else [32, 32, 32, 32, 32, 16, 16]

    return vxm.networks.VxmDenseSemiSupervisedPointCloud(
        inshape=inshape,
        nb_surface_points=nb_surface_points,
        nb_unet_features=[enc_nf, dec_nf],
        bidir=args.bidir,
        use_probs=args.use_probs,
        int_steps=args.int_steps,
        int_resolution=args.int_downsize,
        src_feats=nfeats,
        trg_feats=nfeats,
    )


def compile_model(model, args, image_loss):
    losses = [image_loss]
    weights = [0.5 if args.bidir else 1.0]

    if args.bidir:
        losses.append(image_loss)
        weights.append(0.5)

    losses.append(vxm.losses.MSE().loss)
    weights.append(0.25 / (args.dt_sigma ** 2))

    if args.use_probs:
        flow_shape = tuple(int(d) for d in model.references.pos_flow.shape[1:-1])
        losses.append(vxm.losses.KL(args.kl_lambda, flow_shape).loss)
    else:
        losses.append(vxm.losses.Grad('l2', loss_mult=args.int_downsize).loss)
    weights.append(args.lambda_weight)

    model.compile(optimizer=optimizers.Adam(learning_rate=args.lr), loss=losses, loss_weights=weights)


def main():
    args = build_parser().parse_args()
    cli.setup_device(args.gpu)

    train_files = load_training_files(args)
    cli.ensure_directory(args.model_dir)

    add_feat_axis = not args.multichannel
    atlas_vol, atlas_seg = load_atlas(args)
    labels, nb_labels_sample = resolve_labels(args, atlas_seg)

    generator = make_generator(args, train_files, atlas_vol, atlas_seg, labels, nb_labels_sample, add_feat_axis)
    sample_inputs, _ = next(generator)

    inshape = sample_inputs[0].shape[1:-1]
    nfeats = sample_inputs[0].shape[-1]
    nb_surface_points = sample_inputs[2].shape[1]

    model = build_model(args, inshape, nfeats, nb_surface_points)
    cli.load_weights_if_available(model, args.load_weights)

    if args.image_loss == 'ncc':
        image_loss = vxm.losses.NCC().loss
    else:
        image_loss = vxm.losses.MSE().loss

    compile_model(model, args, image_loss)

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


if __name__ == '__main__':
    main()
