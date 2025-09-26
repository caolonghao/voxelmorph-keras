#!/usr/bin/env python3

"""Train an affine SynthMorph model using the torch-backed Voxelmorph stack."""

from __future__ import annotations

import argparse
import pathlib
import numpy as np
import neurite as ne
import voxelmorph as vxm

from keras import Model, callbacks, layers as KL, optimizers

from . import _torch_utils as cli


REF_TEXT = (
    'If you find this script useful, please consider citing:\n\n'
    '\tM Hoffmann, A Hoopes, B Fischl, AV Dalca\n'
    '\tAnatomy-specific acquisition-agnostic affine registration learned from fictitious images\n'
    '\tSPIE Medical Imaging: Image Processing, 12464, p 1246402, 2023\n'
    '\thttps://doi.org/10.1117/12.2653251\n'
    '\thttps://synthmorph.io/#papers (PDF)\n'
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=type('formatter', (argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter), {}),
        description=f'Train an affine SynthMorph model with synthetic image pairs. {REF_TEXT}',
    )

    parser.add_argument('--label-dir', nargs='+', help='path or glob pattern pointing to input label maps')
    parser.add_argument('--model-dir', type=pathlib.Path, default='models', help='model output directory')
    parser.add_argument('--log-dir', type=pathlib.Path, help='optional TensorBoard log directory')
    parser.add_argument('--sub-dir', help='optional subfolder for logs and model saves')

    parser.add_argument('--shift', type=float, default=30.0, help='maximum translation amplitude')
    parser.add_argument('--rotate', type=float, default=45.0, help='maximum rotation amplitude')
    parser.add_argument('--scale', type=float, default=0.1, help='maximum scaling offset from 1')
    parser.add_argument('--shear', type=float, default=0.1, help='maximum shearing amplitude')
    parser.add_argument('--crop-prob', type=float, default=1.0, help='edge-cropping probability')
    parser.add_argument('--blur-max', type=float, default=3.4, help='maximum blurring SD')
    parser.add_argument('--slice-prob', type=float, default=1.0, help='downsampling probability')
    parser.add_argument('--out-shape', type=int, nargs='+', default=[192] * 3, help='synthesis output shape')
    parser.add_argument('--out-labels', default='fs_lrc.pickle', help='labels to optimise, see README')

    parser.add_argument('--gpu', help='ID of GPU to use')
    parser.add_argument('--epochs', type=int, default=10000, help='training epochs')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--init-epoch', type=int, default=0, help='initial epoch number')
    parser.add_argument('--init-weights', help='weights file to initialise the model with')
    parser.add_argument('--save-freq', type=int, default=100, help='epochs between model saves')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--verbose', type=int, default=1, help='0 silent, 1 bar, 2 line/epoch')

    parser.add_argument('--enc', type=int, nargs='+', default=[256] * 4, help='encoder filters')
    parser.add_argument('--dec', type=int, nargs='+', default=[256] * 0, help='decoder filters')
    parser.add_argument('--add', type=int, nargs='+', default=[256] * 4, help='additional filters')
    parser.add_argument('--feat', type=int, default=64, help='number of feature maps')

    return parser


def prepare_directories(args):
    model_dir = args.model_dir
    log_dir = args.log_dir
    if args.sub_dir:
        model_dir = model_dir / args.sub_dir
        if log_dir:
            log_dir = log_dir / args.sub_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
    return model_dir, log_dir


def load_labels(args):
    labels_in, label_maps = vxm.py.utils.load_labels(args.label_dir)
    generator = vxm.generators.synthmorph(label_maps, batch_size=args.batch_size)
    in_shape = label_maps[0].shape

    labels_out = labels_in
    if args.out_labels:
        labels_out_loaded = np.load(args.out_labels, allow_pickle=True)
        if isinstance(labels_out_loaded, dict):
            labels_out = {k: v for k, v in labels_out_loaded.items() if k in labels_in}
        else:
            labels_out = {i: i for i in labels_out_loaded if i in labels_in}

    return labels_in, labels_out, generator, in_shape


def build_generation_models(args, in_shape, labels_in, labels_out):
    gen_args = dict(
        in_shape=in_shape,
        out_shape=args.out_shape,
        labels_in=labels_in,
        labels_out=labels_out,
        aff_shift=args.shift,
        aff_rotate=args.rotate,
        aff_scale=args.scale,
        aff_shear=args.shear,
        blur_max=args.blur_max,
        crop_prob=args.crop_prob,
        slice_prob=args.slice_prob,
    )

    gen_model_1 = ne.models.labels_to_image(**gen_args, id=0)
    gen_model_2 = ne.models.labels_to_image(**gen_args, id=1)
    ima_1, map_1 = gen_model_1.outputs
    ima_2, map_2 = gen_model_2.outputs

    return gen_model_1, gen_model_2, ima_1, map_1, ima_2, map_2


def build_affine_pipeline(args, gen_model_1, gen_model_2, ima_1, map_1, ima_2, map_2):
    labels_src = gen_model_1.inputs[0]
    labels_tgt = gen_model_2.inputs[0]

    affine_model = vxm.networks.VxmAffineFeatureDetector(
        in_shape=ima_1.shape[1:-1],
        num_chan=ima_1.shape[-1],
        enc_nf=args.enc,
        dec_nf=args.dec,
        add_nf=args.add,
        num_feat=args.feat,
        bidir=True,
        make_dense=True,
    )

    aff_1, aff_2 = affine_model([ima_1, ima_2])
    warp_kwargs = dict(fill_value=0, shape=aff_1.shape[1:-1], shift_center=False)
    mov_1 = vxm.layers.SpatialTransformer(**warp_kwargs)((map_1, aff_1))
    mov_2 = vxm.layers.SpatialTransformer(**warp_kwargs)((map_2, aff_2))

    return labels_src, labels_tgt, mov_1, mov_2, map_2


class AddLoss(KL.Layer):
    def call(self, tensors):
        moving, target = tensors
        self.add_loss(vxm.losses.MSE().loss(moving, target))
        return moving


def main():
    args = build_parser().parse_args()
    cli.setup_device(args.gpu)

    model_dir, log_dir = prepare_directories(args)
    labels_in, labels_out, generator, in_shape = load_labels(args)
    gen_model_1, gen_model_2, ima_1, map_1, ima_2, map_2 = build_generation_models(args, in_shape, labels_in, labels_out)

    labels_src, labels_tgt, mov_1, mov_2, map_2_final = build_affine_pipeline(args, gen_model_1, gen_model_2, ima_1, map_1, ima_2, map_2)

    moving_loss = AddLoss(name='mse_loss')((mov_1, map_2_final))
    model = Model([labels_src, labels_tgt], moving_loss, name='synthmorph_affine')
    model.compile(optimizers.Adam(learning_rate=args.lr))

    if args.init_weights:
        model.load_weights(args.init_weights)

    steps_per_epoch = 100
    checkpoint_cb = callbacks.ModelCheckpoint(
        filepath=model_dir / '{epoch:05d}.weights.h5',
        save_freq=steps_per_epoch * args.save_freq,
        save_weights_only=True,
    )

    callback_list = [checkpoint_cb]
    if log_dir:
        callback_list.append(callbacks.TensorBoard(log_dir=log_dir, write_graph=False))

    model.fit(
        generator,
        initial_epoch=args.init_epoch,
        epochs=args.epochs,
        callbacks=callback_list,
        steps_per_epoch=steps_per_epoch,
        verbose=args.verbose,
    )

    print(f'\nThank you for using SynthMorph! {REF_TEXT}')


if __name__ == '__main__':
    main()
