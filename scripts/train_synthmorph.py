#!/usr/bin/env python3

"""Train a SynthMorph model using the torch-backed Voxelmorph stack."""

from __future__ import annotations

import argparse
import os

# Set Keras backend to PyTorch
os.environ['KERAS_BACKEND'] = 'torch'

import pickle
import numpy as np
import neurite as ne
import voxelmorph as vxm

from keras import Model, callbacks, optimizers

from . import _torch_utils as cli


REF_TEXT = (
    'If you find this script useful, please consider citing:\n\n'
    '\tM Hoffmann, B Billot, DN Greve, JE Iglesias, B Fischl, AV Dalca\n'
    '\tSynthMorph: learning contrast-invariant registration without acquired images\n'
    '\tIEEE Transactions on Medical Imaging (TMI), 41 (3), 543-558, 2022\n'
    '\thttps://doi.org/10.1109/TMI.2021.3116879\n'
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=type('formatter', (argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter), {}),
        description=f'Train a SynthMorph model on synthetic images derived from label maps. {REF_TEXT}',
    )

    parser.add_argument('--label-dir', nargs='+', help='path or glob pattern pointing to input label maps')
    parser.add_argument('--model-dir', default='models', help='model output directory')
    parser.add_argument('--log-dir', help='optional TensorBoard log directory')
    parser.add_argument('--sub-dir', help='optional subfolder for logs and model saves')

    parser.add_argument('--same-subj', action='store_true', help='generate image pairs from same label map')
    parser.add_argument('--blur-std', type=float, default=1.0, help='maximum blurring std. dev.')
    parser.add_argument('--gamma', type=float, default=0.25, help='std. dev. of gamma augmentation')
    parser.add_argument('--vel-std', type=float, default=0.5, help='std. dev. of SVF sampling')
    parser.add_argument('--vel-res', type=float, nargs='+', default=[16], help='SVF scale(s)')
    parser.add_argument('--bias-std', type=float, default=0.3, help='std. dev. of bias field')
    parser.add_argument('--bias-res', type=float, nargs='+', default=[40], help='bias scale(s)')
    parser.add_argument('--out-shape', type=int, nargs='+', help='output shape to pad to')
    parser.add_argument('--out-labels', default='fs_labels.npy', help='labels to optimise, see README')

    parser.add_argument('--gpu', help='ID of GPU to use')
    parser.add_argument('--epochs', type=int, default=1500, help='training epochs')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--init-weights', help='optional weights file to initialise with')
    parser.add_argument('--save-freq', type=int, default=20, help='epochs between model saves')
    parser.add_argument('--reg-param', type=float, default=1.0, help='regularisation weight')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--init-epoch', type=int, default=0, help='initial epoch number')
    parser.add_argument('--verbose', type=int, default=1, help='0 silent, 1 bar, 2 line/epoch')

    parser.add_argument('--int-steps', type=int, default=5, help='number of integration steps')
    parser.add_argument('--enc', type=int, nargs='+', default=[64] * 4, help='U-Net encoder filters')
    parser.add_argument('--dec', type=int, nargs='+', default=[64] * 6, help='U-Net decoder filters')

    return parser


def prepare_directories(args):
    model_dir = args.model_dir
    log_dir = args.log_dir
    if args.sub_dir:
        model_dir = os.path.join(model_dir, args.sub_dir)
        if log_dir:
            log_dir = os.path.join(log_dir, args.sub_dir)
    cli.ensure_directory(model_dir)
    if log_dir:
        cli.ensure_directory(log_dir)
    return model_dir, log_dir


def load_labels(args):
    labels_in, label_maps = vxm.py.utils.load_labels(args.label_dir)
    generator = vxm.generators.synthmorph(
        label_maps,
        batch_size=args.batch_size,
        same_subj=args.same_subj,
        flip=True,
    )
    in_shape = label_maps[0].shape

    if args.out_labels.endswith('.npy'):
        labels_out = sorted(x for x in np.load(args.out_labels) if x in labels_in)
    elif args.out_labels.endswith('.pickle'):
        with open(args.out_labels, 'rb') as f:
            labels_out = {k: v for k, v in pickle.load(f).items() if k in labels_in}
    else:
        labels_out = labels_in

    return labels_in, labels_out, label_maps, generator, in_shape


def build_generation_models(args, in_shape, labels_in, labels_out):
    gen_args = dict(
        in_shape=in_shape,
        out_shape=args.out_shape,
        labels_in=labels_in,
        labels_out=labels_out,
        warp_std=args.vel_std,
        warp_res=args.vel_res,
        blur_std=args.blur_std,
        bias_std=args.bias_std,
        bias_res=args.bias_res,
        gamma_std=args.gamma,
    )

    gen_model_1 = ne.models.labels_to_image_old(**gen_args, id=0)
    gen_model_2 = ne.models.labels_to_image_old(**gen_args, id=1)
    ima_1, map_1 = gen_model_1.outputs
    ima_2, map_2 = gen_model_2.outputs

    return gen_model_1, gen_model_2, ima_1, map_1, ima_2, map_2


def build_registration_model(args, gen_model_1, gen_model_2, ima_1, ima_2):
    inputs = gen_model_1.inputs + gen_model_2.inputs
    input_model = Model(inputs, outputs=(ima_1, ima_2), name='synthmorph_inputs')

    model = vxm.networks.VxmDense(
        inshape=ima_1.shape[1:-1],
        nb_unet_features=[args.enc, args.dec],
        int_steps=args.int_steps,
        int_resolution=2,
        svf_resolution=2,
        input_model=input_model,
        src_feats=ima_1.shape[-1],
        trg_feats=ima_2.shape[-1],
    )

    return model


def add_losses(model, map_1, map_2, args):
    flow = model.references.pos_flow
    pred = vxm.layers.SpatialTransformer(interp_method='linear', name='pred')([map_1, flow])

    dice_loss = vxm.losses.Dice().loss(map_2, pred)
    grad_loss = vxm.losses.Grad('l2', loss_mult=args.reg_param).loss(None, flow)

    model.add_loss(dice_loss)
    model.add_loss(grad_loss)


def main():
    args = build_parser().parse_args()
    cli.setup_device(args.gpu)

    model_dir, log_dir = prepare_directories(args)

    labels_in, labels_out, label_maps, generator, in_shape = load_labels(args)
    gen_model_1, gen_model_2, ima_1, map_1, ima_2, map_2 = build_generation_models(args, in_shape, labels_in, labels_out)

    model = build_registration_model(args, gen_model_1, gen_model_2, ima_1, ima_2)
    add_losses(model, map_1, map_2, args)
    model.compile(optimizer=optimizers.Adam(learning_rate=args.lr))

    if args.init_weights:
        model.load_weights(args.init_weights)

    steps_per_epoch = 100
    checkpoint_cb = callbacks.ModelCheckpoint(
        os.path.join(model_dir, '{epoch:05d}.weights.h5'),
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
