#!/usr/bin/env python3

"""Train a joint SynthMorph model using the torch-backed Voxelmorph stack."""

import os
# Set Keras backend to PyTorch
os.environ['KERAS_BACKEND'] = 'torch'

from __future__ import annotations

import argparse
import numpy as np
import neurite as ne
import voxelmorph as vxm

from keras import Model, callbacks, layers as KL, optimizers, ops

from . import _torch_utils as cli


REF_TEXT = (
    'If you find this script useful, please consider citing:\n\n'
    '\tM Hoffmann, A Hoopes, DN Greve, B Fischl, AV Dalca\n'
    '\tAnatomy-aware and acquisition-agnostic joint registration with SynthMorph\n'
    '\tImaging Neuroscience, 2, pp 1-33, 2024\n'
    '\thttps://doi.org/10.1162/imag_a_00197\n'
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=type('formatter', (argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter), {}),
        description=f'Train a hyperparameter-conditioned SynthMorph model. {REF_TEXT}',
    )

    parser.add_argument('--label-dir', nargs='+', help='path or glob pattern pointing to input label maps')
    parser.add_argument('--model-dir', default='models', help='model output directory')
    parser.add_argument('--log-dir', help='optional TensorBoard log directory')
    parser.add_argument('--sub-dir', help='optional subfolder for logs and model saves')

    parser.add_argument('--shift', type=float, default=30.0, help='maximum translation amplitude')
    parser.add_argument('--rotate', type=float, default=45.0, help='maximum rotation amplitude')
    parser.add_argument('--scale', type=float, default=0.1, help='maximum scaling offset from 1')
    parser.add_argument('--shear', type=float, default=0.1, help='maximum shearing amplitude')
    parser.add_argument('--crop-prob', type=float, default=1.0, help='edge-cropping probability')
    parser.add_argument('--blur-max', type=float, default=3.4, help='maximum blurring SD')
    parser.add_argument('--slice-prob', type=float, default=1.0, help='downsampling probability')
    parser.add_argument('--out-shape', type=int, nargs='+', help='synthesis output shape')
    parser.add_argument('--out-labels', default='fs_large21.pickle', help='labels to optimise, see README')

    parser.add_argument('--gpu', help='ID of GPU to use')
    parser.add_argument('--epochs', type=int, default=10000, help='training epochs')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--init-epoch', type=int, default=0, help='initial epoch number')
    parser.add_argument('--init-weights', help='weights file to initialise the model with')
    parser.add_argument('--save-freq', type=int, default=100, help='epochs between model saves')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--loss-mult', type=float, default=10.0, help='similarity loss multiplier')
    parser.add_argument('--verbose', type=int, default=1, help='0 silent, 1 bar, 2 line/epoch')

    parser.add_argument('--enc', type=int, nargs='+', default=[256] * 4, help='encoder filters')
    parser.add_argument('--dec', type=int, nargs='+', default=[256] * 4, help='decoder filters')
    parser.add_argument('--add', type=int, nargs='+', default=[256] * 4, help='additional filters')
    parser.add_argument('--units', type=int, default=64, help='hypernetwork hidden units')
    parser.add_argument('--int-steps', type=int, default=5, help='number of integration steps')

    return parser


def prepare_directories(args):
    model_dir = args.model_dir
    log_dir = args.log_dir
    if args.sub_dir:
        model_dir = f'{model_dir}/{args.sub_dir}'
        if log_dir:
            log_dir = f'{log_dir}/{args.sub_dir}'
    cli.ensure_directory(model_dir)
    if log_dir:
        cli.ensure_directory(log_dir)
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


def build_joint_model(args, gen_model_1, gen_model_2, ima_1, map_1, ima_2, map_2):
    hyp_input = KL.Input(shape=(1,), name='hyperparam_input')

    nb_unet_features = [args.enc, args.dec] if args.enc and args.dec else None
    hyper_kwargs = dict(int_steps=args.int_steps)
    if nb_unet_features is not None:
        hyper_kwargs['nb_unet_features'] = nb_unet_features

    hyper_model = vxm.networks.HyperVxmJoint(
        inshape=ima_1.shape[1:-1],
        nb_hyp_params=1,
        nb_hyp_units=args.units,
        **hyper_kwargs,
    )

    outputs = hyper_model([ima_1, ima_2, hyp_input])
    if isinstance(outputs, (list, tuple)):
        warped = outputs[0]
        reg = outputs[-1]
    else:
        warped = outputs
        reg = hyper_model.references.pos_flow

    hyp_scalar = ops.squeeze(hyp_input, axis=-1)
    sim_loss = args.loss_mult * vxm.losses.MSE().loss(map_2, warped)
    reg_loss = vxm.losses.Grad('l2').loss(None, reg)
    combined = (1.0 - hyp_scalar) * sim_loss + hyp_scalar * reg_loss

    model_inputs = gen_model_1.inputs + gen_model_2.inputs + [hyp_input]
    model = Model(model_inputs, warped, name='synthmorph_joint')
    model.add_loss(combined)

    return model


def make_generator(args, base_generator):
    while True:
        (labels_src, labels_tgt), _ = next(base_generator)
        hyp = np.random.rand(args.batch_size, 1).astype('float32')
        inputs = [labels_src, labels_tgt, hyp]
        yield inputs, np.zeros((args.batch_size, 0))


def main():
    args = build_parser().parse_args()
    cli.setup_device(args.gpu)

    model_dir, log_dir = prepare_directories(args)
    labels_in, labels_out, base_generator, in_shape = load_labels(args)
    gen_model_1, gen_model_2, ima_1, map_1, ima_2, map_2 = build_generation_models(args, in_shape, labels_in, labels_out)

    joint_model = build_joint_model(args, gen_model_1, gen_model_2, ima_1, map_1, ima_2, map_2)
    if args.init_weights:
        joint_model.load_weights(args.init_weights)
    joint_model.compile(optimizers.Adam(learning_rate=args.lr))

    generator = make_generator(args, base_generator)
    steps_per_epoch = 100
    checkpoint_cb = callbacks.ModelCheckpoint(
        filepath=f'{model_dir}/{{epoch:05d}}.weights.h5',
        save_freq=steps_per_epoch * args.save_freq,
        save_weights_only=True,
    )

    callback_list = [checkpoint_cb]
    if log_dir:
        callback_list.append(callbacks.TensorBoard(log_dir=log_dir, write_graph=False))

    joint_model.fit(
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
