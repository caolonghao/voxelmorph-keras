#!/usr/bin/env python

"""Instance-specific optimisation on the torch-backed Keras stack."""

from __future__ import annotations

import argparse
import numpy as np
import voxelmorph as vxm

from keras import optimizers

from . import _torch_utils as cli


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument('--moving', required=True, help='moving image (source) filename')
    parser.add_argument('--fixed', required=True, help='fixed image (target) filename')
    parser.add_argument('--moved', required=True, help='registered image output filename')
    parser.add_argument('--model', help='weights from a pretrained vxm model for initialisation')
    parser.add_argument('--warp', help='output warp filename')
    parser.add_argument('--multichannel', action='store_true', help='specify that data has multiple channels')

    parser.add_argument('-g', '--gpu', help='GPU number(s) - if not supplied, CPU is used')
    parser.add_argument('--steps', type=int, default=200, help='number of optimisation steps (default: 200)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--multiplier', type=float, default=1000.0, help='local parameter scaling (default: 1000)')

    parser.add_argument('--image-loss', default='mse', choices=['mse', 'ncc'], help='image reconstruction loss (default: mse)')
    parser.add_argument('--lambda', dest='lambda_weight', type=float, default=0.01, help='weight of gradient loss (default: 0.01)')

    return parser


def load_data(args):
    add_feat_axis = not args.multichannel
    moving = vxm.py.utils.load_volfile(args.moving, add_batch_axis=True, add_feat_axis=add_feat_axis)
    fixed, fixed_affine = vxm.py.utils.load_volfile(
        args.fixed,
        add_batch_axis=True,
        add_feat_axis=add_feat_axis,
        ret_affine=True,
    )
    return moving, fixed, fixed_affine


def initialise_flow(model, init_flow: np.ndarray) -> None:
    layer = model.get_layer(f'{model.name}_local_param')
    mult = getattr(layer, 'mult', 1.0)
    kernel = init_flow.squeeze(axis=0) / mult
    layer.set_weights([kernel])


def main():
    args = build_parser().parse_args()
    cli.setup_device(args.gpu)

    moving, fixed, fixed_affine = load_data(args)
    inshape = moving.shape[1:-1]
    nb_feats = moving.shape[-1]

    model = vxm.networks.InstanceDense(
        inshape=inshape,
        nb_feats=nb_feats,
        mult=args.multiplier,
    )

    if args.model:
        vxm_model = vxm.networks.VxmDense.load(args.model)
        init_flow = vxm_model.register(moving, fixed)
        initialise_flow(model, np.asarray(init_flow, dtype='float32'))

    if args.image_loss == 'ncc':
        image_loss = vxm.losses.NCC().loss
    else:
        image_loss = vxm.losses.MSE().loss

    grad_loss = vxm.losses.Grad('l2').loss

    model.compile(
        optimizer=optimizers.Adam(learning_rate=args.lr),
        loss=[image_loss, grad_loss],
        loss_weights=[1.0, args.lambda_weight],
    )

    zeros = np.zeros((1, *inshape, len(inshape)), dtype='float32')

    model.fit(
        [moving],
        [fixed, zeros],
        batch_size=None,
        epochs=args.steps,
        steps_per_epoch=1,
        verbose=1,
    )

    warped, flow = model.predict([moving], verbose=0)
    vxm.py.utils.save_volfile(warped.squeeze(), args.moved, fixed_affine)

    if args.warp:
        vxm.py.utils.save_volfile(flow.squeeze(), args.warp, fixed_affine)


if __name__ == '__main__':
    main()
