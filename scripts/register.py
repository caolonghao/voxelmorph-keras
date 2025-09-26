#!/usr/bin/env python

"""Register two volumes using a trained torch-backed Voxelmorph model."""

from __future__ import annotations

import argparse
import voxelmorph as vxm

from . import _torch_utils as cli


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--moving', required=True, help='moving image (source) filename')
    parser.add_argument('--fixed', required=True, help='fixed image (target) filename')
    parser.add_argument('--moved', required=True, help='warped image output filename')
    parser.add_argument('--model', required=True, help='trained Voxelmorph weights (LoadableModel)')
    parser.add_argument('--warp', help='output warp deformation filename')
    parser.add_argument('--gpu', help='GPU number(s) - if not supplied, CPU is used')
    parser.add_argument('--multichannel', action='store_true', help='indicate that data has multiple channels')
    return parser.parse_args()


def main():
    args = parse_args()
    cli.setup_device(args.gpu)

    add_feat_axis = not args.multichannel
    moving = vxm.py.utils.load_volfile(args.moving, add_batch_axis=True, add_feat_axis=add_feat_axis)
    fixed, fixed_affine = vxm.py.utils.load_volfile(args.fixed, add_batch_axis=True, add_feat_axis=add_feat_axis, ret_affine=True)

    inshape = moving.shape[1:-1]
    nb_feats = moving.shape[-1]

    model = vxm.networks.VxmDense.load(args.model, inshape=inshape, input_model=None)
    warp = model.register(moving, fixed)

    transformer = vxm.networks.Transform(inshape, nb_feats=nb_feats)
    moved = transformer.predict([moving, warp])

    if args.warp:
        vxm.py.utils.save_volfile(warp.squeeze(), args.warp, fixed_affine)

    vxm.py.utils.save_volfile(moved.squeeze(), args.moved, fixed_affine)


if __name__ == '__main__':
    main()

