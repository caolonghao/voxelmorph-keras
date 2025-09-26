#!/usr/bin/env python

"""Apply a saved warp field to a moving image using the torch-backed Voxelmorph stack."""

from __future__ import annotations

import argparse
import voxelmorph as vxm

from . import _torch_utils as cli


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--moving', required=True, help='moving image filename')
    parser.add_argument('--warp', required=True, help='warp image filename')
    parser.add_argument('--moved', required=True, help='warped image output filename')
    parser.add_argument('--interp', default='linear', choices=['linear', 'nearest'], help='interpolation method (default: linear)')
    parser.add_argument('--gpu', help='GPU number - if not supplied, CPU is used')
    parser.add_argument('--multichannel', action='store_true', help='specify that data has multiple channels')
    return parser.parse_args()


def main():
    args = parse_args()
    cli.setup_device(args.gpu)

    add_feat_axis = not args.multichannel
    moving = vxm.py.utils.load_volfile(args.moving, add_batch_axis=True, add_feat_axis=add_feat_axis)
    deform, deform_affine = vxm.py.utils.load_volfile(args.warp, add_batch_axis=True, ret_affine=True)

    transformer = vxm.networks.Transform(
        inshape=moving.shape[1:-1],
        interp_method=args.interp,
        nb_feats=moving.shape[-1],
    )

    moved = transformer.predict([moving, deform])
    vxm.py.utils.save_volfile(moved.squeeze(), args.moved, deform_affine)


if __name__ == '__main__':
    main()

