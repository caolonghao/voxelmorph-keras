#!/usr/bin/env python

"""Evaluate a trained Voxelmorph model by computing Dice scores on image pairs."""

from __future__ import annotations

import argparse
import time
import numpy as np
import voxelmorph as vxm

from . import _torch_utils as cli


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='GPU number - if not supplied, CPU is used')
    parser.add_argument('--model', required=True, help='trained Voxelmorph model file')
    parser.add_argument('--pairs', required=True, help='path to list of image pairs to register')
    parser.add_argument('--img-suffix', help='input image file suffix')
    parser.add_argument('--seg-suffix', help='input segmentation file suffix')
    parser.add_argument('--img-prefix', help='input image file prefix')
    parser.add_argument('--seg-prefix', help='input segmentation file prefix')
    parser.add_argument('--labels', help='optional label list to compute dice for (npy format)')
    parser.add_argument('--multichannel', action='store_true', help='specify that data has multiple channels')
    return parser.parse_args()


def main():
    args = parse_args()
    cli.setup_device(args.gpu)

    if args.img_prefix == args.seg_prefix and args.img_suffix == args.seg_suffix:
        raise ValueError('Image and segmentation suffix/prefix must differ.')

    img_pairs = vxm.py.utils.read_pair_list(args.pairs, prefix=args.img_prefix, suffix=args.img_suffix)
    seg_pairs = vxm.py.utils.read_pair_list(args.pairs, prefix=args.seg_prefix, suffix=args.seg_suffix)

    labels = np.load(args.labels) if args.labels else None
    add_feat_axis = not args.multichannel

    model = vxm.networks.VxmDense.load(args.model, input_model=None)
    registration_model = model.get_registration_model()
    inshape = registration_model.inputs[0].shape[1:-1]
    transform_model = vxm.networks.Transform(inshape, interp_method='nearest')

    reg_times = []
    dice_scores = []

    for idx, (img_pair, seg_pair) in enumerate(zip(img_pairs, seg_pairs)):
        moving_vol = vxm.py.utils.load_volfile(img_pair[0], np_var='vol', add_batch_axis=True, add_feat_axis=add_feat_axis)
        moving_seg = vxm.py.utils.load_volfile(seg_pair[0], np_var='seg', add_batch_axis=True, add_feat_axis=add_feat_axis)
        fixed_vol = vxm.py.utils.load_volfile(img_pair[1], np_var='vol', add_batch_axis=True, add_feat_axis=add_feat_axis)
        fixed_seg = vxm.py.utils.load_volfile(seg_pair[1], np_var='seg')

        start = time.time()
        warp = registration_model.predict([moving_vol, fixed_vol])
        duration = time.time() - start
        if idx != 0:
            reg_times.append(duration)

        warped_seg = transform_model.predict([moving_seg, warp]).squeeze()
        overlap = vxm.py.utils.dice(warped_seg, fixed_seg, labels=labels)
        dice_scores.append(np.mean(overlap))

        print(f'Pair {idx + 1:03d}    Reg Time: {duration:.4f}    Dice: {np.mean(overlap):.4f} +/- {np.std(overlap):.4f}')

    if reg_times:
        print('\nAvg Reg Time: %.4f +/- %.4f (skipping first prediction)' % (np.mean(reg_times), np.std(reg_times)))
    print('Avg Dice: %.4f +/- %.4f' % (np.mean(dice_scores), np.std(dice_scores)))


if __name__ == '__main__':
    main()

