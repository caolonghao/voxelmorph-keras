#!/usr/bin/env python

"""Evaluate unsupervised probabilistic atlas segmentation using a torch-backed model."""

from __future__ import annotations

import argparse
import numpy as np
import voxelmorph as vxm

from . import _torch_utils as cli


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('image', help='input image to test')
    parser.add_argument('seg', help='output segmentation file')
    parser.add_argument('--model', required=True, help='trained probabilistic atlas segmentation model')
    parser.add_argument('--atlas', required=True, help='atlas npz file')
    parser.add_argument('--mapping', required=True, help='atlas mapping filename (npz with mapping array)')
    parser.add_argument('--gpu', help='GPU number - if not supplied, CPU is used')
    parser.add_argument('--max-feats', type=int, default=32, help='number of label posteriors to process at once')
    parser.add_argument('--warped-atlas', help='save warped atlas to output volume file')
    parser.add_argument('--posteriors', help='save posteriors to output volume file')
    parser.add_argument('--warp', help='save warp field to output volume file')
    parser.add_argument('--stats', help='save Gaussian stats to output npz file')
    return parser.parse_args()


def load_atlas(atlas_path: str, mapping_path: str):
    atlas_full = vxm.py.utils.load_volfile(atlas_path, add_batch_axis=True)
    mapping = np.load(mapping_path)['mapping'].astype('int').flatten()
    if len(mapping) != atlas_full.shape[-1]:
        raise ValueError('Mapping shape inconsistent with atlas channels.')

    nb_labels = int(mapping.max() + 1)
    atlas = np.zeros([*atlas_full.shape[:-1], nb_labels])
    for idx, lbl in enumerate(mapping):
        atlas[0, ..., lbl] += atlas_full[0, ..., idx]

    return atlas, atlas_full, mapping


def warp_chunk(atlas_chunk, flow):
    transformer = vxm.networks.Transform(flow.shape[1:-1], interp_method='linear', nb_feats=atlas_chunk.shape[-1])
    warped = transformer.predict([atlas_chunk, flow])
    return warped.squeeze()


def main():
    args = parse_args()
    cli.setup_device(args.gpu)

    atlas, atlas_full, mapping = load_atlas(args.atlas, args.mapping)
    image, affine = vxm.py.utils.load_volfile(args.image, add_batch_axis=True, add_feat_axis=True, ret_affine=True)

    model = vxm.networks.ProbAtlasSegmentation.load(args.model).get_gaussian_warp_model()
    ull_pred, mus, sigmas, flow = model.predict([image, atlas])
    ull_pred = ull_pred[0]
    mus = mus[0]
    sigmas = sigmas[0]
    flow = flow[0:1]

    posteriors = []
    warped_atlas_chunks = []
    total_labels = atlas_full.shape[-1]

    for start in range(0, total_labels, args.max_feats):
        end = min(start + args.max_feats, total_labels)
        atlas_chunk = atlas_full[:, ..., start:end]
        warped_chunk = warp_chunk(atlas_chunk, flow)
        warped_atlas_chunks.append(warped_chunk)

        this_mapping = mapping[start:end]
        post_chunk = np.empty_like(warped_chunk)
        for idx, label in enumerate(this_mapping):
            post_chunk[..., idx] = np.exp(ull_pred[..., label]) * warped_chunk[..., idx]
        posteriors.append(post_chunk)

    posteriors = np.concatenate(posteriors, axis=-1)
    warped_atlas_full = np.concatenate(warped_atlas_chunks, axis=-1)
    segmentation = posteriors.argmax(-1)

    vxm.py.utils.save_volfile(segmentation, args.seg, affine)

    if args.warp:
        vxm.py.utils.save_volfile(flow.squeeze(), args.warp, affine)
    if args.warped_atlas:
        vxm.py.utils.save_volfile(warped_atlas_full, args.warped_atlas, affine)
    if args.posteriors:
        vxm.py.utils.save_volfile(posteriors, args.posteriors, affine)
    if args.stats:
        np.savez(args.stats, mu=mus, sigma=sigmas)


if __name__ == '__main__':
    main()

