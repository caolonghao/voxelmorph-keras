#!/usr/bin/env python

"""Train a 3D atlas with segmentation supervision using the torch-backed Voxelmorph stack."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import neurite as ne
import numpy as np
import voxelmorph as vxm

from keras import Model, callbacks, layers as KL, optimizers

from . import _torch_utils as cli


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-csv', required=True, help='CSV with columns "image" and "seg" listing training pairs')
    parser.add_argument('--labels', required=True, help='npy file containing integer label values used for Dice supervision')
    parser.add_argument('--img-np-var', default='vol', help='npz variable name for images (default: vol)')
    parser.add_argument('--seg-np-var', default='seg', help='npz variable name for segmentations (default: seg)')
    parser.add_argument('--init-template', help='optional initial intensity template volume')
    parser.add_argument('--init-template-seg', help='optional initial atlas segmentation (probabilities)')
    parser.add_argument('--model-dir', default='models', help='output directory (default: models)')
    parser.add_argument('--multichannel', action='store_true', help='set when inputs already contain a feature axis')

    parser.add_argument('--gpu', help='GPU id(s); omit for CPU')
    parser.add_argument('--batch-size', type=int, default=1, help='training batch size (default: 1)')
    parser.add_argument('--epochs', type=int, default=1500, help='number of training epochs (default: 1500)')
    parser.add_argument('--steps-per-epoch', type=int, default=100, help='steps per epoch (default: 100)')
    parser.add_argument('--load-weights', help='optional weights file to resume from')
    parser.add_argument('--initial-epoch', type=int, default=0, help='initial epoch when resuming (default: 0)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')

    parser.add_argument('--enc', type=int, nargs='+', help='U-Net encoder filters (default: 16 32 32 32)')
    parser.add_argument('--dec', type=int, nargs='+', help='U-Net decoder filters (default: 32 32 32 32 32 16 16)')

    parser.add_argument('--image-loss', default='ncc', choices=['ncc', 'mse'], help='image reconstruction loss (default: ncc)')
    parser.add_argument('--image-loss-weight', type=float, default=1.0, help='weight for image reconstruction loss (default: 1.0)')
    parser.add_argument('--mean-loss-weight', type=float, default=1.0, help='weight encouraging atlas towards data mean (default: 1.0)')
    parser.add_argument('--grad-loss-weight', type=float, default=1.0, help='weight for deformation smoothness loss (default: 1.0)')
    parser.add_argument('--dice-loss-weight', type=float, default=1.0, help='weight for Dice loss on warped segmentations (default: 1.0)')
    parser.add_argument('--atlas-dice-loss-weight', type=float, default=1.0, help='weight for Dice loss on atlas segmentation prior (default: 1.0)')

    return parser


def read_supervised_csv(csv_path: str) -> List[Tuple[str, str]]:
    path = Path(csv_path)
    with path.open(newline='') as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = {name.lower() for name in reader.fieldnames or []}
        if 'image' not in fieldnames or 'seg' not in fieldnames:
            raise ValueError('CSV must contain headers "image" and "seg".')

        pairs: List[Tuple[str, str]] = []
        for row in reader:
            image = (row.get('image') or row.get('IMAGE') or '').strip()
            seg = (row.get('seg') or row.get('SEG') or '').strip()
            if not image or not seg:
                continue
            image_path = Path(image)
            seg_path = Path(seg)
            if not image_path.is_absolute():
                image_path = (path.parent / image_path).resolve()
            if not seg_path.is_absolute():
                seg_path = (path.parent / seg_path).resolve()
            pairs.append((str(image_path), str(seg_path)))

    if not pairs:
        raise ValueError('No valid image/segmentation pairs were found in the CSV file.')

    return pairs


def is_npz_file(path: str) -> bool:
    return Path(path).suffix == '.npz'


def load_volume(path: str, add_feat_axis: bool, np_var: str | None) -> np.ndarray:
    kwargs = dict(add_batch_axis=True, add_feat_axis=add_feat_axis)
    if np_var and is_npz_file(path):
        kwargs['np_var'] = np_var
    return vxm.py.utils.load_volfile(path, **kwargs)


def load_segmentation(path: str, labels: np.ndarray, np_var: str | None) -> np.ndarray:
    seg_vol = load_volume(path, add_feat_axis=True, np_var=np_var)
    seg_data = np.squeeze(seg_vol, axis=-1)
    seg_data = np.rint(seg_data).astype('int32')
    one_hot = np.stack([(seg_data == label).astype('float32') for label in labels], axis=-1)
    return one_hot.astype('float32')


def average_samples(samples: Iterable[np.ndarray]) -> np.ndarray:
    samples = list(samples)
    if not samples:
        raise ValueError('Cannot average an empty iterable.')
    avg = np.zeros_like(samples[0], dtype='float32')
    for sample in samples:
        avg += sample.astype('float32')
    avg /= float(len(samples))
    return avg


def compute_initial_templates(
    args,
    pairs: Sequence[Tuple[str, str]],
    labels: np.ndarray,
    add_feat_axis: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    if args.init_template:
        template_img = load_volume(args.init_template, add_feat_axis, args.img_np_var).astype('float32')
    else:
        count = min(100, len(pairs))
        imgs = [load_volume(pairs[i][0], add_feat_axis, args.img_np_var) for i in range(count)]
        template_img = average_samples(imgs)

    if args.init_template_seg:
        template_seg = load_segmentation(args.init_template_seg, labels, args.seg_np_var).astype('float32')
    else:
        count = min(100, len(pairs))
        segs = [load_segmentation(pairs[i][1], labels, args.seg_np_var) for i in range(count)]
        template_seg = average_samples(segs)

    return template_img, template_seg


def make_generator(
    args,
    pairs: Sequence[Tuple[str, str]],
    labels: np.ndarray,
    add_feat_axis: bool,
    atlas_seg_prior: np.ndarray,
):
    nb_dims = atlas_seg_prior.shape[1:-1]

    def generator():
        zeros = np.zeros((args.batch_size, *nb_dims, len(nb_dims)), dtype='float32')
        atlas_targets = np.repeat(atlas_seg_prior, args.batch_size, axis=0).astype('float32')
        while True:
            indices = np.random.randint(len(pairs), size=args.batch_size)
            imgs = []
            segs = []
            for idx in indices:
                img_path, seg_path = pairs[idx]
                img = load_volume(img_path, add_feat_axis, args.img_np_var)
                seg = load_segmentation(seg_path, labels, args.seg_np_var)
                imgs.append(img)
                segs.append(seg)

            batch_imgs = np.concatenate(imgs, axis=0).astype('float32')
            batch_segs = np.concatenate(segs, axis=0).astype('float32')

            yield [batch_imgs, batch_segs], [batch_imgs, batch_imgs, zeros, atlas_targets, atlas_targets]

    return generator


def build_model(
    args,
    inshape: Sequence[int],
    nfeats: int,
    nb_labels: int,
) -> Model:
    enc_nf = args.enc if args.enc else [16, 32, 32, 32]
    dec_nf = args.dec if args.dec else [32, 32, 32, 32, 32, 16, 16]

    template = vxm.networks.TemplateCreation(
        inshape=inshape,
        nb_unet_features=[enc_nf, dec_nf],
        src_feats=nfeats,
        atlas_feats=nfeats,
    )

    image_input = template.inputs[0]
    warped_image, atlas_image, flow = template.outputs

    seg_input = KL.Input(shape=(*inshape, nb_labels), name='template_creation_seg_input')
    atlas_seg_layer = ne.layers.LocalParamWithInput(
        shape=(*inshape, nb_labels),
        initializer='zeros',
        name=f'{template.name}_atlas_seg_param',
    )
    atlas_seg = atlas_seg_layer(image_input)

    warp_layer = vxm.networks.Transform(inshape, nb_feats=nb_labels, interp_method='nearest')
    warped_seg = warp_layer([seg_input, flow])

    model = Model(
        inputs=[image_input, seg_input],
        outputs=[warped_image, atlas_image, flow, warped_seg, atlas_seg],
        name='template_creation_supervised',
    )

    return model


def compile_model(model: Model, args) -> None:
    if args.image_loss == 'ncc':
        image_loss = vxm.losses.NCC().loss
    else:
        image_loss = vxm.losses.MSE().loss

    mean_loss = vxm.losses.MSE().loss
    grad_loss = vxm.losses.Grad('l2').loss
    dice_loss = vxm.losses.Dice().loss

    model.compile(
        optimizer=optimizers.Adam(learning_rate=args.lr),
        loss=[image_loss, mean_loss, grad_loss, dice_loss, dice_loss],
        loss_weights=[
            args.image_loss_weight,
            args.mean_loss_weight,
            args.grad_loss_weight,
            args.dice_loss_weight,
            args.atlas_dice_loss_weight,
        ],
    )


def set_initial_parameters(model: Model, atlas_img: np.ndarray, atlas_seg: np.ndarray) -> None:
    atlas_layer = model.get_layer('template_creation_atlas_param')
    atlas_layer.set_weights([atlas_img.squeeze(axis=0).astype('float32')])

    atlas_seg_layer = model.get_layer('template_creation_atlas_seg_param')
    atlas_seg_layer.set_weights([atlas_seg.squeeze(axis=0).astype('float32')])


def extract_atlas_parameters(model: Model) -> Tuple[np.ndarray, np.ndarray]:
    atlas_img = model.get_layer('template_creation_atlas_param').get_weights()[0]
    atlas_seg = model.get_layer('template_creation_atlas_seg_param').get_weights()[0]
    return atlas_img, atlas_seg


def main():
    args = build_parser().parse_args()
    cli.setup_device(args.gpu)

    pairs = read_supervised_csv(args.data_csv)
    labels = np.load(args.labels)
    if labels.ndim != 1:
        raise ValueError('labels.npy must contain a 1D array of label values.')
    labels = labels.astype('int32')

    cli.ensure_directory(args.model_dir)

    add_feat_axis = not args.multichannel
    template_img, template_seg = compute_initial_templates(args, pairs, labels, add_feat_axis)

    vxm.py.utils.save_volfile(template_img.squeeze(), os.path.join(args.model_dir, 'init_template.nii.gz'))
    vxm.py.utils.save_volfile(template_seg.squeeze(), os.path.join(args.model_dir, 'init_template_seg_prob.npz'))

    inshape = template_img.shape[1:-1]
    nfeats = template_img.shape[-1]
    nb_labels = labels.size

    generator_factory = make_generator(args, pairs, labels, add_feat_axis, template_seg)

    model = build_model(args, inshape, nfeats, nb_labels)
    set_initial_parameters(model, template_img, template_seg)

    if args.load_weights:
        model.load_weights(args.load_weights)

    compile_model(model, args)

    checkpoint_pattern = os.path.join(args.model_dir, 'weights_epoch_{epoch:04d}.weights.h5')
    checkpoint_cb = callbacks.ModelCheckpoint(filepath=checkpoint_pattern, save_weights_only=True, save_freq='epoch')

    if args.initial_epoch == 0:
        cli.save_weights(model, checkpoint_pattern.format(epoch=0))

    generator = generator_factory()

    model.fit(
        generator,
        initial_epoch=args.initial_epoch,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        callbacks=[checkpoint_cb],
        verbose=1,
    )

    atlas_img, atlas_seg = extract_atlas_parameters(model)
    vxm.py.utils.save_volfile(atlas_img, os.path.join(args.model_dir, 'template.nii.gz'))
    vxm.py.utils.save_volfile(atlas_seg, os.path.join(args.model_dir, 'template_seg_prob.npz'))


if __name__ == '__main__':
    main()
