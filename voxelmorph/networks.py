"""Torch-backed Keras network builders for Voxelmorph.

This module reimplements the core VoxelMorph registration network using the
multi-backend Keras API with the PyTorch backend. The original TensorFlow-based
implementation exposed a large catalogue of specialised models; the current
port focuses on the dense registration model (`VxmDense`), together with
utility components such as the U-Net backbone and the generic `Transform`
wrapper.  Additional architectures will be reintroduced incrementally – for
now, attempting to instantiate one of the unported classes raises a clear
``NotImplementedError``.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import math
import warnings

import numpy as np

from keras import Input, Model, ops
from keras import layers as KL
from keras import initializers as KI

import neurite as ne

from . import layers
from .py.utils import default_unet_features

# Public callback export retained for backwards compatibility.
ModelCheckpointParallel = ne.callbacks.ModelCheckpointParallel


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------


def _ensure_iterable(value, length: Optional[int] = None) -> Tuple:
    """Return ``value`` as a tuple, optionally verifying its length."""

    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        result = tuple(value)
    else:
        result = (value,) if value is not None else tuple()
    if length is not None and result and len(result) != length:
        raise ValueError(f'Expected iterable of length {length}, got {len(result)}.')
    return result


def _conv_layer(ndims: int):
    return {1: KL.Conv1D, 2: KL.Conv2D, 3: KL.Conv3D}[ndims]


def _pool_layer(ndims: int):
    return {1: KL.MaxPool1D, 2: KL.MaxPool2D, 3: KL.MaxPool3D}[ndims]


def _upsample_layer(ndims: int):
    return {1: KL.UpSampling1D, 2: KL.UpSampling2D, 3: KL.UpSampling3D}[ndims]


def _infer_ndims(tensor) -> int:
    rank = len(tensor.shape)
    if rank is None:
        raise ValueError('Tensor rank must be statically known to build the network.')
    return rank - 1


def _compute_unet_features(nb_features, nb_levels, feat_mult) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Return encoder/decoder feature counts for the U-Net backbone."""

    if nb_features is None:
        enc_default, _ = default_unet_features()
        enc = tuple(int(f) for f in enc_default)
    elif isinstance(nb_features, Iterable) and not isinstance(nb_features, (int, float)):
        nb_features = list(nb_features)
        if len(nb_features) == 2 and all(isinstance(v, Iterable) for v in nb_features):
            enc = tuple(int(f) for f in nb_features[0])
        else:
            raise ValueError('nb_unet_features must be an int or [encoder, decoder] iterables.')
    else:
        if nb_levels is None:
            raise ValueError('nb_unet_levels must be provided when nb_unet_features is an integer.')
        enc = tuple(int(nb_features * (feat_mult ** i)) for i in range(nb_levels))

    if len(enc) < 2:
        raise ValueError('U-Net requires at least two encoder levels.')

    # Symmetric decoder mirroring the encoder (excluding the bottleneck level).
    dec = enc[:-1][::-1]
    return enc, dec


def _pad_or_trim(filters: Sequence[int], target: int) -> Tuple[int, ...]:
    """Return ``filters`` sized to ``target`` elements."""

    filters = list(filters)
    if not filters:
        raise ValueError('Filter configuration must be non-empty.')
    if len(filters) >= target:
        return tuple(filters[:target])
    last = filters[-1]
    filters.extend([last] * (target - len(filters)))
    return tuple(filters)


def _unet_body(
    input_tensor,
    nb_encoder: Sequence[int],
    nb_decoder: Sequence[int],
    nb_conv_per_level: int,
    kernel_initializer: str,
    pool_size: int,
    name: str,
):
    """Build the convolutional U-Net body returning the final feature map."""

    ndims = _infer_ndims(input_tensor)
    Conv = _conv_layer(ndims)
    Pool = _pool_layer(ndims)
    Up = _upsample_layer(ndims)

    x = input_tensor
    skips = []

    # Encoder pathway.
    for level, filters in enumerate(nb_encoder):
        for conv_idx in range(nb_conv_per_level):
            x = Conv(
                filters,
                kernel_size=3,
                padding='same',
                kernel_initializer=kernel_initializer,
                name=f'{name}_enc_{level}_{conv_idx}',
            )(x)
            x = KL.LeakyReLU(alpha=0.2, name=f'{name}_enc_{level}_{conv_idx}_act')(x)
        if level < len(nb_encoder) - 1:
            skips.append(x)
            x = Pool(pool_size=pool_size, name=f'{name}_pool_{level}')(x)

    # Decoder pathway.
    nb_decoder = _pad_or_trim(nb_decoder, len(skips))
    for level, filters in enumerate(nb_decoder):
        x = Up(size=pool_size, name=f'{name}_up_{level}')(x)
        skip = skips[-(level + 1)]
        x = KL.Concatenate(name=f'{name}_concat_{level}')([x, skip])
        for conv_idx in range(nb_conv_per_level):
            x = Conv(
                filters,
                kernel_size=3,
                padding='same',
                kernel_initializer=kernel_initializer,
                name=f'{name}_dec_{level}_{conv_idx}',
            )(x)
            x = KL.LeakyReLU(alpha=0.2, name=f'{name}_dec_{level}_{conv_idx}_act')(x)

    return x


def _rescale_if_needed(tensor, factor: float, name: str):
    if abs(factor - 1.0) < 1e-6:
        return tensor
    return layers.RescaleTransform(factor, name=name)(tensor)


def _not_yet_ported(name: str):
    class _Placeholder(ne.modelio.LoadableModel):
        def __init__(self, *args, **kwargs):  # pragma: no cover - explicit placeholder
            raise NotImplementedError(
                f'{name} is not yet ported to the torch backend. '
                f'Please open an issue if you rely on this architecture.'
            )

    _Placeholder.__name__ = name
    return _Placeholder


# -----------------------------------------------------------------------------
# Utility/Core Networks
# -----------------------------------------------------------------------------


class Transform(Model):
    """Apply a dense or affine transform using the torch-backed spatial transformer."""

    def __init__(
        self,
        inshape: Sequence[int],
        affine: bool = False,
        interp_method: str = 'linear',
        rescale: Optional[float] = None,
        fill_value: Optional[float] = None,
        nb_feats: int = 1,
        name: str = 'transform',
    ) -> None:
        ndims = len(inshape)
        scan_input = Input(shape=(*inshape, nb_feats), name='scan_input')

        if affine:
            trf_input = Input(shape=(ndims, ndims + 1), name='trf_input')
        else:
            if rescale is None:
                trf_shape = tuple(inshape)
            else:
                trf_shape = tuple(int(math.ceil(dim / rescale)) for dim in inshape)
            trf_input = Input(shape=(*trf_shape, ndims), name='trf_input')

        trf_scaled = _rescale_if_needed(trf_input, rescale or 1.0, name=f'{name}_rescale')
        y_source = layers.SpatialTransformer(
            interp_method=interp_method,
            fill_value=fill_value,
            name=f'{name}_transformer',
        )([scan_input, trf_scaled])

        super().__init__(inputs=[scan_input, trf_input], outputs=y_source, name=name)


class Unet(Model):
    """Lightweight torch-ready U-Net backbone."""

    def __init__(
        self,
        inshape: Optional[Sequence[int]] = None,
        input_model: Optional[Model] = None,
        nb_features=None,
        nb_levels: Optional[int] = None,
        max_pool: int = 2,
        feat_mult: int = 1,
        nb_conv_per_level: int = 1,
        do_res: bool = False,
        nb_upsample_skips: int = 0,
        hyp_input=None,
        hyp_tensor=None,
        final_activation_function: Optional[str] = None,
        kernel_initializer: str = 'he_normal',
        name: str = 'unet',
    ) -> None:
        if do_res:
            raise NotImplementedError('Residual connections are not yet ported to the torch backend.')
        if nb_upsample_skips:
            warnings.warn('nb_upsample_skips is ignored in the current torch-backed implementation.')
        if hyp_input is not None or hyp_tensor is not None:
            raise NotImplementedError('Hyper-network conditioning is not yet supported.')

        if input_model is None:
            if inshape is None:
                raise ValueError('Either inshape or input_model must be provided.')
            primary_input = Input(shape=inshape, name=f'{name}_input')
            inputs = [primary_input]
            tensor = primary_input
        else:
            inputs = list(input_model.inputs)
            outputs = input_model.outputs
            if len(outputs) != 1:
                raise ValueError('input_model must expose a single output tensor for the U-Net body.')
            tensor = outputs[0]

        enc_feats, dec_feats = _compute_unet_features(nb_features, nb_levels, feat_mult)
        body = _unet_body(
            tensor,
            nb_encoder=enc_feats,
            nb_decoder=dec_feats,
            nb_conv_per_level=nb_conv_per_level,
            kernel_initializer=kernel_initializer,
            pool_size=max_pool,
            name=name,
        )

        if final_activation_function:
            body = KL.Activation(final_activation_function, name=f'{name}_final_activation')(body)

        super().__init__(inputs=inputs, outputs=body, name=name)


class VxmDense(ne.modelio.LoadableModel):
    """VoxelMorph network for (unsupervised) nonlinear registration between two images."""

    @ne.modelio.store_config_args
    def __init__(
        self,
        inshape: Sequence[int],
        nb_unet_features=None,
        nb_unet_levels: Optional[int] = None,
        unet_feat_mult: int = 1,
        nb_unet_conv_per_level: int = 1,
        int_steps: int = 7,
        svf_resolution: float = 1.0,
        int_resolution: float = 2.0,
        int_downsize=None,
        bidir: bool = False,
        use_probs: bool = False,
        src_feats: int = 1,
        trg_feats: int = 1,
        unet_half_res: bool = False,
        input_model: Optional[Model] = None,
        hyp_model: Optional[Model] = None,
        fill_value: Optional[float] = None,
        reg_field: str = 'preintegrated',
        name: str = 'vxm_dense',
    ) -> None:
        ndims = len(inshape)
        if ndims not in (1, 2, 3):
            raise ValueError(f'Unsupported dimensionality {ndims}; expected 1, 2, or 3.')
        if unet_half_res:
            warnings.warn('unet_half_res is ignored in the torch-backed port; use svf_resolution instead.')
        if int_downsize is not None:
            warnings.warn('int_downsize is deprecated, use int_resolution instead.')
            int_resolution = int_downsize

        # Configure inputs.
        if input_model is None:
            source = Input(shape=(*inshape, src_feats), name=f'{name}_source_input')
            target = Input(shape=(*inshape, trg_feats), name=f'{name}_target_input')
            inputs = [source, target]
        else:
            inputs = list(input_model.inputs)
            outputs = input_model.outputs
            if len(outputs) < 2:
                raise ValueError('input_model must expose at least two outputs (source, target).')
            source, target = outputs[:2]

        hyp_tensor = None
        if hyp_model is not None:
            hyp_inputs = hyp_model.inputs if isinstance(hyp_model.inputs, (list, tuple)) else [hyp_model.input]
            model_args = hyp_inputs if len(hyp_inputs) > 1 else hyp_inputs[0]
            hyp_tensor = hyp_model(model_args)
            for hyp_in in hyp_inputs:
                if hyp_in not in inputs:
                    inputs.append(hyp_in)

        # U-Net backbone operating on concatenated source/target volumes.
        concat = KL.Concatenate(name=f'{name}_concat')([source, target])
        unet_input_model = Model(inputs=inputs, outputs=concat, name=f'{name}_unet_input')
        unet_model = Unet(
            input_model=unet_input_model,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level,
            name=f'{name}_unet',
        )

        unet_output = unet_model.output
        Conv = _conv_layer(ndims)

        weaknorm = KI.RandomNormal(mean=0.0, stddev=1e-5)
        flow_mean = Conv(
            ndims,
            kernel_size=3,
            padding='same',
            kernel_initializer=weaknorm,
            name=f'{name}_flow',
        )(unet_output)

        if use_probs:
            logsigma_init = KI.RandomNormal(mean=0.0, stddev=1e-10)
            flow_logsigma = Conv(
                ndims,
                kernel_size=3,
                padding='same',
                kernel_initializer=logsigma_init,
                bias_initializer=KI.Constant(value=-10),
                name=f'{name}_log_sigma',
            )(unet_output)
            flow_params = KL.Concatenate(name=f'{name}_prob_concat')([flow_mean, flow_logsigma])
            flow = ne.layers.SampleNormalLogVar(name=f'{name}_z_sample')([flow_mean, flow_logsigma])
        else:
            flow_params = None
            flow = flow_mean

        if hyp_tensor is not None:
            scale = KL.Dense(ndims, activation='tanh', name=f'{name}_hyp_scale')(hyp_tensor)

            def _expand_dims(x):
                for _ in range(ndims):
                    x = ops.expand_dims(x, axis=1)
                return x

            scale = KL.Lambda(_expand_dims, name=f'{name}_hyp_scale_expand')(scale)
            scale = KL.Lambda(lambda x: x + 1.0, name=f'{name}_hyp_scale_offset')(scale)
            flow = KL.Multiply(name=f'{name}_hyp_scale_mul')([flow, scale])

        current_resolution = 1.0
        svf_resolution = float(svf_resolution)
        int_resolution = float(int_resolution)

        flow = _rescale_if_needed(flow, 1.0 / svf_resolution, name=f'{name}_svf_resize')
        current_resolution /= svf_resolution
        svf = flow

        if int_steps > 0 and int_resolution > 0 and int_resolution != svf_resolution:
            target_resolution = 1.0 / int_resolution
            factor = target_resolution / current_resolution
            flow = _rescale_if_needed(flow, factor, name=f'{name}_flow_resize')
            current_resolution = target_resolution

        preint_flow = flow

        pos_flow = flow
        neg_flow = None
        if bidir:
            neg_flow = ne.layers.Negate(name=f'{name}_neg_flow')(flow)

        if int_steps > 0:
            pos_flow = layers.VecInt(method='ss', name=f'{name}_flow_int', int_steps=int_steps)(pos_flow)
            if bidir:
                neg_flow = layers.VecInt(method='ss', name=f'{name}_neg_flow_int', int_steps=int_steps)(neg_flow)
        postint_flow = pos_flow

        # Restore full resolution for warping.
        if current_resolution != 1.0:
            factor_back = 1.0 / current_resolution
            pos_flow = _rescale_if_needed(pos_flow, factor_back, name=f'{name}_flow_full_res')
            if bidir and neg_flow is not None:
                neg_flow = _rescale_if_needed(neg_flow, factor_back, name=f'{name}_neg_flow_full_res')

        y_source = layers.SpatialTransformer(
            interp_method='linear',
            fill_value=fill_value,
            name=f'{name}_transformer',
        )([source, pos_flow])

        outputs = [y_source]
        if bidir and neg_flow is not None:
            y_target = layers.SpatialTransformer(
                interp_method='linear',
                fill_value=fill_value,
                name=f'{name}_neg_transformer',
            )([target, neg_flow])
            outputs.append(y_target)
        else:
            y_target = None

        reg_field = reg_field.lower()
        if use_probs and flow_params is not None:
            outputs.append(flow_params)
        elif reg_field == 'svf':
            outputs.append(svf)
        elif reg_field == 'preintegrated':
            outputs.append(preint_flow)
        elif reg_field == 'postintegrated':
            outputs.append(postint_flow)
        elif reg_field == 'warp':
            outputs.append(pos_flow)
        else:
            raise ValueError(f'Unknown option "{reg_field}" for reg_field.')

        outputs = tuple(KL.Activation('linear', name=f'{name}_out_{idx}')(tensor) for idx, tensor in enumerate(outputs))

        super().__init__(inputs=inputs, outputs=outputs, name=name)

        # Cache references used by higher-level utilities.
        self.references = ne.modelio.LoadableModel.ReferenceContainer()
        self.references.unet_model = unet_model
        self.references.source = source
        self.references.target = target
        self.references.svf = svf
        self.references.preint_flow = preint_flow
        self.references.postint_flow = postint_flow
        self.references.pos_flow = pos_flow
        self.references.neg_flow = neg_flow
        self.references.y_source = y_source
        self.references.y_target = y_target
        self.references.flow_params = flow_params
        self.references.hyp_tensor = hyp_tensor
        self.references.unet_features = unet_output

    def get_registration_model(self) -> Model:
        """Return a model that predicts the forward warp only."""

        return Model(self.inputs, self.references.pos_flow, name=f'{self.name}_warp')

    def register(self, src, trg):
        """Predict the forward transform that maps ``src`` to ``trg`` volumes."""

        return self.get_registration_model().predict([src, trg])

    def apply_transform(self, src, trg, img, interp_method: str = 'linear', fill_value=None):
        """Predict the warp between ``src`` and ``trg`` and apply it to ``img``."""

        warp_model = self.get_registration_model()
        img_input = Input(shape=img.shape[1:], name=f'{self.name}_apply_image')
        warped = layers.SpatialTransformer(
            interp_method=interp_method,
            fill_value=fill_value,
            name=f'{self.name}_apply_transformer',
        )([img_input, warp_model.output])
        model = Model(warp_model.inputs + [img_input], warped, name=f'{self.name}_apply')
        return model.predict([src, trg, img])


# -----------------------------------------------------------------------------
# Semi-supervised segmentation variant
# -----------------------------------------------------------------------------


class VxmDenseSemiSupervisedSeg(ne.modelio.LoadableModel):
    """VoxelMorph network with auxiliary segmentation supervision."""

    @ne.modelio.store_config_args
    def __init__(
        self,
        inshape: Sequence[int],
        nb_labels: int,
        nb_unet_features=None,
        seg_resolution: float = 1.0,
        seg_downsize=None,
        bidir: bool = False,
        bidir_labels: bool = False,
        name: str = 'vxm_dense_seg',
        **kwargs,
    ) -> None:
        if seg_downsize is not None:
            warnings.warn('seg_downsize is deprecated, use seg_resolution instead.')
            seg_resolution = seg_downsize
        if abs(seg_resolution - 1.0) > 1e-6:
            warnings.warn('seg_resolution other than 1.0 is not yet supported in the torch port; '
                          'segmentations are currently warped at full resolution.')

        if bidir_labels and not bidir:
            warnings.warn('bidir_labels requested without bidir=True; enabling bidir warps.')
            bidir = True

        src_feats = kwargs.pop('src_feats', 1)
        trg_feats = kwargs.pop('trg_feats', 1)

        source = Input(shape=(*inshape, src_feats), name=f'{name}_source_input')
        target = Input(shape=(*inshape, trg_feats), name=f'{name}_target_input')
        seg_source = Input(shape=(*inshape, nb_labels), name=f'{name}_source_seg')
        seg_target = None
        if bidir_labels:
            seg_target = Input(shape=(*inshape, nb_labels), name=f'{name}_target_seg')

        base_input_model = Model(inputs=[source, target], outputs=[source, target], name=f'{name}_inputs')
        vxm_model = VxmDense(
            inshape,
            nb_unet_features=nb_unet_features,
            src_feats=src_feats,
            trg_feats=trg_feats,
            bidir=bidir,
            input_model=base_input_model,
            name=f'{name}_core',
            **kwargs,
        )

        core_outputs = list(vxm_model.outputs)
        reg_output = core_outputs.pop(-1)
        if bidir:
            y_target = core_outputs.pop(-1)
        else:
            y_target = None
        y_source = core_outputs[0]

        pos_flow = vxm_model.references.pos_flow
        neg_flow = vxm_model.references.neg_flow if bidir else None

        seg_warper = layers.SpatialTransformer(interp_method='nearest', name=f'{name}_seg_transformer')
        warped_seg_source = seg_warper([seg_source, pos_flow])
        warped_seg_target = None

        outputs = [y_source, warped_seg_source]

        if bidir:
            outputs.append(y_target)

        if bidir_labels and seg_target is not None:
            if neg_flow is None:
                neg_flow = ne.layers.Negate(name=f'{name}_neg_flow_labels')(pos_flow)
            warped_seg_target = seg_warper([seg_target, neg_flow])
            outputs.append(warped_seg_target)

        outputs.append(reg_output)

        outputs = tuple(
            KL.Activation('linear', name=f'{name}_out_{idx}')(tensor)
            for idx, tensor in enumerate(outputs)
        )

        inputs = [source, target, seg_source]
        if seg_target is not None:
            inputs.append(seg_target)

        super().__init__(inputs=inputs, outputs=outputs, name=name)

        self.references = vxm_model.references
        self.references.seg_source = seg_source
        self.references.seg_target = seg_target
        self.references.seg_warped_source = warped_seg_source
        if warped_seg_target is not None:
            self.references.seg_warped_target = warped_seg_target


# -----------------------------------------------------------------------------
# Instance-specific deformation field
# -----------------------------------------------------------------------------


class InstanceDense(ne.modelio.LoadableModel):
    """Learn a free-form warp field per training instance."""

    @ne.modelio.store_config_args
    def __init__(
        self,
        inshape: Sequence[int],
        nb_feats: int = 1,
        mult: float = 1.0,
        warp_shape: Optional[Sequence[int]] = None,
        name: str = 'instance_dense',
        fill_value: Optional[float] = None,
    ) -> None:
        ndims = len(inshape)
        source = Input(shape=(*inshape, nb_feats), name=f'{name}_source_input')

        if warp_shape is None:
            warp_shape = tuple(inshape)
        else:
            warp_shape = tuple(int(d) for d in warp_shape)

        flow_param = ne.layers.LocalParamWithInput(
            shape=(*warp_shape, ndims),
            mult=mult,
            name=f'{name}_local_param',
        )(source)

        if warp_shape != tuple(inshape):
            factor = inshape[0] / warp_shape[0]
            flow = layers.RescaleTransform(factor, name=f'{name}_flow_rescale')(flow_param)
        else:
            flow = flow_param

        warped = layers.SpatialTransformer(
            interp_method='linear',
            fill_value=fill_value,
            name=f'{name}_transformer',
        )([source, flow])

        outputs = (
            KL.Activation('linear', name=f'{name}_warped')(warped),
            KL.Activation('linear', name=f'{name}_flow')(flow),
        )

        super().__init__(inputs=[source], outputs=outputs, name=name)

        self.references = ne.modelio.LoadableModel.ReferenceContainer()
        self.references.source = source
        self.references.flow = flow
        self.references.warped = warped


# -----------------------------------------------------------------------------
# Point-cloud supervision
# -----------------------------------------------------------------------------


class VxmDenseSemiSupervisedPointCloud(ne.modelio.LoadableModel):
    """VoxelMorph model regularised by point-cloud correspondences."""

    @ne.modelio.store_config_args
    def __init__(
        self,
        inshape: Sequence[int],
        nb_surface_points: int,
        nb_unet_features=None,
        bidir: bool = False,
        surface_weight: float = 1.0,
        name: str = 'vxm_dense_pointcloud',
        **kwargs,
    ) -> None:
        ndims = len(inshape)
        if ndims not in (2, 3):
            raise ValueError('Point-cloud supervision currently supports 2D or 3D volumes.')

        source = Input(shape=(*inshape, 1), name=f'{name}_source_input')
        target = Input(shape=(*inshape, 1), name=f'{name}_target_input')
        surface_points = Input(shape=(nb_surface_points, ndims), name=f'{name}_surface_input')

        base_input_model = Model(inputs=[source, target], outputs=[source, target], name=f'{name}_inputs')
        vxm_model = VxmDense(
            inshape,
            nb_unet_features=nb_unet_features,
            bidir=bidir,
            src_feats=1,
            trg_feats=1,
            input_model=base_input_model,
            name=f'{name}_core',
            **kwargs,
        )

        core_outputs = list(vxm_model.outputs)
        reg_output = core_outputs.pop(-1)
        if bidir:
            y_target = core_outputs.pop(-1)
        else:
            y_target = None
        y_source = core_outputs[0]

        pos_flow = vxm_model.references.pos_flow

        warped_surface = KL.Lambda(
            lambda tensors: utils.point_spatial_transformer(tensors),
            name=f'{name}_surface_warp',
        )([surface_points, pos_flow])

        if surface_weight != 1.0:
            warped_surface = KL.Lambda(
                lambda x: x * surface_weight,
                name=f'{name}_surface_weight',
            )(warped_surface)

        outputs = [y_source]
        if bidir and y_target is not None:
            outputs.append(y_target)
        outputs.append(warped_surface)
        outputs.append(reg_output)

        outputs = tuple(
            KL.Activation('linear', name=f'{name}_out_{idx}')(tensor)
            for idx, tensor in enumerate(outputs)
        )

        super().__init__(inputs=[source, target, surface_points], outputs=outputs, name=name)

        self.references = vxm_model.references
        self.references.surface_points = surface_points
        self.references.warped_surface = warped_surface


# -----------------------------------------------------------------------------
# Template creation (unconditional and conditional)
# -----------------------------------------------------------------------------


class TemplateCreation(ne.modelio.LoadableModel):
    """Register images to a learnable atlas template."""

    @ne.modelio.store_config_args
    def __init__(
        self,
        inshape: Sequence[int],
        nb_unet_features=None,
        nb_unet_levels: Optional[int] = None,
        unet_feat_mult: int = 1,
        nb_unet_conv_per_level: int = 1,
        int_steps: int = 7,
        src_feats: int = 1,
        atlas_feats: Optional[int] = None,
        name: str = 'template_creation',
        **kwargs,
    ) -> None:
        if atlas_feats is None:
            atlas_feats = src_feats

        image_input = Input(shape=(*inshape, src_feats), name=f'{name}_image_input')
        atlas_layer = ne.layers.LocalParamWithInput(
            shape=(*inshape, atlas_feats),
            initializer='zeros',
            name=f'{name}_atlas_param',
        )
        atlas_tensor = atlas_layer(image_input)

        input_model = Model(inputs=[image_input], outputs=[image_input, atlas_tensor], name=f'{name}_inputs')

        vxm_model = VxmDense(
            inshape,
            nb_unet_features=nb_unet_features,
            nb_unet_levels=nb_unet_levels,
            unet_feat_mult=unet_feat_mult,
            nb_unet_conv_per_level=nb_unet_conv_per_level,
            int_steps=int_steps,
            bidir=False,
            src_feats=src_feats,
            trg_feats=atlas_feats,
            input_model=input_model,
            name=f'{name}_core',
            **kwargs,
        )

        core_outputs = list(vxm_model.outputs)
        reg_output = core_outputs.pop(-1)
        warped_image = core_outputs[0]

        outputs = (
            KL.Activation('linear', name=f'{name}_warped')(warped_image),
            KL.Activation('linear', name=f'{name}_atlas')(atlas_tensor),
            KL.Activation('linear', name=f'{name}_reg')(reg_output),
        )

        super().__init__(inputs=[image_input], outputs=outputs, name=name)

        self.references = vxm_model.references
        self.references.atlas = atlas_tensor


class ConditionalTemplateCreation(ne.modelio.LoadableModel):
    """Template creation conditioned on phenotype information."""

    @ne.modelio.store_config_args
    def __init__(
        self,
        inshape: Sequence[int],
        pheno_input_shape: Sequence[int],
        nb_unet_features=None,
        src_feats: int = 1,
        atlas_feats: Optional[int] = None,
        decoder_units: int = 128,
        decoder_layers: int = 2,
        name: str = 'conditional_template_creation',
        **kwargs,
    ) -> None:
        if atlas_feats is None:
            atlas_feats = src_feats

        image_input = Input(shape=(*inshape, src_feats), name=f'{name}_image_input')
        pheno_input = Input(shape=pheno_input_shape, name=f'{name}_pheno_input')

        dense = pheno_input
        for idx in range(decoder_layers):
            dense = KL.Dense(decoder_units, activation='relu', name=f'{name}_dense_{idx}')(dense)

        atlas_flat = KL.Dense(np.prod((*inshape, atlas_feats)), activation='linear', name=f'{name}_atlas_flat')(dense)
        atlas_tensor = KL.Reshape((*inshape, atlas_feats), name=f'{name}_atlas_reshape')(atlas_flat)

        input_model = Model(
            inputs=[image_input, pheno_input],
            outputs=[image_input, atlas_tensor],
            name=f'{name}_inputs',
        )

        vxm_model = VxmDense(
            inshape,
            nb_unet_features=nb_unet_features,
            src_feats=src_feats,
            trg_feats=atlas_feats,
            bidir=False,
            input_model=input_model,
            name=f'{name}_core',
            **kwargs,
        )

        core_outputs = list(vxm_model.outputs)
        reg_output = core_outputs.pop(-1)
        warped_image = core_outputs[0]

        outputs = (
            KL.Activation('linear', name=f'{name}_warped')(warped_image),
            KL.Activation('linear', name=f'{name}_atlas')(atlas_tensor),
            KL.Activation('linear', name=f'{name}_reg')(reg_output),
        )

        super().__init__(inputs=[image_input, pheno_input], outputs=outputs, name=name)

        self.references = vxm_model.references
        self.references.atlas = atlas_tensor
        self.references.pheno_input = pheno_input


# -----------------------------------------------------------------------------
# Probabilistic atlas segmentation
# -----------------------------------------------------------------------------


class ProbAtlasSegmentation(ne.modelio.LoadableModel):
    """Warp a probabilistic atlas to obtain soft segmentations."""

    @ne.modelio.store_config_args
    def __init__(
        self,
        inshape: Sequence[int],
        nb_labels: int,
        nb_unet_features=None,
        nb_unet_conv_per_level: int = 1,
        init_mu=None,
        init_sigma=None,
        warp_atlas: bool = True,
        stat_post_warp: bool = False,
        stat_nb_feats: int = 16,
        network_stat_weight: float = 1e-3,
        supervised_model: bool = False,
        gaussian_likelihood: bool = True,
        name: str = 'prob_atlas_seg',
        **kwargs,
    ) -> None:
        ndims = len(inshape)
        if ndims not in (1, 2, 3):
            raise ValueError('Only 1D, 2D, or 3D volumes are supported.')

        if init_mu is not None and not gaussian_likelihood:
            warnings.warn('init_mu ignored when gaussian_likelihood is False.')
        if init_sigma is not None and not gaussian_likelihood:
            warnings.warn('init_sigma ignored when gaussian_likelihood is False.')

        atlas_input = Input(shape=(*inshape, nb_labels), name=f'{name}_atlas_input')
        image_input = Input(shape=(*inshape, 1), name=f'{name}_image_input')

        base_model = Model(inputs=[atlas_input, image_input], outputs=[atlas_input, image_input], name=f'{name}_inputs')
        vxm_model = VxmDense(
            inshape,
            nb_unet_features=nb_unet_features,
            nb_unet_conv_per_level=nb_unet_conv_per_level,
            src_feats=nb_labels,
            trg_feats=1,
            input_model=base_model,
            name=f'{name}_core',
            **kwargs,
        )

        atlas = vxm_model.references.source
        image = vxm_model.references.target
        warped_atlas = vxm_model.references.y_source if warp_atlas else atlas
        flow = vxm_model.references.pos_flow

        if stat_post_warp and not warp_atlas:
            raise ValueError('stat_post_warp=True requires warp_atlas=True.')

        features = vxm_model.references.unet_features
        if features is None:
            features = KL.Concatenate(name=f'{name}_features')([atlas, image])

        combined = KL.Concatenate(name=f'{name}_post_warp_concat')([warped_atlas, image]) if stat_post_warp else features

        Conv = _conv_layer(ndims)
        stat_hidden = Conv(
            stat_nb_feats,
            kernel_size=3,
            padding='same',
            activation='relu',
            name=f'{name}_stat_hidden',
        )(combined)
        stat_features = Conv(
            nb_labels,
            kernel_size=3,
            padding='same',
            activation='relu',
            name=f'{name}_stat_features',
        )(stat_hidden)

        spatial_axes = list(range(1, ndims + 1))

        def _reduce_spatial(x):
            return ops.mean(x, axis=spatial_axes, keepdims=True)

        stat_mu = None
        stat_logssq = None

        if gaussian_likelihood:
            weaknorm = KI.RandomNormal(mean=0.0, stddev=1e-5)
            stat_mu_vol = Conv(
                nb_labels,
                kernel_size=3,
                padding='same',
                kernel_initializer=weaknorm,
                bias_initializer=weaknorm,
                name=f'{name}_mu_vol',
            )(stat_features)
            stat_logssq_vol = Conv(
                nb_labels,
                kernel_size=3,
                padding='same',
                kernel_initializer=weaknorm,
                bias_initializer=weaknorm,
                name=f'{name}_logssq_vol',
            )(stat_features)

            stat_mu = KL.Lambda(_reduce_spatial, name=f'{name}_mu_pool')(stat_mu_vol)
            stat_logssq = KL.Lambda(_reduce_spatial, name=f'{name}_logssq_pool')(stat_logssq_vol)

            if init_mu is not None:
                init_mu_arr = ops.convert_to_tensor(np.asarray(init_mu), dtype='float32')
                stat_mu = KL.Lambda(
                    lambda x: network_stat_weight * x + init_mu_arr,
                    name=f'{name}_mu_init',
                )(stat_mu)

            if init_sigma is not None:
                init_logssq = ops.convert_to_tensor([2 * np.log(s) for s in init_sigma], dtype='float32')
                stat_logssq = KL.Lambda(
                    lambda x: network_stat_weight * x + init_logssq,
                    name=f'{name}_sigma_init',
                )(stat_logssq)

            def gaussian_loglikelihood(args):
                img, mu, logsigmasq = args
                var = ops.exp(logsigmasq)
                return -0.5 * (
                    ops.log(2.0 * np.pi) + logsigmasq + ops.square(img - mu) / ops.maximum(var, 1e-6)
                )

            uloglhood = KL.Lambda(gaussian_loglikelihood, name=f'{name}_loglike')(
                [image, stat_mu, stat_logssq]
            )
        else:
            uloglhood = stat_features

        def log_pdf(args):
            prob_ll, atl = args
            return prob_ll + ops.log(ops.clip(atl, 1e-6, 1.0))

        logpdf = KL.Lambda(log_pdf, name=f'{name}_logpdf')([uloglhood, warped_atlas])

        if supervised_model:
            loss_vol = KL.Softmax(axis=-1, name=f'{name}_pdf')(logpdf)
        else:
            loss_vol = KL.Lambda(
                lambda x: ops.logsumexp(x, axis=-1, keepdims=True),
                name=f'{name}_loss_vol',
            )(logpdf)

        outputs = [loss_vol, flow]
        super().__init__(inputs=[image_input, atlas_input], outputs=outputs, name=name)

        self.references = ne.modelio.LoadableModel.ReferenceContainer()
        self.references.vxm_model = vxm_model
        self.references.uloglhood = uloglhood
        self.references.stat_mu = stat_mu if gaussian_likelihood else None
        self.references.stat_logssq = stat_logssq if gaussian_likelihood else None
        self.references.warped_atlas = warped_atlas

    def get_gaussian_warp_model(self):
        if self.references.stat_mu is None or self.references.stat_logssq is None:
            raise RuntimeError('Gaussian statistics unavailable; enable gaussian_likelihood to export them.')
        outputs = [
            self.references.uloglhood,
            self.references.stat_mu,
            self.references.stat_logssq,
            self.references.vxm_model.references.pos_flow,
        ]
        return Model(self.inputs, outputs, name=f'{self.name}_gaussian_model')


# -----------------------------------------------------------------------------
# Affine feature detector (SynthMorph-inspired)
# -----------------------------------------------------------------------------


class VxmAffineFeatureDetector(ne.modelio.LoadableModel):
    """Simplified affine registration network inspired by SynthMorph."""

    @ne.modelio.store_config_args
    def __init__(
        self,
        in_shape: Sequence[int],
        num_chan: int = 1,
        num_feat: int = 32,
        enc_nf: Sequence[int] = (64, 64, 64),
        per_level: int = 1,
        dropout: float = 0.0,
        weighted: bool = True,
        rigid: bool = False,
        make_dense: bool = True,
        bidir: bool = False,
        return_moved: bool = False,
        return_feat: bool = False,
        name: str = 'vxm_affine_det',
    ) -> None:
        num_dim = len(in_shape)
        if num_dim not in (2, 3):
            raise ValueError('Only 2D or 3D inputs supported for affine detector.')

        source = Input(shape=(*in_shape, num_chan), name=f'{name}_source_input')
        target = Input(shape=(*in_shape, num_chan), name=f'{name}_target_input')

        conv = _conv_layer(num_dim)
        pool = _pool_layer(num_dim)

        def feature_encoder(x, prefix: str):
            out = x
            for level, filters in enumerate(enc_nf):
                for rep in range(per_level):
                    out = conv(
                        filters,
                        kernel_size=3,
                        padding='same',
                        kernel_initializer='he_normal',
                        name=f'{name}_{prefix}_enc_{level}_{rep}',
                    )(out)
                    out = KL.LeakyReLU(0.2)(out)
                    if dropout > 0:
                        out = KL.Dropout(dropout)(out)
                if level < len(enc_nf) - 1:
                    out = pool(pool_size=2, name=f'{name}_{prefix}_pool_{level}')(out)
            out = conv(num_feat, kernel_size=3, padding='same', name=f'{name}_{prefix}_feat')(out)
            out = KL.LeakyReLU(0.2)(out)
            return out

        feat_source = feature_encoder(source, 'src')
        feat_target = feature_encoder(target, 'tgt')

        barycenter_axes = tuple(range(1, num_dim + 1))

        def barycenter(tensor):
            coords = ne.utils.barycenter(tensor, axes=barycenter_axes, normalize=True, shift_center=True)
            return coords * ops.convert_to_tensor(in_shape, dtype='float32')

        cen_src = KL.Lambda(barycenter, name=f'{name}_cen_src')(feat_source)
        cen_tgt = KL.Lambda(barycenter, name=f'{name}_cen_tgt')(feat_target)

        def channel_weights(feat):
            sums = ops.sum(feat, axis=barycenter_axes)
            denom = ops.sum(sums, axis=-1, keepdims=True)
            return sums / ops.maximum(denom, 1e-6)

        weights_src = KL.Lambda(channel_weights, name=f'{name}_weights_src')(feat_source)
        weights_tgt = KL.Lambda(channel_weights, name=f'{name}_weights_tgt')(feat_target)
        weights = KL.Multiply(name=f'{name}_weights_mul')([weights_src, weights_tgt]) if weighted else None

        fit_args_1 = [cen_src, cen_tgt]
        fit_args_2 = [cen_tgt, cen_src]
        if weights is not None:
            fit_args_1.append(weights)
            fit_args_2.append(weights)

        aff_src = KL.Lambda(lambda x: utils.fit_affine(*x), name=f'{name}_aff_src')(fit_args_1)
        aff_tgt = KL.Lambda(lambda x: utils.fit_affine(*x), name=f'{name}_aff_tgt')(fit_args_2)

        if rigid:
            def _rigid(matrix):
                params = utils.affine_matrix_to_params(matrix)
                params = params[..., : num_dim * (num_dim + 1) // 2]
                return layers.ParamsToAffineMatrix(ndims=num_dim)(params)

            aff_src = KL.Lambda(_rigid, name=f'{name}_rigid_src')(aff_src)
            aff_tgt = KL.Lambda(_rigid, name=f'{name}_rigid_tgt')(aff_tgt)

        transforms = [aff_src]
        if bidir:
            transforms.append(aff_tgt)

        if make_dense:
            transforms = [
                layers.AffineToDenseShift(in_shape, shift_center=False, name=f'{name}_dense_{idx}')(trf)
                for idx, trf in enumerate(transforms)
            ]

        outputs = list(transforms)

        if return_moved:
            for idx, trf in enumerate(transforms):
                img = source if idx == 0 else target
                moved = layers.SpatialTransformer(shift_center=False, name=f'{name}_moved_{idx}')([img, trf])
                outputs.append(moved)

        if return_feat:
            outputs.extend([feat_source, feat_target])

        outputs = [KL.Activation('linear', name=f'{name}_out_{idx}')(tensor) for idx, tensor in enumerate(outputs)]

        super().__init__(inputs=[source, target], outputs=outputs if len(outputs) > 1 else outputs[0], name=name)

        self.references = ne.modelio.LoadableModel.ReferenceContainer()
        self.references.source = source
        self.references.target = target
        self.references.base_transform = transforms[0]
        if bidir and len(transforms) > 1:
            self.references.inverse_transform = transforms[1]
        self.references.features_source = feat_source
        self.references.features_target = feat_target


# -----------------------------------------------------------------------------
# Hyper-network variants
# -----------------------------------------------------------------------------


class HyperVxmDense(ne.modelio.LoadableModel):
    """VoxelMorph with a lightweight hypernetwork controlling the flow field."""

    @ne.modelio.store_config_args
    def __init__(
        self,
        inshape: Sequence[int],
        nb_hyp_params: int = 1,
        nb_hyp_layers: int = 3,
        nb_hyp_units: int = 64,
        name: str = 'hyper_vxm_dense',
        **kwargs,
    ) -> None:
        hyp_input = Input(shape=(nb_hyp_params,), name=f'{name}_hyp_input')
        x = hyp_input
        for idx in range(nb_hyp_layers):
            x = KL.Dense(nb_hyp_units, activation='relu', name=f'{name}_dense_{idx}')(x)

        hyp_model = Model(inputs=hyp_input, outputs=x, name=f'{name}_hypernet')

        vxm_model = VxmDense(inshape, hyp_model=hyp_model, name=f'{name}_core', **kwargs)

        super().__init__(name=name, inputs=vxm_model.inputs, outputs=vxm_model.outputs)

        self.references = vxm_model.references
        self.references.hypernet = hyp_model


class HyperVxmJoint(ne.modelio.LoadableModel):
    """Joint registration model conditioned on auxiliary parameters."""

    @ne.modelio.store_config_args
    def __init__(
        self,
        inshape: Sequence[int],
        nb_hyp_params: int,
        nb_hyp_units: int = 64,
        name: str = 'hyper_vxm_joint',
        **kwargs,
    ) -> None:
        hyp_input = Input(shape=(nb_hyp_params,), name=f'{name}_hyp_input')
        hyp_hidden = KL.Dense(nb_hyp_units, activation='relu', name=f'{name}_hyp_hidden')(hyp_input)
        hyp_flow = KL.Dense(nb_hyp_units, activation='relu', name=f'{name}_hyp_flow')(hyp_hidden)

        hyp_model = Model(inputs=hyp_input, outputs=hyp_flow, name=f'{name}_hypernet')

        hyp_map_flat = KL.Dense(np.prod((*inshape, 1)), activation='tanh', name=f'{name}_hyp_map_flat')(hyp_hidden)
        hyp_map = KL.Reshape((*inshape, 1), name=f'{name}_hyp_map')(hyp_map_flat)

        source = Input(shape=(*inshape, 1), name=f'{name}_source_input')
        target = Input(shape=(*inshape, 1), name=f'{name}_target_input')

        concat_source = KL.Concatenate(name=f'{name}_source_concat')([source, hyp_map])
        concat_target = KL.Concatenate(name=f'{name}_target_concat')([target, hyp_map])

        input_model = Model(
            inputs=[source, target, hyp_input],
            outputs=[concat_source, concat_target],
            name=f'{name}_inputs',
        )

        vxm_model = VxmDense(
            inshape,
            src_feats=2,
            trg_feats=2,
            input_model=input_model,
            hyp_model=hyp_model,
            name=f'{name}_core',
            **kwargs,
        )

        super().__init__(inputs=vxm_model.inputs, outputs=vxm_model.outputs, name=name)

        self.references = vxm_model.references
        self.references.hypernet = hyp_model
        self.references.hyp_map = hyp_map


__all__ = [
    'ModelCheckpointParallel',
    'Transform',
    'Unet',
    'VxmDense',
    'VxmDenseSemiSupervisedSeg',
    'VxmDenseSemiSupervisedPointCloud',
    'InstanceDense',
    'ProbAtlasSegmentation',
    'TemplateCreation',
    'ConditionalTemplateCreation',
    'HyperVxmDense',
    'VxmAffineFeatureDetector',
    'HyperVxmJoint',
]
