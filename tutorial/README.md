# VoxelMorph tutorials

This folder collects step-by-step guides for the most common VoxelMorph workflows. Start with the local quickstarts below, then explore the linked notebooks for more advanced experiments.

## Local guides
- [Quickstart: register two volumes](quickstart_registration.md)
- [Quickstart: train a model from scratch](quickstart_training.md)
- [Tutorial: build a 3D atlas](atlas_generation_3d.md)

## External notebooks and demos
- [VoxelMorph tutorial](http://tutorial.voxelmorph.net/) - conceptual overview and TensorFlow-era walkthrough
- [Deformable SynthMorph demo](https://colab.research.google.com/drive/1zaDnAJGUokS0knqWttuTgrRJMb6zxukI?usp=sharing) - train without real data using synthetic shapes
- [Affine SynthMorph demo](https://colab.research.google.com/drive/1QClknfaZIYklBjmBy-nUn83h85bpo8r0?usp=sharing) - anatomy-aware affine registration
- [CT->MRI SynthMorph demo](https://colab.research.google.com/drive/1aWbFiyQw5mtJbTglpniAOMimGo_l8BYP?usp=drive_link) - cross-modal registration after Hounsfield clipping
- [SynthMorph shapes walkthrough](https://colab.research.google.com/drive/14s2h0j_Aoncp587vmpjsQBe6PQTxyVP5) - run a trained 3D shapes model
- [VoxelMorph training on OASIS](https://colab.research.google.com/drive/1ZefmWXBupRNsnIbBbGquhVDsk-7R7L1S?usp=sharing) - end-to-end atlas registration with public data
- [Warp annotations alongside images](https://colab.research.google.com/drive/1V0CutSIfmtgDJg1XIkEnGteJuw0u7qT-#scrollTo=h1KXYz-Nauwn) - semi-supervised setup
- [Template (atlas) construction](https://colab.research.google.com/drive/1SkQbrWTQHpQFrG4J2WoBgGZC9yAzUas2?usp=sharing) - learn reference anatomies
- [Visualise warps as grids](https://colab.research.google.com/drive/1F8f1imh5WfyBv-crllfeJBFY16-KHl9c?usp=sharing)
- [Invert non-diffeomorphic warps](https://colab.research.google.com/drive/1juAJRYhPPDNbO9yRtlc0VGhbIFSuhpJ2?usp=sharing)

## Bundled sample data
The root `data/` directory ships with an atlas, a test scan, labels, and probability maps. Use them to follow the quickstarts or sanity-check your environment before switching to your own dataset. Additional pretrained weights live in `data/readme.md`.
