# Freehand face control

## Usage
- Make a base mesh + other blend shapes
- Export as objs with the z axis as the forward direction
- Load them all into a `Mesh` with the base mesh first
- Draw target lines as paths in an SVG
- Load them in via `load_svg` with `out_width` and `out_height` trying to match the size of the obj
- Run `python optimize.py`

## Installing
```
conda create -n torch-nightly python=3.8 
conda activate torch-nightly
pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
pip install svgelements
```
