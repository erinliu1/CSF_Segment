## CSF Segmentation Tool

### 1. Configure your data paths
Edit `src/csf_segment/data_config.py` and fill in the TODOs. An example is given at the bottom.

This will ask you for the filepath patterns for your SBRef images, fMRI volumes, and anatomical V4 masks. **If you don't already have the anatomical V4 masks extracted, you can do so with FreeSurfer [here](https://github.com/erinliu1/FreeSurfer-Fourth-Ventricle-Mask).**

### 2. Install
```bash
pip install -e .
```

### 3. Run
```bash
python -m csf_segment.main 
```
or with CLI if you want to change the parameters. 
```bash
csf-segment --model-class fusion --threshold 0.5
```
* 2 possible model classes: `static` (only uses static features), or `fusion` (static features + fMRI timeseries)
* threshold determines what is the minimum probability predicted by the model to select the voxel