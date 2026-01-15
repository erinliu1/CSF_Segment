## CSF Segmentation Tool

### 1. Configure your data paths
Edit `src/csf_segment/data_config.py` and fill in the TODOs. Follow the example for regex filename matching.

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