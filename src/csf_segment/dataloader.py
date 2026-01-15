import numpy as np

class MyDataLoader:
    """
    Loads data to be fed into CSF segmentation model for prediction.
    Each participant's loaded data includes:
        - sbref: normalized SBRef (x, y, z) between 0 and 1
        - positions: normalized voxel coordinates (x, y, z, 3) between -1 and 1
        - age: 0 (<60) or 1 (>=60)
        - fmri_psc: percent signal change above 15-percentile baseline (x, y, z, 316) where (60 + 60 s / 0.378 s per TR = 316)
    """

    def __init__(self,
        sbref,                  # cropped npy
        fmri,                   # cropped npy
        age,                    # age of the participant (int)
        TR=0.378,               # fMRI repetition time (seconds) that the model expects 
        edge_segments=(60,60),  # tuple of (start_seconds, end_seconds) of fMRI time series to load; removes middle part of the time series
    ):
        self.sbref = self._get_sbref_normalized(sbref)
        self.positions = self._get_positions(self.sbref.shape)
        self.age = self._get_age(age)
        self.fmri_psc = self._get_fmri_psc(fmri, edge_segments, TR)

    def _get_sbref_normalized(self, sbref):
        # return normalized sbref with all values between [0, 1]
        return (sbref - np.min(sbref)) / (np.max(sbref) - np.min(sbref))

    def _get_positions(self, shape):
        # return normalized voxel coordinates between [-1, 1] for each voxel in the given shape
        x = np.arange(shape[0])[:, None, None]
        y = np.arange(shape[1])[None, :, None]
        z = np.arange(shape[2])[None, None, :]
        x_center, y_center = shape[0] // 2, shape[1] // 2
        x_dist = (x - x_center) / (x_center if x_center > 0 else 1)
        y_dist = (y - y_center) / (y_center if y_center > 0 else 1)
        positions = np.stack(
            np.meshgrid(x_dist[:, 0, 0], y_dist[0, :, 0], z[0, 0, :], indexing="ij"),
            axis=-1,
        )
        return positions

    def _get_age(self, age):
        # return 0 if age < 60, 1 if age >= 60
        return 0 if int(age) < 60 else 1
   
    def _get_fmri_psc(self, fmri, edge_segments, TR, numVolsTrim=10):
        # compute 15th percentile baseline per voxel over time, excluding the first numVolsTrim volumes
        baseline = np.percentile(fmri[..., numVolsTrim:], 15, axis=-1, keepdims=True)
        baseline = np.clip(baseline, 1e-6, None)
        fmri_psc = ((fmri - baseline) / baseline) * 100

        if edge_segments is None:
            return fmri_psc

        # get the beginning and end segments of the fmri psc and trim out the middle part
        num_timesteps = fmri_psc.shape[-1]
        start_seconds, end_seconds = edge_segments
        start_frames = int(start_seconds / TR)
        end_frames = int(end_seconds / TR)
        fmri_psc = np.concatenate((fmri_psc[..., :start_frames], fmri_psc[..., num_timesteps - end_frames:]), axis=-1)
        return fmri_psc
