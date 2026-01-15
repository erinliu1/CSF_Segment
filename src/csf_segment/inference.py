import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from .model import MyFusion, MyMLP

class Inference():
    def __init__(
        self, 
        dataloader, 
        model_class,    # model class to use: 'fusion' or 'static'
        save_dir,       # folder to save pred mask npy and/or visualization
        threshold=0.5,  # probability threshold to binarize prediction
        save_npy=True,
        save_visualization=True,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.shape = dataloader.sbref.shape
        self.sbref = dataloader.sbref
        self.model_class = model_class
        self.threshold = threshold
        self.save_dir = save_dir

        self.X_ts, self.X_static = self.reshape_inputs(
            fmri_psc=dataloader.fmri_psc, 
            sbref=dataloader.sbref,
            positions=dataloader.positions, 
            age=dataloader.age)

        self.pred = self.get_average_prediction()
        if save_npy:
            os.makedirs(self.save_dir, exist_ok=True)
            np.save(f'{self.save_dir}/pred_{self.model_class}.npy', self.pred)
        
        self.thresholded_pred = self._threshold_prediction(self.pred)

        if save_visualization:
            os.makedirs(self.save_dir, exist_ok=True)
            self._save_visualization()
        
    def reshape_inputs(self, fmri_psc, sbref, positions, age):
        """
        Input dimensions:
        - fmri_psc shape: (nx, ny, nz, num_timesteps)
        - sbref shape: (x, y, z)
        - positions shape: (x, y, z, 3)
        - age: 1 or 0 (or None)

        Output dimensions:
        - X_ts shape: (num_voxels, num_timesteps)
        - X_static shape: (num_voxels, num_static_features)
        """
        nx, ny, nz = sbref.shape
        num_voxels = nx * ny * nz

        # get time series features as shape (num_voxels, num_timesteps)
        T = fmri_psc.shape[-1] # num_timesteps
        X_ts = fmri_psc.reshape(-1, T)

        # stack static features as shape (num_voxels, num_static_features)
        static_features = []
        static_features.append(sbref.reshape(-1, 1))
        static_features.append(positions.reshape(-1, 3))
        static_features.append(np.full((num_voxels, 1), age))
        X_static = np.hstack(static_features)

        return self._to_tensor(X_ts), self._to_tensor(X_static)
    
    def _to_tensor(self, X):
        return torch.tensor(X, dtype=torch.float32).to(self.device)

    def get_average_prediction(self):
        PACKAGE_DIR = os.path.dirname(__file__)
        model_checkpoints_dir = os.path.join(PACKAGE_DIR, f'model_checkpoints/{self.model_class}')
        preds = []
        for model_name in os.listdir(f'{model_checkpoints_dir}'):
            model_path = f'{model_checkpoints_dir}/{model_name}'
            model = self._load_model(model_path)
            pred = self._get_prediction(model)
            preds.append(pred)
        avg_pred = np.mean(np.stack(preds, axis=0), axis=0)
        return avg_pred
            
    def _load_model(self, model_path):        
        if self.model_class == 'fusion':
            model = MyFusion()
        elif self.model_class == 'static':
            model = MyMLP()
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        model.load_state_dict(state_dict)
        return model.to(self.device)
        
    def _get_prediction(self, model):
        with torch.no_grad():
            if self.model_class == 'fusion':
                pred = model(self.X_ts, self.X_static).cpu().numpy()
            elif self.model_class == 'static':
                pred = model(self.X_static).cpu().numpy()
        return pred.reshape(self.shape)

    def _threshold_prediction(self, pred):
        pred_thresholded = (pred >= self.threshold).astype(np.uint8)
        return pred_thresholded

    def _save_visualization(self):   
        fig, axes = plt.subplots(3, 4, figsize=(14, 10))

        # --- SBRef ---
        for i, ax in enumerate(axes[0]):
            ax.imshow(self.sbref[:, :, i], cmap="viridis")
            ax.set_title(f"SBRef Slice {i+1}")
            ax.axis("off")

        # --- Prediction ---
        for i, ax in enumerate(axes[1]):
            ax.imshow(self.pred[:, :, i], cmap="viridis")
            ax.set_title(f"Prediction Slice {i+1}")
            ax.axis("off")

        # --- Thresholded ---
        for i, ax in enumerate(axes[2]):
            ax.imshow(self.thresholded_pred[:, :, i], cmap="viridis")
            ax.set_title(f"Threshold ({self.threshold}) Slice {i+1}")
            ax.axis("off")

        save_path = f"{self.save_dir}/visualization_{self.model_class}.png"
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
