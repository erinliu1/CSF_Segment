import os
import nibabel as nib
import numpy as np
import json

class Cropper:
    def __init__(self, paths, save_directory):
        self.anatomical_path = paths['anatomical_path']
        self.sbref_path = paths['sbref_path']
        self.fmri_path = paths['fmri_path']
        self.save_directory = save_directory
        os.makedirs(self.save_directory, exist_ok=True)

        self.anatomical_nifti, self.anatomical_npy = self.load_anatomical()
        self.sbref_nifti = self.load_sbref_nifti()
        self.sbref_npy = self.load_sbref_npy_full()
        self.bbox_coordinates = self.get_bbox()
    
    def load_anatomical(self):
        anatomical_nifti = nib.load(self.anatomical_path)
        anatomical_npy = anatomical_nifti.get_fdata()
        return anatomical_nifti, anatomical_npy

    def load_sbref_nifti(self):
        return nib.load(self.sbref_path)

    def load_sbref_npy_full(self):
        return self.sbref_nifti.get_fdata()

    def get_bbox(
        self,
        dilation=3,
        save_json=False, 
        save_nifti=False,
    ):
        """
        Create a bounding box around the anatomical v4 mask and return bbox coordinates as a dictionary.
        (optionally) Save bbox coordinates in a JSON file and save bbox mask as NIfTI file.
        """
        nx, ny, nz = self.anatomical_npy.shape

        # get the edge coordinates of the anatomical mask
        coordinates = np.argwhere(self.anatomical_npy > 0)                  # get all nonzero coordinates
        if coordinates.size == 0:
            raise ValueError(f"Mask is empty: {self.anatomical_path}")
        min_x, min_y, min_z = np.min(coordinates, axis=0)                   # get the minimum nonzero x, y, z coordinates
        max_x, max_y, max_z = np.max(coordinates, axis=0)                   # get the maximum nonzero x, y, z coordinates

        # dilate edge coordinates by d voxels in x and y directions
        d = dilation
        min_x, min_y = max(min_x - d, 0), max(min_y - d, 0)                 # ensure min coords don't go below 0
        max_x, max_y = min(max_x + d, nx - 1), min(max_y + d, ny - 1)       # ensure max coords don't exceed the volume size
        min_z, max_z = 0, 3                                                 # by default, take the bottom 4 slices only

        bbox_coordinates = {
            "min_x": int(min_x),
            "min_y": int(min_y),
            "min_z": int(min_z),
            "max_x": int(max_x),
            "max_y": int(max_y),
            "max_z": int(max_z)
        }

        if save_json: # save bbox coordinates into json file (can use later to load the predicted mask into the full volume)
            with open(f'{self.save_directory}/bbox.json', 'w') as json_file:
                json.dump(bbox_coordinates, json_file)

        if save_nifti: # save the bbox mask as NIfTI image with 1s within the bbox, 0s everywhere else (can load in FreeSurfer to visualize)
            bbox_mask = np.zeros(self.anatomical_npy.shape)
            bbox_mask[min_x:max_x+1, min_y:max_y+1, min_z:max_z+1] = 1
            bbox_obj = nib.Nifti1Image(bbox_mask, affine=self.anatomical_nifti.affine, header=self.anatomical_nifti.header)
            nib.save(bbox_obj, f'{self.save_directory}/bbox.nii.gz')
        
        return bbox_coordinates

    def unpack_bbox(self):
        return (
            self.bbox_coordinates['min_x'],
            self.bbox_coordinates['min_y'],
            self.bbox_coordinates['min_z'],
            self.bbox_coordinates['max_x'],
            self.bbox_coordinates['max_y'],
            self.bbox_coordinates['max_z'],
        )

    def crop_sbref(self, save_npy=True):
        """
        Crop the sbref image using the provided bbox coordinates and return the cropped sbref array.
        (optionally) Save the cropped sbref as a .npy file.
        """
        min_x, min_y, min_z, max_x, max_y, max_z = self.unpack_bbox()
        if len(self.sbref_npy.shape) == 3:
            sbref_cropped = self.sbref_npy[min_x:max_x+1, min_y:max_y+1, min_z:max_z+1]
        elif len(self.sbref_npy.shape) == 4:
            # handle SBRef images that have 2 volumes (phase and magnitude). Choose the volume representing the magnitude
            # the heuristic we use here is to choose the one with the higher mean value, but check if the output looks correct
            print(f"⚠️ SBRef image has unexpected shape {self.sbref_npy.shape}.")
            sbref_cropped_dim0 = self.sbref_npy[:,:,:,0][min_x:max_x+1, min_y:max_y+1, min_z:max_z+1]
            sbref_cropped_dim1 = self.sbref_npy[:,:,:,1][min_x:max_x+1, min_y:max_y+1, min_z:max_z+1]
            sbref_cropped = sbref_cropped_dim0 if np.mean(sbref_cropped_dim0) > np.mean(sbref_cropped_dim1) else sbref_cropped_dim1
        else:
            raise Exception(f"\tError: Unexpected SBRef image with shape: {self.sbref_npy.shape}.")
        if save_npy:
            np.save(f'{self.save_directory}/sbref.npy', sbref_cropped)

        return sbref_cropped

    def crop_fmri(self, save_npy=True):
        """
        Crop the fmri volume using the provided bbox coordinates and return the cropped fmri array.
        (optionally) Save the cropped fmri as a .npy file.
        """
        min_x, min_y, min_z, max_x, max_y, max_z = self.unpack_bbox()
        fmri_obj = nib.load(self.fmri_path)
        fmri_image = fmri_obj.get_fdata()
        if len(fmri_image.shape) != 4:
            raise Exception(f"\tError: Unexpected fMRI image with shape: {fmri_image.shape}. Should be 4D.")
        fmri_cropped = fmri_image[min_x:max_x+1, min_y:max_y+1, min_z:max_z+1, :]
        
        if save_npy:
            np.save(f'{self.save_directory}/fmri.npy', fmri_cropped)

        return fmri_cropped

    def load_sbref(self):
        """
        Load the cropped sbref npy file if it exists; otherwise, crop the sbref image and save the npy file.
        """
        if not os.path.exists(f'{self.save_directory}/sbref.npy'):
            return self.crop_sbref(save_npy=True)
        return np.load(f'{self.save_directory}/sbref.npy')
        
    def load_fmri(self):
        """
        Load the cropped fmri npy file if it exists; otherwise, crop the fmri image and save the npy file.
        """
        if not os.path.exists(f'{self.save_directory}/fmri.npy'):
            return self.crop_fmri(save_npy=True)
        return np.load(f'{self.save_directory}/fmri.npy')
        
    def uncrop(self, cropped):
        """
        Uncrop the given cropped npy array back to the original volume size as in the original SBRef image.
        """
        min_x, min_y, min_z, max_x, max_y, max_z = self.unpack_bbox()

        # create an empty volume of original volume size and place the cropped data back into it
        uncropped = np.zeros(self.sbref_npy.shape)
        uncropped[min_x:max_x+1, min_y:max_y+1, min_z:max_z+1] = cropped

        uncropped_obj = nib.Nifti1Image(uncropped, affine=self.sbref_nifti.affine, header=self.sbref_nifti.header)
        nib.save(uncropped_obj, f'{self.save_directory}/result.nii.gz')

        return uncropped