import re

"""
Instructions: Fill out the functions below with the filepath names and filename patterns for your data.
Details in the comments.
"""

class DataConfig:
    def __init__(self):
        self.main_dir = self.get_main_directory()
        self.participants = self.get_participants()
        self.age_lookup = self.get_age_lookup()

    def get_main_directory(self):
        # full directory containing the folders for each participant
        raise NotImplementedError #TODO
    
    def get_participants(self):
        # list of participant IDs to use (should match folder names in get_main_directory)
        raise NotImplementedError #TODO
    
    def get_age_lookup(self):
        # dictionary mapping participant IDs to their ages (int)
        raise NotImplementedError #TODO

    def get_sbref_folder(self, participant):
        # returns the full path to the given participant's SBRef files
        raise NotImplementedError #TODO
    
    def is_sbref_file(self, filename):
        # checks if the given filename matches the SBRef file pattern. 
        # If so, return the run ID (inside the first paranthesis in the regex, e.g. run01). 
        # Otherwise, return None.
        raise NotImplementedError #TODO

    def get_fmri_folder(self, participant):
        # returns the full path to the given participant's fMRI files
        raise NotImplementedError #TODO
    
    def is_fmri_file(self, filename):
        # checks if the given filename matches the fMRI file pattern.
        # If so, return the run ID (inside the first paranthesis in the regex, e.g. run01). 
        # Otherwise, return None.
        raise NotImplementedError #TODO
    
    def get_anatomical_folder(self, participant):
        # returns the full path to the given participant's anatomical fourth ventricle mask files
        raise NotImplementedError #TODO
    
    def is_anatomical_file(self, filename):
        # checks if the given filename matches the anatomical fourth ventricle mask file pattern.
        # If so, return the run ID (inside the first paranthesis in the regex, e.g. run01). 
        # Otherwise, return None.
        raise NotImplementedError #TODO

#--------------------- EXAMPLE ---------------------

class ExampleDataConfig:

    def __init__(self):
        self.main_dir = "/orcd/data/ldlewis/001/om2/shared/aging"
        self.participants = ['ag105', 'ag229']
        self.age_lookup = {
            'ag105': 65,
            'ag229': 58,
        }
        
    def get_sbref_folder(self, participant):
        return f"{self.main_dir}/{participant}/ses-02-night/mri/sbref"

    def is_sbref_file(self, filename):
        # checks for filenames that match 'run01_SBRef.nii.gz', 'run02_SBRef.nii.gz', etc. 
        match = re.match(r"^(run\d{2})_SBRef\.nii\.gz$", filename)
        return match.group(1) if match else None

    def get_fmri_folder(self, participant):
        return f"{self.main_dir}/{participant}/ses-02-night/mri/stcfsl"
    
    def is_fmri_file(self, filename):
        # checks for filenames that match 'run01_rest_stc.nii.gz', 'run02_rest_stc.nii.gz', etc.
        match = re.match(r"^(run\d{2})_rest_stc\.nii\.gz$", filename)
        return match.group(1) if match else None

    def get_anatomical_folder(self, participant):
        return f"/orcd/data/ldlewis/001/om/erinliu/anatomical_segment/data/{participant}/CSF"
        # return f"{self.main_dir}/{participant}/ses-02-night/mri/masks/CSF"
    
    def is_anatomical_file(self, filename):
        # checks for filenames that match 'v4_run01.nii.gz', 'v4_run02.nii.gz', etc.
        match = re.match(r"^v4_(run\d{2})\.nii\.gz$", filename)
        return match.group(1) if match else None
