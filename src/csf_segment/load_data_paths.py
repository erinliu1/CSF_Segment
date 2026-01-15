import os
import json

def load_data_paths(config, output_root, save_json=True):
    """
    Load the data from config.main_dir for participants listed in config.participants
     - For each participant, it stores the filepaths of each run's sbref, anatomical mask, and fMRI files.
     - Only includes runs that have all required files: sbref, anatomical mask, and fMRI files
     - Paths are returned as a nested dictionary structure: participant ID -> run ID -> file paths
    (optionally) Save the data paths dictionary as a JSON file 'data_paths.json'.
    """
    data_paths = {}
    for participant in sorted(config.participants):
        participant_paths = {} # where we will store this participant's runs (keys) and file paths (values)
            
        # look into the sbref folder for all matching sbref files
        sbref_folder = config.get_sbref_folder(participant)
        
        if not os.path.exists(sbref_folder):
            print(f"Skipping participant {participant}: SBRef folder does not exist.")
            continue
    
        for sbref_file in sorted(os.listdir(sbref_folder)):
            run_id = config.is_sbref_file(sbref_file)
            if run_id is not None:
                participant_paths[run_id] = {'sbref_path': f'{sbref_folder}/{sbref_file}'}

        # look into the anatomical folder for all matching anatomical files
        anatomical_folder = config.get_anatomical_folder(participant)

        if not os.path.exists(anatomical_folder):
            print(f"Skipping participant {participant}: Anatomical v4 masks folder does not exist.")
            continue
        
        for anatomical_file in sorted(os.listdir(anatomical_folder)):
            run_id = config.is_anatomical_file(anatomical_file)
            if run_id is not None:
                participant_paths.setdefault(run_id, {})['anatomical_path'] = f'{anatomical_folder}/{anatomical_file}'

        # look into the fmri folder for all matching fmri files
        fmri_folder = config.get_fmri_folder(participant)
        if not os.path.exists(fmri_folder):
            print(f"Skipping participant {participant}: fMRI folder does not exist.")
            continue

        for fmri_file in sorted(os.listdir(fmri_folder)):
            run_id = config.is_fmri_file(fmri_file)
            if run_id is not None:
                participant_paths.setdefault(run_id, {})['fmri_path'] = f'{fmri_folder}/{fmri_file}'

        # include only the runs that have all required files
        for run, paths in participant_paths.items():
            required = ['sbref_path', 'anatomical_path', 'fmri_path']
            missing_files = [key for key in required if key not in paths]
            if len(missing_files) > 0:
                print(f"Skipping participant {participant}, {run}: missing {', '.join(missing_files)}")
                continue
            data_paths.setdefault(participant, {})[run] = paths
        
    if save_json:
        with open(os.path.join(output_root, "data_paths.json"), "w") as f:
            json.dump(data_paths, f, indent=4, sort_keys=True)

    return data_paths
