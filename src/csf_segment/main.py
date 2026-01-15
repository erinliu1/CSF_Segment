import os
import argparse

from .data_config import DataConfig
from .load_data_paths import load_data_paths
from .cropper import Cropper
from .dataloader import MyDataLoader
from .inference import Inference

def main(
    output_root: str = 'data',      # root directory to save all outputs
    model_class: str = 'fusion',    # model class to use: 'fusion' or 'static'
    threshold: float = 0.5,         # probability threshold to binarize prediction
):
    try:
        config = DataConfig()
    except NotImplementedError as e:
        raise SystemExit(
            "Please set up src/csf_segment/data_config.py with your data paths."
        ) from e
   
    os.makedirs(output_root, exist_ok=True)
    data_paths = load_data_paths(config, output_root=output_root)

    for participant, runs in data_paths.items():
        print(f'Processing participant: {participant}')
        for run, paths in runs.items():
            print(f'  Processing run: {run}')

            save_dir = os.path.join(output_root, participant, run)
            os.makedirs(save_dir, exist_ok=True)

            cropper = Cropper(paths, save_directory=save_dir)
            sbref = cropper.load_sbref()
            fmri = cropper.load_fmri()

            dataloader = MyDataLoader(sbref, fmri, age=config.age_lookup[participant])
            
            inference = Inference(
                dataloader, 
                model_class=model_class,
                threshold=threshold,
                save_dir=save_dir, 
            )

            _ = cropper.uncrop(inference.thresholded_pred)
            print(f'    ✅ Saved result.nii.gz to {save_dir}')
   
    print('All done!')

def cli():
    parser = argparse.ArgumentParser(description="Run CSF segmentation")
    parser.add_argument("--output-root", default="data")
    parser.add_argument(
        "--model-class",
        choices=["fusion", "static"],
        default="fusion",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold between 0 and 1",
    )

    args = parser.parse_args()
    if not (0.0 <= args.threshold <= 1.0):
        parser.error("--threshold must be between 0 and 1")

    main(
        output_root=args.output_root,
        model_class=args.model_class,
        threshold=args.threshold,
    )

if __name__ == "__main__":
    cli()