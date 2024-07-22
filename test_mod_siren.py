"""
Script to test the modulated SIREN model.
"""

import os
import pathlib

import numpy as np
import torch

from src.configuration.configuration import load_configuration, parse_args
from src.data.mri_dataset import MRIDataset
from src.data.mri_sampler import MRISampler
from src.networks.modulated_siren import ModulatedSiren
from src.util.error import metrics_error, visual_error
from src.util.tiling import image_to_patches
from src.util.visualization import metrics_boxplot, metrics_density_plot


def save_args_to_file(args, output_dir):
    """
    Save the arguments to a file.

    Args:
        args (argparse.Namespace): The arguments to save.
        output_dir (str): The directory to save the arguments in.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    config_path = os.path.join(output_dir, "config.txt")

    with open(config_path, "w") as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")


def save_metrics_summary(psnr_values, ssim_values, nrmse_values, output_dir):
    """
    Save the summary of metrics to a text file.

    Args:
        psnr_values (list): List of PSNR values.
        ssim_values (list): List of SSIM values.
        nrmse_values (list): List of NRMSE values.
        output_dir (str): The directory to save the summary in.
    """
    metrics_summary = {
        "PSNR": {
            "mean": np.mean(psnr_values),
            "std": np.std(psnr_values),
            "min": np.min(psnr_values),
            "max": np.max(psnr_values),
        },
        "SSIM": {
            "mean": np.mean(ssim_values),
            "std": np.std(ssim_values),
            "min": np.min(ssim_values),
            "max": np.max(ssim_values),
        },
        "NRMSE": {
            "mean": np.mean(nrmse_values),
            "std": np.std(nrmse_values),
            "min": np.min(nrmse_values),
            "max": np.max(nrmse_values),
        },
    }

    summary_path = os.path.join(output_dir, "metrics_summary.txt")
    with open(summary_path, "w") as f:
        for metric, values in metrics_summary.items():
            f.write(f"{metric}:\n")
            for stat, value in values.items():
                f.write(f"  {stat}: {value}\n")
            f.write("\n")


def test_mod_siren(config):
    """
    Test the modulated SIREN model.

    Args:
        config (argparse.Namespace): The configuration to use for testing.
    """
    print("Testing the modulated SIREN...")

    output_dir = f"{config.testing.output_dir}/{config.testing.output_name}/test"

    # Set the device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Load the model
    mod_siren = ModulatedSiren(
        dim_in=config.model.dim_in,
        dim_hidden=config.model.dim_hidden,
        dim_out=config.model.dim_out,
        num_layers=config.model.num_layers,
        latent_dim=config.model.latent_dim,
        w0=config.model.w0,
        w0_initial=config.model.w0_initial,
        use_bias=config.model.use_bias,
        dropout=config.model.dropout,
        modulate=True,
        encoder_type=config.model.encoder_type,
        encoder_path=config.model.encoder_path,
        outer_patch_size=config.model.outer_patch_size,
        inner_patch_size=config.model.inner_patch_size,
        siren_patch_size=config.model.siren_patch_size,
        device=device,
        activation=config.model.activation,
    )

    mod_siren.load_state_dict(torch.load(config.testing.model_path))

    mod_siren.to(device)

    if config.data.visual_samples:
        print("Evaluating visual samples ...")
        sampler = MRISampler(pathlib.Path(config.data.dataset), config.data.test_files)

        with torch.no_grad():
            mod_siren.eval()

            for i in range(config.data.visual_samples):
                print(
                    f"Processing visual sample {i + 1}/{config.data.visual_samples}..."
                )
                # Load the image
                fully_sampled_img, undersampled_img, filename = (
                    sampler.get_random_sample()
                )

                # unsqueeze image to add batch dimension
                fully_sampled_img = fully_sampled_img.unsqueeze(0).float().to(device)
                undersampled_img = undersampled_img.unsqueeze(0).float().to(device)

                fully_sampled_patch, _ = image_to_patches(
                    fully_sampled_img,
                    config.model.outer_patch_size,
                    config.model.inner_patch_size,
                )
                undersampled_patch, undersampled_information = image_to_patches(
                    undersampled_img,
                    config.model.outer_patch_size,
                    config.model.inner_patch_size,
                )

                output_dir_temp = os.path.join(output_dir, filename)
                if not os.path.exists(output_dir_temp):
                    os.makedirs(output_dir_temp)

                visual_error(
                    mod_siren,
                    output_dir_temp,
                    filename,
                    fully_sampled_patch,
                    undersampled_patch,
                    undersampled_information,
                    device,
                    config.model.outer_patch_size,
                    config.model.inner_patch_size,
                    config.model.siren_patch_size,
                )

    print("Evaluating metric samples ...")
    sampler = MRISampler(pathlib.Path(config.data.dataset), config.data.test_files)

    if not config.data.metric_samples:
        config.data.metric_samples = len(sampler)

    psnr_values = []
    ssim_values = []
    nrmse_values = []
    filenames = []

    with torch.no_grad():
        mod_siren.eval()

        for i in range(config.data.metric_samples):
            print(f"Processing metric sample {i + 1}/{config.data.metric_samples}...")
            # Load the image
            fully_sampled_img, undersampled_img, filename = sampler.get_random_sample()

            # unsqueeze image to add batch dimension
            fully_sampled_img = fully_sampled_img.unsqueeze(0).float().to(device)
            undersampled_img = undersampled_img.unsqueeze(0).float().to(device)

            fully_sampled_patch, _ = image_to_patches(
                fully_sampled_img,
                config.model.outer_patch_size,
                config.model.inner_patch_size,
            )
            undersampled_patch, undersampled_information = image_to_patches(
                undersampled_img,
                config.model.outer_patch_size,
                config.model.inner_patch_size,
            )

            psnr, ssim, nrmse = metrics_error(
                mod_siren,
                fully_sampled_patch,
                undersampled_patch,
                undersampled_information,
                device,
                config.model.outer_patch_size,
                config.model.inner_patch_size,
                config.model.siren_patch_size,
            )

            psnr_values.append(psnr)
            ssim_values.append(ssim)
            nrmse_values.append(nrmse)
            filenames.append(filename)

        # Write them to a csv file
        with open(os.path.join(output_dir, f"metrics_error.csv"), "w") as f:
            f.write("FILENAME,PSNR,SSIM,NRMSE\n")
            for filename, psnr_value, ssim_value, nrmse_value in zip(
                filenames, psnr_values, ssim_values, nrmse_values
            ):
                f.write(f"{filename},{psnr_value},{ssim_value},{nrmse_value}\n")

        save_metrics_summary(psnr_values, ssim_values, nrmse_values, output_dir)

        # Visualize the metrics
        metrics_boxplot(
            {"PSNR": psnr_values, "SSIM": ssim_value, "NRMSE": nrmse_value}, output_dir
        )

        metrics_density_plot(
            {"PSNR": psnr_values, "SSIM": ssim_value, "NRMSE": nrmse_value}, output_dir
        )


if __name__ == "__main__":
    args = parse_args()
    config_path = pathlib.Path(args.config)
    config = load_configuration(config_path, testing=True)
    print(config)
    test_mod_siren(config)
