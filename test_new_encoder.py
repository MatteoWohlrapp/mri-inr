"""Script to load an image run it through the new Autoencoder in tiles and then reconstruct the image from the tiles."""
import torch
import pathlib
from src.data.mri_dataset import MRIDataset
from src.networks.encoding.new_encoder import Autoencoder_v1, Autoencoder_v2
from src.util.tiling import filter_black_patches, reintegrate_black_patches, image_to_patches, patches_to_image
from src.util.visualization import save_image
from src.util.error import visual_error


def main():
    # Load the model
    autoencoder = Autoencoder_v2()
    autoencoder.load_state_dict(torch.load(r"C:\Users\jan\Documents\python_files\adlm\refactoring\output\model_1139.pth"))
    autoencoder.eval()

    # Load the image
    dataset = MRIDataset(pathlib.Path(r"C:\Users\jan\Documents\python_files\adlm\data\brain\singlecoil_train\processed_files"), number_of_samples=1)
    with torch.no_grad():
        image = dataset.get_random_image()[0].unsqueeze(0).float()

        # Split the image into patches
        patches, info  = image_to_patches(image, 24, 24)

        visual_error(
            autoencoder,
            r"C:\Users\jan\Documents\python_files\adlm\refactoring\output",
            "test",
            patches,
            patches,
            info,
            torch.device("cpu"),
            24,
            24,
            24
        
        )

if __name__ == "__main__":
    main()