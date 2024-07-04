'''Test a modulated Siren model by:
1. Loading it.
2. Test it on a single image.
'''
import torch
import pathlib
import numpy as np

from src.reconstruction.modulated_siren import ModulatedSiren
from src.data.mri_dataset import MRIDataset
from src.util.tiling import extract_with_inner_patches_with_info, extract_center, alternative_tiles_to_image, alternative_image_to_tiles, alternative_tiles_to_image2
from src.util.visualization import show_image, show_image_comparison, show_batch


def test_mod_siren(model_path: pathlib.Path):
    # Load the model
    mod_siren = ModulatedSiren(
        dim_in=2,
        dim_hidden=256,
        dim_out=1,
        num_layers=5,
        latent_dim=256,
        w0=1.0,
        w0_initial=30.0,
        use_bias=True,
        dropout=0.1,
        modulate=True,
        encoder_type="custom",
        outer_patch_size=32,
        inner_patch_size=16,
    )

    mod_siren.load_state_dict(torch.load(model_path))
    mod_siren.eval()

    tile_number = 54
    dataset = MRIDataset(pathlib.Path(r"C:\Users\jan\Documents\python_files\adlm\data\brain\singlecoil_train"), number_of_samples=14)
    undersampled_image = dataset[2][1][tile_number].float()
    fullysampled_image = dataset[2][0][tile_number].float()
    print(undersampled_image.shape)

    # Test the model
    with torch.no_grad():
        output = mod_siren(undersampled_image.unsqueeze(0))

    # Show the image
    show_image_comparison((extract_center(undersampled_image,32,16),extract_center(fullysampled_image,32,16), output))


def reconstruct_img(image: torch.Tensor, model: ModulatedSiren):
    if image.dim() == 2:
        image = image.unsqueeze(0)
    tiles, info = extract_with_inner_patches_with_info(image, 32, 16)
    with torch.no_grad():
        output = model(tiles).unsqueeze(0)
    print(output.shape)
    

def find_latest_model(directory: pathlib.Path):
    models = list(directory.glob("**/*.pth"))
    return pathlib.Path(max(models, key=lambda x: x.stat().st_mtime))

def reconstruction_script():
    model = ModulatedSiren(
        dim_in=2,
        dim_hidden=256,
        dim_out=1,
        num_layers=5,
        latent_dim=256,
        w0=1.0,
        w0_initial=30.0,
        use_bias=True,
        dropout=0.1,
        modulate=True,
        encoder_type="custom",
        outer_patch_size=32,
        inner_patch_size=16,
    )
    model.load_state_dict(torch.load(pathlib.Path(r"./output/mod_siren/mod_siren_2024-06-24_10-02-41/model_checkpoints/model_epoch_6200.pth")))
    model.eval()

    dataset = MRIDataset(pathlib.Path(r"C:\Users\jan\Documents\python_files\adlm\data\brain\singlecoil_val"), number_of_samples=1)
    image = dataset._getitem_img(0)[1].float()
    fullysampled_image = dataset._getitem_img(0)[0].float()
    patches, info = alternative_image_to_tiles(image.unsqueeze(0), 32, 10)
    with torch.no_grad():
        output = model(patches)
    #show_image(output[1000,:,:])
    #rec_image = alternative_tiles_to_image(output.squeeze(1), info, 16,10)
    rec_image2 = alternative_tiles_to_image2(output.squeeze(1), info, 16,10)
    show_image_comparison((image,fullysampled_image,rec_image2))

if __name__ == "__main__":
    reconstruction_script()
    #print(find_latest_model(pathlib.Path(r"C:\Users\jan\Documents\python_files\adlm\refactoring\output\mod_siren")))