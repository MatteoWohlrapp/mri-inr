'''Test a modulated Siren model by:
1. Loading it.
2. Test it on a single image.
'''
import torch
import pathlib

from src.reconstruction.modulated_siren import ModulatedSiren
from src.data.mri_dataset import MRIDataset
from src.util.tiling import extract_with_inner_patches_with_info
from src.util.visualization import show_image, show_image_comparison


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

    
    dataset = MRIDataset(pathlib.Path(r"C:\Users\jan\Documents\python_files\adlm\data\brain\singlecoil_train"), number_of_samples=1)
    undersampled_image = dataset[0][1][50].float()

    # Test the model
    with torch.no_grad():
        output = mod_siren(undersampled_image.unsqueeze(0))
    print(undersampled_image.shape)
    print(output.shape)

    # Show the image
    show_image_comparison((undersampled_image, output))

if __name__ == "__main__":
    test_mod_siren(pathlib.Path(r"C:\Users\jan\Documents\python_files\adlm\refactoring\models\model_checkpoints\model1_model.pth"))