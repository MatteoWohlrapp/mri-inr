import pathlib
from src.util.visualization import metrics_density_plot, metrics_boxplot, save_image, plot_k_space, show_image, show_image_complex, show_batch
from src.data.mri_sampler import MRISampler
import numpy as np
from src.networks.encoding.custom_mri_encoder import config, build_autoencoder, CustomEncoder
from src.networks.encoding.new_encoder import HardcodedAutoencoder, HardcodedEncoder
from src.data.preprocessing import load_mri_scan_complex, load_mri_scan, load_mri_scan2
import torch
import os
import numpy as np
def plot_images(n = 500):
    # Just create plots of n fully sampled images.
    input_dir = pathlib.Path(r"../../dataset/fastmri/brain/singlecoil_val/processed_files")
    output_dir = pathlib.Path(r"./output/images")
    output_dir.mkdir(parents=True, exist_ok=True)
    sampler = MRISampler(input_dir)
    for i in range(n):
        print(i)
        image ,_ , slice_id =  sampler.get_random_sample()
        print(slice_id)
        save_image(image, slice_id, output_dir)

def test_plotting():
    metrics = {
        "metric1": np.random.rand(1000),
        "metric2": np.random.rand(1000),
    }
    output_dir = r"./output"
    metrics_density_plot(metrics, output_dir)
    metrics_boxplot(metrics, output_dir)


def compare_autoencoder_implemetaions():
    param_path = pathlib.Path(r'C:\Users\jan\Documents\python_files\adlm\refactoring\output\mod_siren\mod_siren_2024-06-24_10-02-41\model_checkpoints\images\selected_img\custom_encoder.pth')
    autoencoder1 = build_autoencoder(config)
    autoencoder2 = HardcodedAutoencoder()
    #print(torch.load(param_path))
    autoencoder1.load_state_dict(torch.load(param_path)['state_dict'])
    autoencoder2.load_state_dict(torch.load(param_path)['state_dict'])
    autoencoder1.eval()
    autoencoder2.eval()
    image = torch.rand((100, 1, 32, 32))
    print(torch.equal(autoencoder1(image), autoencoder2(image)))

def compare_encoder_implementaions():
    param_path = pathlib.Path(r'C:\Users\jan\Documents\python_files\adlm\refactoring\output\mod_siren\mod_siren_2024-06-24_10-02-41\model_checkpoints\images\selected_img\custom_encoder.pth')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder1 = CustomEncoder(param_path, device).to(device)
    encoder2 = HardcodedEncoder(param_path, device)
    encoder1.eval()
    encoder2.eval()
    image = torch.rand((100, 1, 32, 32)).to(device)
    print(torch.equal(encoder1(image), encoder2(image)))


def k_space():
    path = pathlib.Path(r"C:\Users\jan\Documents\python_files\adlm\refactoring\output\mod_siren\mod_siren_2024-06-24_10-02-41\model_checkpoints\images\selected_img\file_brain_AXFLAIR_200_6002471.h5")
    acceleration = 6
    center_fraction = 0.05
    plot_k_space(path, acceleration, center_fraction)


def create_sampling_comaprison_image():
    path = pathlib.Path(r"C:\Users\jan\Documents\python_files\adlm\refactoring\output\mod_siren\mod_siren_2024-06-24_10-02-41\model_checkpoints\images\selected_img\file_brain_AXFLAIR_200_6002471.h5")
    accelerations = [2, 4, 6, 8]
    center_fractions = [0.0025, 0.05, 0.1, 0.2]
    output_dir = pathlib.Path(r"./output/images/k_space_low_res")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for acceleration in accelerations:
        for center_fraction in center_fractions:
            image_name = f"k_space_accel_{acceleration}_center_{center_fraction}.png"
            image_path = output_dir / image_name
            plot_k_space(path, acceleration, center_fraction,image_path)

def create_sampling_comparions_image_space():
    from src.data.preprocessing import load_mri_scan
    import matplotlib.pyplot as plt
    # same as create_sampling_comaprison_image but ploting images in the image space instead of k-space
    path = pathlib.Path(r"C:\Users\jan\Documents\python_files\adlm\refactoring\output\mod_siren\mod_siren_2024-06-24_10-02-41\model_checkpoints\images\selected_img\file_brain_AXFLAIR_200_6002471.h5")
    accelerations = [2, 4, 6, 8]
    center_fractions = [0.0025, 0.05, 0.1, 0.2]
    output_dir = pathlib.Path(r"./output/images/image_space_low_res")
    output_dir.mkdir(parents=True, exist_ok=True)
    for acceleration in accelerations:
        for center_fraction in center_fractions:
            image_name = f"img_space_accel_{acceleration}_center_{center_fraction}.png"
            image = load_mri_scan(path, undersampled=True, center_fraction=center_fraction, acceleration=acceleration)[1]
            plt.imshow(image, cmap="gray")
            plt.axis("off")
            plt.savefig(output_dir / image_name, bbox_inches="tight", pad_inches=0, dpi=100)


def apply_high_pass_filter(tensor:torch.Tensor):
    high_pass_filter = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],dtype=tensor.dtype)
    return torch.nn.functional.conv2d(tensor.unsqueeze(0), high_pass_filter.unsqueeze(0).unsqueeze(0), padding=1).squeeze(0)

def apply_high_pass_filter_5x5(tensor:torch.Tensor):
    high_pass_filter = torch.tensor([[0, 0, 1, 0, 0], [0, 1, 2, 1, 0], [1, 2, -16, 2, 1], [0, 1, 2, 1, 0], [0, 0, 1, 0, 0]],dtype=tensor.dtype)
    return torch.nn.functional.conv2d(tensor.unsqueeze(0), high_pass_filter.unsqueeze(0).unsqueeze(0), padding=2).squeeze(0)


def apply_sobel_filter(tensor:torch.Tensor):
    sobel_filter = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]],dtype=tensor.dtype)
    return torch.nn.functional.conv2d(tensor.unsqueeze(0), sobel_filter.unsqueeze(0).unsqueeze(0), padding=1).squeeze(0)

def apply_sharpness_filter(tensor:torch.Tensor):
    sharpness_filter = torch.tensor([[0, -1, 0], [-1, 5, -1], [0, -1, 0]],dtype=tensor.dtype)
    return torch.nn.functional.conv2d(tensor.unsqueeze(0), sharpness_filter.unsqueeze(0).unsqueeze(0), padding=1).squeeze(0)

def crop_tensor(tensor:torch.Tensor, size:int = 250):
    width, height = tensor.shape[-2:]
    width_start = (width - size) // 2
    width_end = width_start + size
    height_start = (height - size) // 2
    height_end = height_start + size
    return tensor[..., width_start:width_end, height_start:height_end]

def visualize_complex_mri(path:pathlib.Path):
    center_fraction = 0.1
    acceleration = 4
    number = 8
    undersampled = True
    image_abs_fully_sampled, k_space_fully_sampled = load_mri_scan(path, undersampled=False, center_fraction=center_fraction, acceleration=acceleration)
    image_abs, k_space_under = load_mri_scan(path, undersampled=undersampled, center_fraction=center_fraction, acceleration=acceleration)
    image_abs2, k_space_under2 = load_mri_scan2(path, undersampled=undersampled, center_fraction=center_fraction, acceleration=acceleration)
    image_real, image_imag = load_mri_scan_complex(path, undersampled=undersampled, center_fraction=center_fraction, acceleration=acceleration)
    image_phase = torch.atan2(image_imag, image_real)
    
    images = [k_space_under[number], k_space_under2[number], k_space_fully_sampled[number], image_abs[number], image_abs2[number], image_abs_fully_sampled[number], apply_sobel_filter(image_abs[number]), apply_sobel_filter(image_abs2[number]), apply_sobel_filter(image_abs_fully_sampled[number])]
    show_batch(torch.stack([crop_tensor(image) for image in images], dim=0), ncols=3)


from torch.utils.data import Dataset
from src.util.timing import time_function

class TestAsyncLoadingDataset(Dataset):
    """Class to find out if the load is async."""
    def __init__(self, n:int, size:int):
        self.n = n
        self.size = size

    def __len__(self):
        return self.n
    
    def __getitem__(self, idx):
        return torch.rand(2, self.size, self.size)
    
import time
def test_async_loading(iterations:int , size:int ):
    starting_time = time.time()
    last_timestamp = time.time()
    from torch.utils.data import DataLoader
    dataset = TestAsyncLoadingDataset(iterations, size)
    dataloader = DataLoader(dataset, batch_size=10, num_workers=10)
    for batch in dataloader:
        #print(time.time() - last_timestamp)
        last_timestamp = time.time()
    print(time.time() - starting_time)



def test_datasets():
    """Compare MRIDataset with MRIDatasetLessRAM using the torch DataLoader
    by just loading images and monito the time it takes."""
    from src.data.mri_dataset import MRIDataset, MRIDatasetLessRAM
    from torch.utils.data import DataLoader
    import time
    
    import numpy as np

    data_root = pathlib.Path(r"C:\Users\jan\Documents\python_files\adlm\data\brain\singlecoil_train\processed_files_complex")
    number_of_samples = 100
    outer_patch_size = 32
    inner_patch_size = 32
    output_dir = pathlib.Path(r"./output")
    acceleration = 6
    center_fraction = 0.1

    # MRIDataset
    start = time.time()
    dataset = MRIDataset(
        data_root,
        number_of_samples=number_of_samples,
        outer_patch_size=outer_patch_size,
        inner_patch_size=inner_patch_size,
        output_dir=output_dir,
        acceleration=acceleration,
        center_fraction=center_fraction,
    )
    dataloader = DataLoader(dataset, batch_size=10, num_workers=1)
    for batch in dataloader:
        for i in range(1000):
            x = torch.rand(1,100,100)   
    print(f"Time for MRIDataset: {time.time() - start}")



    # MRIDatasetLessRAM
    start = time.time()
    dataset = MRIDatasetLessRAM(
        data_root,
        number_of_samples=number_of_samples,
        outer_patch_size=outer_patch_size,
        inner_patch_size=inner_patch_size,
        output_dir=output_dir,
        acceleration=acceleration,
        center_fraction=center_fraction,
    )
    dataloader = DataLoader(dataset, batch_size=10, num_workers=1)
    for batch in dataloader:
        for i in range(1000):
            x = torch.rand(1,100,100)     
    print(f"Time for MRIDatasetLessRAM: {time.time() - start}")




if __name__ == "__main__":
    # Example usage
    #visualize_complex_mri(pathlib.Path(r"C:\Users\jan\Documents\python_files\adlm\data\brain\singlecoil_train\file_brain_AXFLAIR_200_6002425.h5"))
    test_datasets()