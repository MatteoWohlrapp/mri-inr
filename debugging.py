import pathlib
from src.util.visualization import metrics_density_plot, metrics_boxplot, save_image
from src.data.mri_sampler import MRISampler
import numpy as np
from src.networks.encoding.custom_mri_encoder import config, build_autoencoder, CustomEncoder
from src.networks.encoding.new_encoder import HardcodedAutoencoder, HardcodedEncoder
import torch

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

if __name__ == "__main__":
    compare_encoder_implementaions()