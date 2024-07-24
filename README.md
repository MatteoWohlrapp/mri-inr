# MRI-INR - MRI Image Reconstruction with Modulated SIREN Networks

This is the repository for the Applied Deep Learning in Medicine practical at the chair of AI in Medicine at TUM. 

Group Members:
- Andreas Ehrensberger 
- Jan Rogalka 
- Matteo Wohlrapp



## Project Overview
This project implements implicit neural representations for MRI images using modulated SIREN (Sinusoidal Representation Networks). It is designed to reconstruct high-quality MRI images from undersampled k-space data. The network is based on work from 
Mehta et al. [1], and existing implementations [2].

The repository is structured as follows: <br> 
In the `src` folder are all the necessary files to train and test a model. Under `src/data`, everything related to the dataset and data loading can be found. `src/train` contains the trainer and auxiliary functions, while `src/networks` includes the actual neural network. The folder `src/configuration` contains information about the argument parsing, and `src/utils` defines helper functions. In `configuration` under the root directory, you can find several predefined training and testing configurations, including for the experiments we conducted. 

### Results 
We ran experiments to test different model configurations and evaluated our baseline model on various k-space sampling densities. All our models were trained on around 3000 single-coil simulated fastMRI [3] flair brain scans. The tests were performed on a validation set of flair images from the fastMRI dataset as well. We calculated the metrics for the results based on all available 940 files. 

### Ablations 
In total, six different ablations were tested: 
- Baseline: Implementation with a pre-trained encoder trained on the fastMRI brain scans
- Edge: Introducing an additional edge loss to the model based on a Sobel filter 
- VGG: Using a VGG encoder trained on ImageNet instead of our own trained one
- Morlet: Exchanging the sine-based activation functions with Morlet-based ones
- Perceptual: Using a perceptual loss based on a VGG encoder. The encoder for the perceptual loss can be downloaded [here](https://drive.google.com/drive/folders/1EtIOmlIY6Ts-GZ9rHm28EGMkKKuOzT1G?usp=share_link). Make sure you put it under a `model` folder in the root directory. 
- Residual: Adding residual connections to the MLP layers and increasing the depth of the network while reducing the latent dimension

The configuration for each specific experiment can be found under `configuration/ablations`. To run and test the residual connections, you will need to check out to the `residual-connections` branch. The trained models can be downloaded from [here](https://drive.google.com/drive/folders/1xa68eJXUBpLyakrB4nkHdmRmEen8fbkm?usp=sharing).


#### Quantitative Results
The quantitative results can be found below.
TODO: Add actual values
| Configuration | PSNR Mean | PSNR Std | PSNR Min | PSNR Max | SSIM Mean | SSIM Std | SSIM Min | SSIM Max | NRMSE Mean | NRMSE Std | NRMSE Min | NRMSE Max |
|---------------|-----------|----------|----------|----------|-----------|----------|----------|----------|-------------|-----------|-----------|-----------|
| Baseline      | 30.2      | 0.5      | 29.5     | 31.0     | 0.89      | 0.01     | 0.87     | 0.91     | 0.032       | 0.004     | 0.030     | 0.038     |
| Edge          | 32.1      | 0.4      | 31.6     | 32.7     | 0.91      | 0.02     | 0.89     | 0.93     | 0.028       | 0.003     | 0.025     | 0.031     |
| VGG           | 28.9      | 0.6      | 28.0     | 29.8     | 0.87      | 0.02     | 0.85     | 0.90     | 0.035       | 0.005     | 0.032     | 0.040     |
| Morlet        | 29.5      | 0.7      | 28.7     | 30.5     | 0.88      | 0.01     | 0.86     | 0.90     | 0.033       | 0.004     | 0.030     | 0.037     |
| Perceptual    | 29.5      | 0.7      | 28.7     | 30.5     | 0.88      | 0.01     | 0.86     | 0.90     | 0.033       | 0.004     | 0.030     | 0.037     |
| Residual      | 29.5      | 0.7      | 28.7     | 30.5     | 0.88      | 0.01     | 0.86     | 0.90     | 0.033       | 0.004     | 0.030     | 0.037     |

#### Qualitative Results
The qualitative results can be found below.
TODO: Add images

### Sampling density 
The fastMRI framework allows different k-space masks to be set, which results in different sampling densities. In this experiment, we tested different accelerations, higher numbers specifying less image information being retained, and center fractions, which means how much information is kept in the center of the k-space. In total, four different variations were tested: 
- Acceleration 8, center fraction 0.05 
- Acceleration 6, center fraction 0.05 
- Acceleration 6, center fraction 0.1 
- Acceleration 4, center fraction 0.2 

The configuration for each specific experiment can be found under `configuration/configuration`. The trained models can be downloaded from [here](https://drive.google.com/drive/folders/1xa68eJXUBpLyakrB4nkHdmRmEen8fbkm?usp=sharing)

#### Quantitative Results
The quantitative results can be found below.
TODO: Add actual values
| Configuration   | PSNR Mean | PSNR Std | PSNR Min | PSNR Max | SSIM Mean | SSIM Std | SSIM Min | SSIM Max | NRMSE Mean | NRMSE Std | NRMSE Min | NRMSE Max |
|-----------------|-----------|----------|----------|----------|-----------|----------|----------|----------|-------------|-----------|-----------|-----------|
| Acc 8, Cf 0.005 | 30.2      | 0.5      | 29.5     | 31.0     | 0.89      | 0.01     | 0.87     | 0.91     | 0.032       | 0.004     | 0.030     | 0.038     |
| Acc 6, Cf 0.005 | 30.2      | 0.5      | 29.5     | 31.0     | 0.89      | 0.01     | 0.87     | 0.91     | 0.032       | 0.004     | 0.030     | 0.038     |
| Acc 6, Cf 0.01  | 30.2      | 0.5      | 29.5     | 31.0     | 0.89      | 0.01     | 0.87     | 0.91     | 0.032       | 0.004     | 0.030     | 0.038     |
| Acc 4, Cf 0.01  | 30.2      | 0.5      | 29.5     | 31.0     | 0.89      | 0.01     | 0.87     | 0.91     | 0.032       | 0.004     | 0.030     | 0.038     |


#### Qualitative Results
The qualitative results can be found below.
TODO: Add images

### Models 
You can download the models from our result section [here](https://drive.google.com/drive/folders/1xa68eJXUBpLyakrB4nkHdmRmEen8fbkm?usp=sharing) to continue training or run your own evaluation. The configuration files for the respective models can be found under `configuration`. 

## Installation
Ensure that you have Python installed on your system (Python 3.8+ recommended). Install all required dependencies by running:

```bash
pip install -r requirements.txt
```

This will install all necessary Python packages as specified in the `requirements.txt` file.

## Configuration
The project uses YAML files for configuration to specify parameters for training and testing. Modify these files to adjust various parameters like dataset paths, network architecture, learning rates, etc. Alternatively, you can adjust all of the parameters through command line arguments. 

Example configuration files are located in the `/configuration` directory:
- `train_modulated_siren.yml` for training setup.
- `test_modulated_siren.yml` for testing setup.

## Dataset Preparation Guide 
<!-- TODO still needs to be changed when the configuration file for this is added -->
This section provides a comprehensive guide on preparing the dataset for both training and testing purposes. This is only necessary before running for the first time. Follow the steps below to ensure your dataset is correctly set up and ready for use.

### Step 1: Specify Dataset Location
- Initially, the script is configured with a placeholder path for the dataset.
- **Action Required:** Update the script with the actual path where your dataset is located. This ensures the script can access and process the dataset correctly.

### Step 2: Configure Mask Parameters
- The dataset preparation involves applying masks to the data. These masks are defined by a list of tuples, with each tuple containing two key parameters:
  1. **Center Fraction:** Specifies the fraction of low-frequency k-space data to retain.
  2. **Acceleration:** Determines the rate at which data is undersampled.
- **Action Required:** Add the mask parameters to the script. Each tuple in the list specifies one mask configuration to be applied.

### Step 3: Data Transformation and Storage
- The script processes .h5 files in the dataset by iterating over each scan.
- Each image within a scan is treated as a separate entity and is transformed from k-space to image space and normalized to [0,1].
- The transformed images are saved in the specified location, ready for further processing or training.

### Step 4: CSV File Generation
- Upon completion of the data transformation process, a CSV file is generated.
- This CSV file contains essential metadata about the created files, including file locations.
- **Utility:** The CSV file serves as a directory, enabling efficient file location, filtering, and access during training or testing phases.

### Final Notes
- Ensure all the specified paths and parameters in the script are correctly set before running the dataset preparation process.
- The prepared dataset, now in image space and accompanied by a comprehensive CSV file, is ready for use in your machine learning models for training and testing purposes.
```bash
python preprocessing_script.py -p <path to folder with the original data>
```

## Encoder 
To train a custom autoencoder you the train_encoder.py script. The basic configuration is already set up in the `train_encoder.yml` file. You can adjust the parameters in the file. The parameters are like the ones used for training the SIREN network.

If you don't want to train your own encoder, or want to use the VGG encoder, you can download the used encoders from [here](https://drive.google.com/drive/folders/1pJXmrPyM-sMoYMpeX0dgH1sxZj2EhTcD?usp=sharing).

## Running the Application
To run the application, use one of the two main scripts. You can specify a custom configuration file or use the provided examples.

**Training:**
```bash
python train_mod_siren.py --config configuration/train_modulated_siren.yml
```
In order to ensure training works, it is necessary to specify the correct encoder path. You can see visualizations of the training by running `tensorboard --logdir={output_folder}/runs`. Because of the complicated tiling and processing work necessary, we opted to load all the data samples into memory. If that is not possible, you can switch the dataset implementation to the `MRIDatasetLowMemory` class in `src/data/mri_dataset.py`.

**Testing:**
```bash
python test_mod_siren.py --config configuration/test_modulated_siren.yml
```
When testing, make sure to specify the correct folder and model path. Then, a `test` subfolder will be automatically generated in the target folder. You can sepcify the number of visual samples in the configuration file. In addition, you can specify test files if you want to visualize specific files. You can also add the number of samples where the metrics are calculated on. When no number of metrics samples is specified, the whole dataset is used. 

## Configuration parameters 
There is a number of configuration parameters for both training and testing. They all have default values, that can be found `src/configuration/configuration.py`. Below, you can find a list of all parameters you can modify. 

### Common Parameters for Both Training and Testing

- **model**
  - `dim_in`: Input dimension of the model.
  - `dim_hidden`: Dimension of hidden layers.
  - `dim_out`: Output dimension of the model.
  - `latent_dim`: Dimension of the latent space.
  - `num_layers`: Number of layers in the model.
  - `w0`: The omega_0 parameter for SIREN.
  - `w0_initial`: Initial value of omega_0.
  - `use_bias`: Boolean flag to use bias in layers.
  - `dropout`: Dropout rate.
  - `encoder_type`: Type of encoder used. Available options are 'vgg' or 'custom'
  - `encoder_path`: Path to a custom encoder model.
  - `outer_patch_size`: Size of the outer patch in the input.
  - `inner_patch_size`: Size of the inner patch in the input.
  - `siren_patch_size`: Size of the patch the siren network is actually trained on.
  - `activation`: The type of activation functions used. Options are `sine` and `morlet`. 

### Training Configuration

- **data**
  - `train`
    - `dataset`: Path to the training dataset.
    - `num_samples`: Number of samples to use from the training dataset.
    - `mri_type`: Type of MRI images (e.g., FLAIR).
    - `num_workers`: Number of workers for data loading.
  - `val`
    - `dataset`: Path to the validation dataset.
    - `num_samples`: Number of samples to use from the validation dataset.
  - `acceleration`: Acceleration rate used for the data.
  - `center_fraction`: Fraction of the center that is sampled. 

- **training**
  - `lr`: Learning rate.
  - `batch_size`: Batch size.
  - `epochs`: Total number of epochs to train.
  - `output_dir`: Directory to save output files.
  - `output_name`: Base name for output files.
  - `optimizer`: Type of optimizer to use (`Adam`, `SGD`, etc.).
  - `logging`: Specify if tensorboard should be turned on or off.
  - `criterion`: Specify which criterion to use, options are `MSE`, `Perceptual`, and `Edge`.
  - `model`
    - `continue_training`: Boolean to indicate whether to continue training from a previous checkpoint.
    - `model_path`: Path to the model checkpoint for resuming training. If not specified but continue training, the last model with the same name is searched. 
    - `optimizer_path`: Path to the optimizer checkpoint. If not specified but continue training, the last model with the same name is searched. 

### Testing Configuration

- **data**
  - `dataset`: Path to the testing dataset.
  - `visual_samples`: Number of samples that are visualized. 
  - `metric_samples`: Number of samples where metrics are calculated on.
  - `test_files`: Specific files to test within the dataset.
  - `acceleration`: Acceleration rate used for the data.
  - `center_fraction`: Fraction of the center that is sampled. 

- **testing**
  - `output_dir`: Base directory of the output files.
  - `output_name`: Specific folder within the base directory.
  - `model_path`: Path to the model file for testing.

## Output
All output files, including saved models and reconstructed images, are stored in a subdirectory within the `output_name` directory specified by the `output_dir` argument within the configuration file. This allows for easy organization and retrieval of results from different runs. You can find a folder for model checkpoints, snapshots of the current training and validation results, tensorboard logs, a copy of the configuration file, information on which files you trained the model on, and a progress overview. When testing, you will get an additional `test` folder, which then includes visual samples, a CSV file, and a summary of the results. 


## Sources
1. Mehta, I., Gharbi, M., Barnes, C., Shechtman, E., Ramamoorthi, R., & Chandraker, M. (2021). Modulated Periodic Activations for Generalizable Local Functional Representations. 2021 IEEE/CVF International Conference on Computer Vision (ICCV), 14194-14203. 
2. https://github.com/lucidrains/siren-pytorch
3. https://fastmri.med.nyu.edu
