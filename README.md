# MRI-INR - MRI Image Reconstruction with Modulated SIREN Networks

This is the repository for the Applied Deep Learning in Medicine practical at the chair of AI in Medicine at TUM. 

Group Members:
- Andreas Ehrensberger 
- Jan Rogalka 
- Matteo Wohlrapp



## Project Overview
This project implements implicit neural representations for MRI images using modulated SIREN (Sinusoidal Representation Networks). It is designed to reconstruct high-quality images from undersampled k-space images.

## Installation
Ensure that you have Python installed on your system (Python 3.8+ recommended). Install all required dependencies by running:

```bash
pip install -r requirements.txt
```

This will install all necessary Python packages as specified in the `requirements.txt` file.

## Configuration
The project uses YAML files for configuration to specify parameters for training and testing. Modify these files to adjust various parameters like dataset paths, network architecture, learning rates, etc. Alternatively, you can adjust all of the parameters through command line arguments. 

Example configuration files are located in the `src/configuration` directory:
- `train_modulated_siren.yml` for training setup.
- `test_modulated_siren.yml` for testing setup.

## Running the Application
To run the application, use one of the two main scripts. You can specify a custom configuration file or use the provided examples.

**Training:**
```bash
python train_mod_siren.py --config src/configuration/train_modulated_siren.yml
```
You can see visualizations of the training by running `tensorboard --logdir={output_folder}/runs`.

**Testing:**
```bash
python test_mod_siren.py --config src/configuration/test_modulated_siren.yml
```

The `--config` flag is used to specify the path to the configuration file. <br>

## Configuration parameters 
There is a number of configuration parameters for both training and testing. 

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
  - `encoder_type`: Type of encoder used.
  - `encoder_path`: Path to a custom encoder model.
  - `outer_patch_size`: Size of the outer patch in the input.
  - `inner_patch_size`: Size of the inner patch in the input.

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

- **training**
  - `lr`: Learning rate.
  - `batch_size`: Batch size.
  - `epochs`: Total number of epochs to train.
  - `limit_io`: Boolean to limit I/O operations.
  - `output_dir`: Directory to save output files.
  - `output_name`: Base name for output files.
  - `optimizer`: Type of optimizer to use (`Adam`, `SGD`, etc.).
  - `model`
    - `continue_training`: Boolean to indicate whether to continue training from a previous checkpoint.
    - `model_path`: Path to the model checkpoint for resuming training.
    - `optimizer_path`: Path to the optimizer checkpoint.

### Testing Configuration

- **data**
  - `dataset`: Path to the testing dataset.
  - `num_samples`: Number of samples to use from the testing dataset.
  - `test_files`: Specific files to test within the dataset.

- **testing**
  - `output_dir`: Directory to save output files.
  - `output_name`: Base name for output files.
  - `model_path`: Path to the model file for testing.

## Output
All output files, including saved models and reconstructed images, are stored in a subdirectory withing the `output_name` directory specified by the `output_dir` argument within the configuration file. This allows for easy organization and retrieval of results from different runs.