# MRI-INR - MRI Image Reconstruction with Modulated SIREN Networks

This is the repository for the Applied Deep Learning in Medicine practical at the chair of AI in Medicine at TUM. 

Group Members:
- Andreas Ehrensberger 
- Jan Rogalka 
- Matteo Wohlrapp


To train the model, simply execute the main.py file. You can see visualizations with `tensorboard --logdir=runs`. 

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

Example configuration files are located in the `./configuration` directory:
- `train.yml` for training setup.
- `test.yml` for testing setup.

## Running the Application
To run the application, use the `main.py` script. You can specify a custom configuration file or use the provided examples. Just make sure to specify the execution mode:

**Training:**
```bash
python main.py -c ./configuration/train.yml
```
You can see visualizations of the training by running `tensorboard --logdir=runs`.

**Testing:**
```bash
python main.py -c ./configuration/test.yml
```

The `-c` flag is used to specify the path to the configuration file. <br>
To run the `autoencoder` encoder, you will need to download the VGG-16 pretrained model on ImageNet from [here](https://github.com/Horizon2333/imagenet-autoencoder/tree/main) and add it under `output/model_checkpoints` in the root of the repository.

## Command-Line Arguments
Use the `-h` option to view all available command-line arguments and their descriptions:

```bash
python main.py -h
```

This will display help information for each argument, including default values and choices where applicable.

## Output
All output files, including saved models and reconstructed images, are stored in a subdirectory withing the `output` directory specified by the `--name` argument within the configuration file. This allows for easy organization and retrieval of results from different runs.


## Detailed Description

Our model aims to generate high-resolution images from their low-resolution counterparts through implicit neural representations. Here's a detailed description of the network's structure and workflow:

#### Input and Tile-Based Processing

- **Input**: The input to the network is a low-resolution image.
- **Tile-based Approach**: The low-resolution image is divided into smaller tiles of size \( \text{tile size} \times \text{tile size} \).
  - This method helps in managing computational load and improving efficiency.

#### Encoding

- **Tile Encoding**: Each tile is encoded into a specified latent dimension.
  - **Latent Dimension**: We typically use a latent dimension of 256 for encoding.
- **Encoding Network**: The encoder network processes each tile to produce a latent vector that captures the essential features of the tile.

#### Modulation

- **Modulator Network**: The latent vector produced by the encoder is then fed into the modulator network.
  - **Function**: The modulator adjusts the weights of the subsequent synthesizer network.
  - This modulation ensures that the weights of the synthesizer are tailored specifically for each tile, based on its encoded features.

#### Synthesis

- **Synthesizer Network**: The synthesizer network receives two types of inputs:
  - **Coordinates**: Coordinates relative to the tile (normalized to be between -1 and 1 for both dimensions).
  - **Modulated Parameters**: Parameters modulated by the modulator network.
- **Activation Function**: The synthesizer uses the sine function as its activation function.
  - **Parameters of Sine Function**: The frequency and other parameters of the sine function are modulated by the output of the modulator network.     

The synthesizer network generates the high-resolution data for each tile based on the input coordinates and the modulated parameters.

#### Training and Reconstruction

- **Training**: During the training phase, the network is optimized to reconstruct the high-resolution tiles from the low-resolution input tiles.      
  - High-resolution images are not directly reconstructed during training; instead, the focus is on individual tiles.

- **Reconstruction**: For visualization and verification, a reconstruction function is used to combine the individual high-resolution tiles back into a complete high-resolution image.

#### Summary of Steps

1. **Low-Resolution Image** is divided into tiles of size \( \text{tile size} \times \text{tile size} \).
2. Each **tile** is encoded into a latent vector.
3. The **encoded vector** is fed into the **modulator network**.
4. The **modulator network** tunes the weights of the **synthesizer network**.
5. The **synthesizer network** receives **coordinates** and **modulated parameters** to generate the high-resolution output for each tile.
6. **Training** focuses on perfecting this process tile-by-tile.
7. **Reconstruction function** assembles the tiles to visualize the complete high-resolution image.
