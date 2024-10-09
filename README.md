
---

# Anime Character Face Generation with WQGAN-GP(In Progress)

This project aims to generate anime character faces using the **Wasserstein Kantorovich Gradient Penalty (WQGAN-GP)** model. The dataset used in this project is the [Anime Face Dataset](https://www.kaggle.com/datasets/splcher/animefacedataset), which contains thousands of anime faces. This project involves building, training, and evaluating the model using **PyTorch** and has been tested on **Colab**.

## Project Overview

The main goal of this project is to generate high-quality anime face images using the **WGAN-GP** model. WGAN-GP aims to solve issues common in traditional GAN models, such as mode collapse and vanishing gradients. This project makes it possible to generate anime faces by improving the stability of the GAN model.

### Key Features:
- **WGAN-GP Implementation**: The model is trained using the Wasserstein loss along with gradient penalty, ensuring the Lipschitz constraint during training.
- **Custom Dataset Class**: Processes the anime face images into a suitable format for GAN training.
- **PyTorch Usage**: The model is built and trained using the PyTorch framework.
- **Colab Notebook**: The project is run on Google Colab for fast prototyping and GPU support.

---

## Installation

To run the project, the following dependencies are required:

1. **PyTorch**: For model building and training
2. **torchvision**: For image transformations
3. **NumPy**, **PIL**: For data processing

### Installing Dependencies:
You can install the dependencies in your Colab environment using the following commands:
```bash
pip install torch torchvision numpy pillow
```

### Downloading the Dataset with Kaggle API

To download the Kaggle dataset, you need to upload your Kaggle API key to the Colab environment.

1. Upload your Kaggle API key:
   ```python
   from google.colab import files
   files.upload()
   ```

2. Move the key file to the correct directory:
   ```bash
   !mkdir -p ~/.kaggle
   !mv kaggle.json ~/.kaggle/
   !chmod 600 ~/.kaggle/kaggle.json
   ```

3. Download and unzip the dataset:
   ```bash
   !kaggle datasets download -d splcher/animefacedataset
   !unzip animefacedataset.zip -d animefacedataset
   ```

---

## Dataset

The dataset used in this project is the **Anime Face Dataset**, available on Kaggle: [Anime Face Dataset](https://www.kaggle.com/datasets/splcher/animefacedataset).

- The dataset contains thousands of images of anime character faces in various pixel dimensions.

**License**: Database Contents License (DbCL) v1.0.

### Preprocessing

- To maintain consistency during training, images with a pixel range of 100x140 have been filtered and processed.

- The images were resized and normalized to meet the input requirements of the WGAN-GP model. The images were converted to tensors and scaled to the [-1, 1] range to fit the `tanh` activation function.

---

## Model Architecture

### Generator

The **Generator** network takes a random noise vector (latent space) and generates a 128x128 RGB image. The model uses transposed convolution layers to gradually upscale the noise into higher resolution images.

- **Input**: Random noise vector of size `(batch_size, z_dim, 1, 1)`
- **Output**: Image of size `(batch_size, 3, 128, 128)`
- **Activation Function**: `tanh` (outputs in the [-1, 1] range)

### Critic (Discriminator)

The **Critic** network takes an image as input and predicts how realistic it is. It uses convolution layers to reduce the image size to a single value representing the Wasserstein distance.

- **Input**: Image of size `(batch_size, 3, 128, 128)`
- **Output**: Scalar value representing realism
- **Activation Function**: `LeakyReLU`

### Gradient Penalty

During training, **gradient penalty** is applied to the Critic model to maintain the Lipschitz constraint, helping to prevent issues like mode collapse.

---

## Training

### Important Hyperparameters

- **Learning Rate**: 
  - Generator: 2e-4
  - Critic: 4e-4
- **Batch Size**: 128
- **Latent Dimension (z_dim)**: 200
- **Number of Epochs**: 150
- **Gradient Penalty Coefficient**: 10
- **Critic Update Steps**: 5

### Training Loop

1. **Critic Training**: The Critic is updated multiple times for each generator step.
   - **Wasserstein Loss**: The difference between the Critic's predictions on real and fake images is calculated.
   - **Gradient Penalty**: A penalty is applied to the Critic's gradients to ensure stability.

2. **Generator Training**: The Generator tries to fool the Critic by maximizing the Critic's predictions on fake images.

3. **Logging and Visualization**: Losses and generated images are displayed at each info_step step.

### Learning Rate Scheduler

A StepLR scheduler is used for both the Generator and Critic. The learning rate is reduced at specific steps to provide a more stable training environment. In the early stages of training, where high loss values are expected, the StepLR scheduler is not activated to stabilize the environment. However, as training progresses and the model focuses on finer details, the StepLR is deactivated after a certain step to prevent the learning rate from decreasing too much.

---

## Results

The model generates anime character faces from random noise vectors. The images and losses produced during training are visualized and logged.

---

## Checkpoint

To avoid losing progress during training, checkpoints are saved at specific steps. The weights of both the generator and critic models, as well as the optimizer states, are saved.

```python
torch.save({
    'epoch': epoch,
    'model_state_dict': gen.state_dict(),
    'optimizer_state_dict': gen_opt.state_dict(),
}, f"{root_path}G-{name}.pkl")
```

To load a checkpoint:
```python
checkpoint = torch.load(f"{root_path}G-{name}.pkl")
gen.load_state_dict(checkpoint['model_state_dict'])
gen_opt.load_state_dict(checkpoint['optimizer_state_dict'])
```

---

## License

The dataset used in this project is licensed under the **Database Contents License (DbCL) v1.0**.

The code is licensed under the **MIT License** and is freely available for use and modification.

---

## Contact

If you have any questions or feedback, feel free to reach out via [GitHub](https://github.com/mesut-by).

---
