<p align="center">
  <img src="https://github.com/mesut-by/AnimeFace-WGAN-GP/blob/main/test/last_faces.png?raw=true" width="30%" />
  <img src="https://github.com/mesut-by/AnimeFace-WGAN-GP/blob/main/test/graph.png?raw=true" width="30%" />
  <img src="https://github.com/mesut-by/AnimeFace-WGAN-GP/blob/main/test/last_faces2.png?raw=true" width="30%" />
</p>

---

# Anime Character Face Generation with WGAN-GP

This project aims to generate anime character faces using the **Wasserstein Gradient Penalty (WGAN-GP)** model. The dataset used in this project is the [Anime Face Dataset](https://www.kaggle.com/datasets/splcher/animefacedataset), which contains thousands of anime faces. This project involves building, training, and evaluating the model using **PyTorch** and has been tested on **Colab**.

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

- The images were resized to meet the input requirements of the WGAN-GP model.

---

### Model Architecture

#### Generator
The **Generator** network takes a random noise vector (latent space) and generates a 128x128 RGB image. The model uses transposed convolution layers to gradually upscale the noise into higher resolution images. Additionally, **extraBlock** layers have been added to the generator to increase its capacity. These layers are strategically placed to allow the model to learn finer details.

- **Input**: Random noise vector of size `(batch_size, z_dim, 1, 1)`
- **Output**: Image of size `(batch_size, 3, 128, 128)`
- **Activation Function**: `tanh` (outputs in the [-1, 1] range)

The added **extraBlock** layers are placed after two generator layers (at 8x8 and 16x16 resolutions) and perform the following operations:
- **Conv2d**
- **BatchNorm2d**
- **ReLU**

These extra blocks contribute to the model’s ability to learn more complex features, leading to better overall performance.

#### Critic (Discriminator)
The **Critic** network takes an image as input and predicts how realistic it is. Convolution layers are used to reduce the image size down to a single scalar value representing realism. Additionally, **spectral normalization** has been applied to each Conv2d layer to ensure the model trains more stably.

- **Input**: Image of size `(batch_size, 3, 128, 128)`
- **Output**: Scalar value representing realism
- **Activation Function**: `LeakyReLU`

Spectral normalization ensures that each layer remains Lipschitz continuous, which stabilizes the gradients during training.

#### Gradient Penalty
During training, **gradient penalty** is applied to the Critic model to enforce the Lipschitz constraint, helping prevent issues like mode collapse.

---

## Training

### Important Hyperparameters

- **Learning Rate**: 
  - Generator: 2e-4
  - Critic: 5e-4
- **Batch Size**: 128
- **Image Size**: 128
- **Latent Dimension (z_dim)**: 200
- **Number of Epochs**: 200
- **Gradient Penalty Coefficient**: 10
- **Critic Update Steps**: 5


---

### Training Loop

1. **Critic Training**: The Critic is updated multiple times for each generator step.
   - **Wasserstein Loss**: The difference between the Critic's predictions on real and fake images is calculated.
   - **Gradient Penalty**: A penalty is applied to the Critic's gradients to ensure stability.
   
2. **Generator Training**: The Generator tries to fool the Critic by maximizing the Critic's predictions on fake images. 
   - After **6000 steps**, a noise injection technique called **noise_strength** is introduced. This injects random noise into the fake images to help the generator produce more realistic outputs. This is applied every 20 steps after 6000 steps, with a noise strength of **0.05**.

   ```python
   if cur_step > 6000 and cur_step % 20 == 0:
       noise_strength = 0.05
       noise = torch.randn_like(fake) * noise_strength
       fake = fake + noise
   ```

3. **Logging and Visualization**: Losses and generated images are displayed at each `info_step` step for better tracking of the model's progress.

### Learning Rate Scheduler

A **StepLR scheduler** is applied to both the Generator and Critic. The learning rate is carefully adjusted during training to create a more stable training process:
- The StepLR is **not activated** for the first 3000 steps to allow for quicker stabilization in the early training phases when high loss values are expected.
- After **3000 steps**, the StepLR starts reducing the learning rate in steps to refine the training.
- The learning rate is **deactivated** after **15,000 steps** to prevent the learning rate from falling too low (close to 1e-6).

```python
gen_scheduler = torch.optim.lr_scheduler.StepLR(gen_opt, step_size=3000, gamma=0.6)
crit_scheduler = torch.optim.lr_scheduler.StepLR(crit_opt, step_size=2500, gamma=0.7)
```

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
}, f"{root_path}gen_model.pth")
```

To load a checkpoint:
```python
checkpoint = torch.load(f"{root_path}gen_model.pth")
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
