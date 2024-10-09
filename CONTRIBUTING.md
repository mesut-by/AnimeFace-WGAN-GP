# Contributing to AnimeFaceGeneration

Thank you for considering contributing to this project! This project aims to generate anime character faces using WQGAN-GP and is currently in development. The project is coded for Colab and uses the "splcher/animefacedataset" from Kaggle. Contributions are welcome, especially in improving the model or optimizing the training process.

## How Can You Contribute?

1. **Fork the Repository**: Start by forking the repository to your own GitHub account.
   ```bash
   git clone https://github.com/mesut-by/AnimeFaceGeneration
   ```

2. **Create a New Branch**: Create a new branch for your changes.
   ```bash
   git checkout -b feature-name
   ```

3. **Make Your Changes**: Work on your improvements. Consider focusing on:
   - Optimizing the WQGAN-GP model architecture.
   - Improving the training speed and stability.
   - Experimenting with different loss functions or adding regularization techniques.

4. **Test Your Changes**: If applicable, test your changes thoroughly to ensure they improve the model or project.
   ```bash
   # If tests are available
   pytest
   ```

5. **Commit Your Changes**: Use a clear and descriptive commit message.
   ```bash
   git commit -m "Add feature: Improved model training speed"
   ```

6. **Push to Your Fork**: Push your changes to your repository.
   ```bash
   git push origin feature-name
   ```

7. **Submit a Pull Request**: Create a pull request (PR) from your branch to this repository’s `main` branch. Provide a detailed description of the changes you've made.

## Guidelines for Contributions

- **Dataset**: We are using the "splcher/animefacedataset" from Kaggle. Please do not upload the dataset directly. Instead, contributors can download it from [Kaggle](https://www.kaggle.com/datasets/splcher/animefacedataset).
- **Coding Style**: Follow the project's coding style for consistency. Colab is being used for this project, so ensure your contributions are compatible with Colab notebooks.
- **Testing**: If you make changes to the model or training script, ensure they are well-tested. If you can, provide Colab links for easy testing by others.
- **Pull Requests**: Pull requests should be concise and focused on a single feature or improvement. If you plan to make multiple changes, it's better to open separate pull requests for each.

## Areas Where Help is Needed

- Model performance improvement: Better training stability or higher-quality anime faces.
- Loss function experimentation: Try out different loss functions or add new layers (e.g., residual blocks).
- Feature matching loss, self-supervision, or spectral normalization improvements.
  
## Contact Information

If you have any questions or need further clarification, feel free to open an issue in the repository or contact with me.
```
