# MLX_CLIP Repository 📚🤖

[![GitHub](https://img.shields.io/github/license/harperreed/mlx-clip)](https://github.com/harperreed/mlx-clip/blob/main/LICENSE)

Welcome to the MLX_CLIP repository! 🎉 This repository contains an implementation of the CLIP (Contrastive Language-Image Pre-training) model using the MLX library. CLIP is a powerful model that learns to associate images with their corresponding textual descriptions, enabling various downstream tasks such as image retrieval and zero-shot classification. 🖼️📝

## Repository Structure 🏗️

The repository is structured as follows:

```
mlx_clip/
├── LICENSE
├── README.md
├── assets
│   ├── README.md
├── example.py
├── mlx_clip
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-312.pyc
│   │   ├── convert.cpython-312.pyc
│   │   ├── image_processor.cpython-312.pyc
│   │   ├── model.cpython-312.pyc
│   │   └── tokenizer.cpython-312.pyc
│   ├── convert.py
│   ├── image_processor.py
│   ├── model.py
│   └── tokenizer.py
└── requirements.txt
```

- `mlx_clip`: The main package containing the MLX_CLIP implementation.
  - `__init__.py`: Initializes the `mlx_clip` package and provides a high-level interface for loading and using the CLIP model.
  - `convert.py`: Provides functionality to convert pre-trained CLIP weights from the Hugging Face repository to the MLX format.
  - `image_processor.py`: Implements the image processing pipeline for preparing images to be fed into the CLIP model.
  - `model.py`: Defines the CLIP model architecture and provides methods for loading pre-trained weights.
  - `tokenizer.py`: Implements the tokenization logic for processing text inputs before feeding them into the CLIP model.
- `assets`: Contains sample image assets for testing and demonstration purposes.
- `example.py`: Provides an example script demonstrating how to use the MLX_CLIP model for generating image and text embeddings.
- `requirements.txt`: Lists the required dependencies for running the MLX_CLIP code.

## Getting Started 🚀

To get started with the MLX_CLIP repository, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/harperreed/mlx-clip.git
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Load the pre-trained CLIP model:
   ```python
   from mlx_clip import mlx_clip

   model_dir = "path/to/pretrained/model"
   clip = mlx_clip(model_dir)
   ```

4. Use the CLIP model for generating image and text embeddings:
   ```python
   image_path = "path/to/image.jpg"
   image_embedding = clip.image_ecoder(image_path)

   text = "A description of the image"
   text_embedding = clip.text_encoder(text)
   ```

## Contributing 🤝

Contributions to the MLX_CLIP repository are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request. Make sure to follow the existing code style and provide appropriate documentation for your changes. 📝✨

## License 📜

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments 🙏

This is heavily based on the [mlx-experiments clip implementation](https://github.com/ml-explore/mlx-examples/tree/main/clip). I needed to use it in a project and so made it a simple library. Lot's of gold in there hills.

Feel free to explore the repository and leverage the power of CLIP using MLX! If you have any questions or need further assistance, please don't hesitate to reach out. Happy coding! 😄💻
