# MLX_CLIP 📚🤖

[![GitHub](https://img.shields.io/github/license/harperreed/mlx-clip)](https://github.com/harperreed/mlx-clip/blob/main/LICENSE)

Welcome to the MLX_CLIP repository! 🎉 This repository contains an implementation of the CLIP (Contrastive Language-Image Pre-training) model using the MLX library. CLIP is a powerful model that learns to associate images with their corresponding textual descriptions, enabling various downstream tasks such as image retrieval and zero-shot classification. 🖼️📝

## Features ✨

- Easy-to-use MLX_CLIP model for generating image and text embeddings
- Support for loading pre-trained CLIP weights from Hugging Face
- Efficient conversion of weights to MLX format for optimal performance
- Seamless integration with the MLX library for accelerated inference on Apple Silicon devices

## Getting Started 🚀

To get started with MLX_CLIP, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/harperreed/mlx_clip.git
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
   image_embedding = clip.image_encoder(image_path)

   text = "A description of the image"
   text_embedding = clip.text_encoder(text)
   ```



## Examples 💡

Check out the `example.py` file for a simple example of how to use MLX_CLIP to generate image and text embeddings.

## Model Conversion 🔄

MLX_CLIP provides a convenient utility to convert pre-trained CLIP weights from the Hugging Face repository to the MLX format. To convert weights, use the `convert_weights` function from `mlx_clip.convert`:

```python
from mlx_clip.convert import convert_weights

hf_repo = "openai/clip-vit-base-patch32"
mlx_path = "path/to/save/converted/model"
convert_weights(hf_repo, mlx_path)
```

## Contributing 🤝

Contributions to MLX_CLIP are welcome! If you encounter any issues, have suggestions for improvements, or want to add new features, please open an issue or submit a pull request. Make sure to follow the existing code style and provide appropriate documentation for your changes.

## License 📜

MLX_CLIP is licensed under the [MIT License](LICENSE).

## Acknowledgments 🙏

MLX_CLIP is heavily based on the [mlx-experiments clip implementation](https://github.com/ml-explore/mlx-examples/tree/main/clip). Special thanks to the MLX team for their incredible work!

## Contact 📞

For any questions or inquiries, feel free to reach out to the project maintainer:

Harper Reed
- Email: harper@modest.com
- GitHub: [harperreed](https://github.com/harperreed)

Happy coding with MLX_CLIP! 😄💻🚀
