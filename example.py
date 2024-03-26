import mlx_clip

# Initialize the mlx_clip model with the given model name.
clip = mlx_clip.mlx_clip("mlx_model")

# Encode the image from the specified file path and obtain the image embeddings.
# The embeddings are a numerical representation of the image content.
image_embeddings = clip.image_ecoder("assets/cat.jpeg")
# Print the image embeddings to the console.
#print(image_embeddings)

# Encode the text description and obtain the text embeddings.
# The embeddings are a numerical representation of the textual description.
text_embeddings = clip.text_encoder("a photo of a cat")
# Print the text embeddings to the console.
print(text_embeddings)
