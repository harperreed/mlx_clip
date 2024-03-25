
import mlx_clip

clip = mlx_clip.mlx_clip("mlx_model")

imemb = clip.generate_image_embedding("assets/cat.jpg")
print(imemb)

txemb = clip.generate_text_embedding( "a photo of a cat")
print(txemb)
