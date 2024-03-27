# tests/test_mlx_clip.py

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mlx_clip import mlx_clip
from mlx_clip.convert import convert_weights
from mlx_clip.image_processor import CLIPImageProcessor
from mlx_clip.model import CLIPModel
from mlx_clip.tokenizer import CLIPTokenizer
import mlx.core as mx

# Helpers

def create_dummy_image(size=(224, 224)):
    image = Image.new("RGB", size, "white")
    return image

def create_temp_dir():
    return tempfile.mkdtemp()

# Tests

def test_convert_weights():
    hf_repo = "openai/clip-vit-base-patch32"
    with tempfile.TemporaryDirectory() as temp_dir:
        convert_weights(hf_repo, temp_dir)
        assert len(os.listdir(temp_dir)) > 0

def test_clip_image_processor():
    image_processor = CLIPImageProcessor()
    image = create_dummy_image()
    processed_image = image_processor([image])
    assert processed_image.shape == (1, 224, 224, 3)

def test_clip_tokenizer():
    with tempfile.TemporaryDirectory() as temp_dir:
        convert_weights("openai/clip-vit-base-patch32", temp_dir)
        tokenizer = CLIPTokenizer.from_pretrained(temp_dir)
        text = "This is a test sentence."
        tokens = tokenizer(text)
        assert len(tokens) > 0

def test_clip_model():
    with tempfile.TemporaryDirectory() as temp_dir:
        convert_weights("openai/clip-vit-base-patch32", temp_dir)
        model = CLIPModel.from_pretrained(temp_dir)

        image = create_dummy_image()
        image_processor = CLIPImageProcessor()
        processed_image = image_processor([image])

        tokenizer = CLIPTokenizer.from_pretrained(temp_dir)
        text = "This is a test sentence."
        tokens = tokenizer(text)

        output = model(input_ids=mx.array(tokens).reshape(1, -1), pixel_values=processed_image)
        assert output.text_embeds is not None
        assert output.image_embeds is not None

def test_mlx_clip_end_to_end():
    hf_repo = "openai/clip-vit-base-patch32"
    with tempfile.TemporaryDirectory() as temp_dir:
        convert_weights(hf_repo, temp_dir)
        clip = mlx_clip(temp_dir)

        image_path = "assets/cat.jpeg"
        image_embeddings = clip.image_encoder(image_path)
        assert len(image_embeddings) > 0

        text = "a photo of a cat"
        text_embeddings = clip.text_encoder(text)
        assert len(text_embeddings) > 0
