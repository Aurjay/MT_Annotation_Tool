import os
import numpy as np
from PIL import Image
from pathlib import Path

def add_gaussian_noise(image, mean=0, std=25):
    """
    Adds Gaussian noise to an image uniformly across the entire image.
    
    Parameters:
    - image: PIL.Image object
    - mean: Mean of the Gaussian noise
    - std: Standard deviation of the Gaussian noise
    
    Returns:
    - Noisy image as a PIL.Image object
    """
    np_image = np.array(image, dtype=np.float32)
    noise = np.random.normal(mean, std, np_image.shape).astype(np.float32)
    noisy_image = np_image + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)  # Ensure pixel values are within [0, 255]
    return Image.fromarray(noisy_image)

def process_images(input_folder, output_folder, noise_mean=0, noise_std=25):
    """
    Process all images in the input folder, adding Gaussian noise to each and saving to the output folder.
    
    Parameters:
    - input_folder: Path to the input folder containing images
    - output_folder: Path to the output folder where noisy images will be saved
    - noise_mean: Mean of the Gaussian noise
    - noise_std: Standard deviation of the Gaussian noise
    """
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    for image_path in input_folder.glob('*.*'):
        try:
            image = Image.open(image_path).convert('RGB')  # Ensure the image is in RGB format
            noisy_image = add_gaussian_noise(image, noise_mean, noise_std)
            noisy_image_path = output_folder / image_path.name
            noisy_image.save(noisy_image_path)
            print(f"Processed {image_path.name}")
        except Exception as e:
            print(f"Could not process {image_path.name}: {e}")



if __name__ == "__main__":
    input_folder = r'Path to input folder'  
    output_folder = r'Path to output folder'  
    noise_mean = 0  
    noise_std = 60  
    process_images(input_folder, output_folder, noise_mean, noise_std)
