from PIL import Image
import numpy as np
from scipy import ndimage
from skimage.filters import unsharp_mask
def resize_image(img, size=32):
    _img = img.copy()
    _img = _img.resize((size, size))
    return _img

def unsharpmaskk(img):
  _img = img.copy()

  blurred = ndimage.gaussian_filter(_img, sigma=1)
  unsharp_maskk = _img - 0.5 * blurred

  threshold_value = unsharp_mask(unsharp_maskk, radius=5, amount=2)
  binary_image = unsharp_maskk > threshold_value

  # Convert the binary image to uint8 and scale it to 0-255
  binary_image = binary_image.astype(np.float32)
  return _img  

# Replace 'input_image_path.jpg' with the path to your input image
input_image_path = 'aha.png'

# Load the input image
input_img = Image.open(input_image_path)

# Define the desired size for the resized image (e.g., 64x64)
desired_size = 32

# Resize the image using the resize_image function
resized_img = resize_image(input_img, size=desired_size)
unsharpmas = unsharpmaskk(resized_img)
# Convert the image to RGB mode (remove alpha channel)
resized_img = resized_img.convert("RGB")
unsharpmas = unsharpmas.convert("RGB")
# Replace 'output_image_path.jpg' with the path where you want to save the resized image
output_image_path = 'output_image_path.jpg'
output_unsharpmas = 'unsharpmas.jpg'
# Save the resized image as JPEG
resized_img.save(output_image_path)
unsharpmas.save(output_unsharpmas)
print("Resized image saved successfully!")
