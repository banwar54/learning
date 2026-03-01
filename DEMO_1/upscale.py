from realesrgan import RealESRGAN
from PIL import Image
import torch

# Select device (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = RealESRGAN(device, scale=2)  # You can use scale=2 or 4
model.load_weights('RealESRGAN_x4.pth', download=True)

# Load image
image = Image.open('DEMO_1/input.jpg').convert('RGB')

# Upscale
upscaled_image = model.predict(image)

# Save result
upscaled_image.save('DEMO_1/output_upscaled.jpg')

print("Upscaling complete!")