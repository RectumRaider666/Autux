#!/usr/bin/env python3

from pathlib import Path
from tqdm import tqdm
from PIL import Image
from math import gcd
import numpy as np
import os

# All supported Read & Write image Formats

SUPPORTED = {[
  {"AVIF": [".avif"]}, {"BMP": [".bmp"]}, {"DDS": [".dds"]}, {"DIB": [".dib"]}, {"EPS": [".eps"]}, {"GIF": [".gif"]},
  {"ICNS": [".icns"]}, {"ICO": [".ico"]}, {"JPEG": [".jfif", ".jpe", ".jpeg", ".jpg"]}, {"MPO": [".mpo"]}, 
  {"PCX": [".pcx"]}, {"PNG": [".png"]}, {"PBM": [".pbm"]}, {"PGM": [".pgm"]}, {"PPM": [".ppm"]}, {"QOI": [".qoi"]},
  {"SGI": [".sgi"]}, {"TGA": [".tga"]}, {"TIFF": [".tif", ".tiff"]}, {"WebP": [".webp"]}, {"XBM": [".xbm"]}, {"BLP": [".blp"]},
  {"IM": [".im"]}, {"MSP": [".msp"]}, {"SPIDER": [".spi"]}
]}
SUPPORTED_FORMAT = list(SUPPORTED.keys())
SUPPORTED_EXT = None
for itm in SUPPORTED.values():
    SUPPORTED_EXT.append(itm)

IMAGES_DIR=os.path.join(Path.cwd(), 'images', 'raw')
OUT_RES=os.path.join(Path.cwd(), 'images', 'training-sets', 'Resolution')
OUT_ASPR=os.path.join(Path.cwd(), 'images', 'training-sets', 'Aspect-Ratio')
OUT_UNM=os.path.join(Path.cwd(), 'images', 'training-sets', 'Unmatched')

class PreProcess():
    """Convert image files, prepare directories, and build vectors"""
    def __init__(self):
        self.raw = Path(IMAGES_DIR)

    def format(raw:str|Path=IMAGES_DIR, format:str="PNG"):
        """Convert all files to a target format and rename"""
        for item in tqdm(Path(raw).iterdir(), desc="Converting Images", unit="image"):
            # Ensure its a file
            if not item.is_file():
                continue
            
            # Ensure its a valid image file otherwise remove from set
            if item.suffix.lower() not in SUPPORTED_EXT:
                continue
        
            # Open, Convert, Save, Delete
            try:
                count = 0
                with Image.open(item) as image:
                    image = image.convert("RGB")
                    count += 1
                    output = Path(f"{raw}/{count}")
                    image.save(output, format=format)
                    if Path(output).is_file():
                        item.unlink()
            except Exception as e:
                print(f"Error: {e}")

    def img(filename:str|Path) -> np.ndarray:
        """Process an image into a normalized, flattened RGB vector."""
        path = Path(filename)
        with Image.open(path) as image:
            image = image.convert("RGB")
            image = image.resize()
            pixels = np.array(image, dtype=np.float32)
        vector = pixels.flatten() / 255.0
        return vector

    def res(res:tuple[int,int]=(832,1248)):
        """Copy all images of the same Resolution to training-sets"""
        for item in tqdm(Path(IMAGES_DIR).iterdir(), desc="Moving Images", unit="image"):
            try:
                with Image.open(item) as image:
                    name = item.name
                    resolution = image.size
                    if resolution != res:
                        continue
                    output = os.path.join(OUT_RES, name)
                    image.save(Path(output), "PNG")
            except Exception as e:
                print(f"Error: {e}")

    def aspr(aspr:tuple[int,int]=(2,3)):
        """Copy all images of the same Aspect-Ratio to training-sets"""
        for item in tqdm(Path(IMAGES_DIR).iterdir(), desc="Copying Images", unit="image"):
            if not Path(item).is_file():
                continue
            try:
                with Image.open(item) as image:
                    name = item.name
                    resolution = image.size
                    divisor = gcd(resolution[0], resolution[1])
                    ratio = resolution[0] // divisor, resolution[1] // divisor
                    if ratio != aspr:
                        continue
                    output = os.path.join(Path.cwd(), "images", "training-sets", "Aspect-Ratio", name)
                    image.save(Path(output), "PNG")
            except Exception as e:
                print(f"Error processing {item.name}: {e}")
                continue

class NeuralNetwork():
    """Store parameters and perform calculations"""
    def __init__(self, model:str="xai"):
        name = os.path.join(Path.cwd(), "models", model, ".npz")
        with np.load(Path(name)) as data:
            self.W = data["weights"]
            self.B = data["biases"]
            self.name = model
            self.path = Path(name)
    
    def new(input_count: int, neuron_count: int, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Create and save a model with configurable inputs and neurons."""
        if input_count <= 0 or neuron_count <= 0:
            raise ValueError("Input and neuron counts must be positive.")
        newW = np.random.randn(neuron_count, input_count) * 0.01
        newB = np.zeros(neuron_count)
        path = os.path.join(Path.cwd(), "models", name, ".npz")
        np.savez(Path(path), weights=newW, biases=newB)
        print(f"Model parameters saved to {path}")
        print("Weight matrix shape:", newW.shape)
        print("Bias vector shape:", newB.shape)
        return newW, newB

    def sigmoid(z: np.ndarray) -> np.ndarray:
        """Map raw neuron outputs to values between zero and one."""
        z = np.clip(z, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-z))

    def fp(self, inputs:np.ndarray) -> np.ndarray:
        """Calculate and return a standard forward processing operation"""
        z = self.W @ inputs + self.B
        predict = self.sigmoid(z)
        return predict

class Trainer():
    """Prepare and execute training sessions"""
    def __init__(self, model:str="xai"):
        name = os.path.join(Path.cwd(), "models", model, ".npz")
        with np.load(Path(name)) as data:
            self.W = data["weights"]
            self.B = data["biases"]
            self.name = model

    def train_xai(self, yes_dir:str|Path, no_dir:str|Path, epoc:int, rate:float, size:tuple[int,int]) -> tuple[np.ndarray,np.ndarray]:
        """Train a single-output binary classifier with gradient descent."""
        path = Path(self.name)
        yes_dir = Path(yes_dir)
        no_dir = Path(no_dir)
        if epoc <= 0:
            raise ValueError("epocs must be positive.")
        if rate <= 0:
            raise ValueError("Learning rate must be positive.")
        if not yes_dir.is_dir() or not no_dir.is_dir():
            raise NotADirectoryError(
                "Both yes_dir and no_dir must be existing directories."
            )
        if W.ndim != 2 or W.shape[0] != 1 or B.shape != (1,):
            raise ValueError(
                "This trainer requires one output neuron and one bias."
            )
        expected_inputs = size[0] * size[1] * 3
        if W.shape[1] != expected_inputs:
            raise ValueError(
        f"Model expects {W.shape[1]} inputs, but proc_img "
                f"produces {expected_inputs}. Recreate the model "
                "or use matching image dimensions."
            )
        samples = []
        for directory, target in (
            (yes_dir, 1.0),
            (no_dir, 0.0),
        ):
            for image_path in directory.iterdir():
                if (
                    image_path.is_file()
                    and image_path.suffix.lower() in SUPPORTED_EXT
                ):
                    samples.append((image_path, target))
        if not samples:
            raise ValueError("No supported images found in the training directories.")
    
        for epoc in tqdm(range(epoc), desc="Training epocs"):
            np.random.shuffle(samples)
            total_loss = 0.0
            progress = tqdm(
                samples,
                desc=f"epoc {epoc + 1}/{epoc}",
                leave=False,
            )
            for i, (image_path, target) in enumerate(progress):
                inputs = PreProcess.proc_img(image_path)
    
                # Forward pass.
                prediction = NeuralNetwork.fp(inputs)
    
                # Binary cross-entropy loss.
                p = float(np.clip(prediction[0], 1e-7, 1.0 - 1e-7))
    
                loss = -(
                    target * np.log(p)
                    + (1.0 - target) * np.log(1.0 - p)
                )
                total_loss += loss
    
                # Backpropagation for sigmoid + binary cross-entropy.
                dz = prediction - target
                dW = dz.reshape(-1, 1) @ inputs.reshape(1, -1)
                dB = dz
    
                # Gradient descent.
                W -= rate * dW
                B -= rate * dB
    
                progress.set_postfix(
                    loss=f"{total_loss / (i + 1):.4f}"
                )
            tqdm.write(
                f"epoc {epoc + 1}/{epoc} — "
                f"average loss: {total_loss / len(samples):.4f}"
            )
            
        # Use the same parameter keys as create_model() and load_model().
        np.savez(path, weights=W, biases=B)
        print(f"Trained model saved to {path}")
        return W, B