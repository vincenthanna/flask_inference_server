import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from numpy import asarray


rgb_weights = [0.2989, 0.5870, 0.1140]





img = Image.open('cat.jpg').convert('L')

img = img.resize((28, 28))


b = asarray(img)
print(b.shape)
