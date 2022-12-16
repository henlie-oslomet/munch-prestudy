import yaml
from matplotlib import pyplot as plt
from yaml.loader import SafeLoader


def load_config():
    with open('../config.yml') as f:
        data = yaml.load(f, Loader=SafeLoader)

    return data


def show(image):
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()
