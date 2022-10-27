from source.GAN import GAN
from source.data_loader import get_data
from source.utils import load_config

config = load_config()

if __name__ == '__main__':
    train_ds, val_ds = get_data()
    gan = GAN()
    gan.fit(train_ds, val_ds, 40000)
