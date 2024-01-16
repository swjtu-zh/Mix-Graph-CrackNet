
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import struct as st
import numpy as np
import cv2
from glob import glob
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall, Precision
from model import Mix_Graph_CrackNet
from metrics import loss, dice_coef, iou

""" Global parameters """
H = 256
W = 512


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def shuffling(x, y):
    x, y = shuffle(x, y, random_state=42)
    return x, y


def load_data(path):
    x = sorted(glob(os.path.join(path, "image", "*raw"))) #Modify according to the situation of your own dataset.
    y = sorted(glob(os.path.join(path, "mask", "*bmp"))) #Modify according to the situation of your own dataset.
    return x, y

def load_3ddata(path, width, height):
    filename = path.decode()
    try:
        with open(filename, 'rb') as fp:
            EleByte = fp.read(4 * width * height)  # 4-byter per element
            Fmt = '@' + str(width * height) + 'f'
            Val = st.unpack_from(Fmt, EleByte)
            return np.expand_dims(np.array(Val, dtype=np.float32).reshape((height, width)), axis=-1)
    except FileNotFoundError:
        print('Not Found '+filename)
    except OSError:
        print('Failed to read '+filename)
    except:
        print('unknown error for '+filename)

def read_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = x/255.0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=-1)
    return x

def read_mask(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=-1)
    return x

def tf_parse(x, y):
    def _parse(x, y):
        x = load_3ddata(x, W, H)
        #x = read_image(x)  If the dataset consists of 2D images, use this function instead of load_3ddata().
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([H, W, 1])
    y.set_shape([H, W, 1])
    return x, y


def tf_dataset(X, Y, batch=2):
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(10)
    return dataset

if __name__ == "__main__":

    """ GPU Checking """
    cuda_gpu_available = tf.test.is_gpu_available(cuda_only=True)
    print("GPU Available: " + f"{cuda_gpu_available}")

    """ Seeding """
    # np.random.seed(42)
    # tf.random.set_seed(42)

    """ Directory for storing files """
    create_dir("files")

    """ Hyperparameters """
    batch_size = 4
    lr = 1e-4
    num_epochs = 240
    model_path = os.path.join("files", "model_best.ckpt")
    csv_path = os.path.join("files", "data.csv")

    """ Dataset """
    dataset_path = "new_data"
    train_path = os.path.join(dataset_path, "train")
    valid_path = os.path.join(dataset_path, "validation")

    train_x, train_y = load_data(train_path)
    train_x, train_y = shuffling(train_x, train_y)
    valid_x, valid_y = load_data(valid_path)

    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Valid: {len(valid_x)} - {len(valid_y)}")

    train_dataset = tf_dataset(train_x, train_y, batch=batch_size)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=batch_size)

    """ Model """
    model = Mix_Graph_CrackNet(if_use_learnable_mechanism=True, Minchnum=32, if_use_single_gbsc=True)
    model.build((batch_size, H, W, 1))
    model.compile(loss=loss, optimizer=Adam(lr), metrics=[dice_coef, iou, Recall(), Precision()])

    # if os.path.exists(model_curpath):
    #     print('-------------load the model-----------------')
    #     model.load_weights(model_curpath)

    callbacks = [
        ModelCheckpoint(model_path, verbose=1, save_best_only=True, save_weights_only=True, mode='max', monitor='val_iou'),
        CSVLogger(csv_path),
        TensorBoard(),
        # EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=False),
    ]

    model.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=valid_dataset,
        shuffle=True,
        callbacks=callbacks
    )
