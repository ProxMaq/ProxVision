import tensorflow as tf
from train import args


train_ds=tf.keras.preprocessing.image_dataset_from_directory(
    args.train_data,
    image_size=(96,96),
    batch_size=3

)

val_ds=tf.keras.preprocessing.image_dataset_from_directory(
    args.val_data,
    validation_split=0.067,
    subset="validation",
    seed=123,
    image_size=(96,96),
    batch_size=3
)
