
# 导入必要的库
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 加载数据集
train_ds = keras.preprocessing.image_dataset_from_directory(
    "path/to/dataset",
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=(32, 32),
    batch_size=32,
)
val_ds = keras.preprocessing.image_dataset_from_directory(
    "path/to/dataset",
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=(32, 32),
    batch_size=32,
)

# 编译CNN模型
model = keras.Sequential(
    [
        layers.experimental.preprocessing.Rescaling(1./255),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(2),
        layers.Conv2D(128, 3, activation="relu"),
        layers.MaxPooling2D(2),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(10),
    ]
)

# 模型训练&评估
model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
model.fit(train_ds, epochs=10, validation_data=val_ds)
model.evaluate(val_ds)
