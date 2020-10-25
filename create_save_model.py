import Mask.model as modellib
import os
import tensorflow as tf
import keras as K

from Mask.meta.config.moles_config import MolesConfig

dir_path = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = dir_path + "/models/"
MODEL_PATH = MODEL_DIR + "/mask_rcnn_moles_0074.h5"

# Create the MaskRCNN model
config = MolesConfig()
print('creating model...')
model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)

# load weights
print('loading weights...')
model.load_weights(MODEL_PATH, by_name=True,)

inputs = model.keras_model.inputs
inputs = { tensor.op.name: tensor for tensor in inputs }
outputs = model.keras_model.outputs
outputs = { tensor.op.name: tensor for tensor in outputs }

model.keras_model.save("./models/save_model")

#K.models.save_model(model.keras_model, , save_format="tf")

