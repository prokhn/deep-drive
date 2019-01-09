import json
from keras import models
from keras.layers import Lambda
from keras.layers.core import Activation, Reshape, Permute
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint

# BatchNormalization = lambda: Lambda(lambda x: x)

class SegnetBuilder:
    @staticmethod
    def build(model_name, img_h, img_w, img_layers, n_labels, kernel=3,
              save_path='models/{}.json') -> models.Sequential:
        encoding_layers = [
            Conv2D(64, kernel, padding='same', input_shape=(None, None, img_layers)),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(64, kernel, padding='same'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(),

            Conv2D(128, (kernel, kernel), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(128, (kernel, kernel), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(),

            Conv2D(256, (kernel, kernel), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(256, (kernel, kernel), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(256, (kernel, kernel), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(),

            Conv2D(512, (kernel, kernel), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(512, (kernel, kernel), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(512, (kernel, kernel), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(),

            Conv2D(512, (kernel, kernel), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(512, (kernel, kernel), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(512, (kernel, kernel), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(),
            ]

        decoding_layers = [
            UpSampling2D(),
            Conv2D(512, (kernel, kernel), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(512, (kernel, kernel), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(512, (kernel, kernel), padding='same'),
            BatchNormalization(),
            Activation('relu'),

            UpSampling2D(),
            Conv2D(512, (kernel, kernel), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(512, (kernel, kernel), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(256, (kernel, kernel), padding='same'),
            BatchNormalization(),
            Activation('relu'),

            UpSampling2D(),
            Conv2D(256, (kernel, kernel), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(256, (kernel, kernel), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(128, (kernel, kernel), padding='same'),
            BatchNormalization(),
            Activation('relu'),

            UpSampling2D(),
            Conv2D(128, (kernel, kernel), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(64, (kernel, kernel), padding='same'),
            BatchNormalization(),
            Activation('relu'),

            UpSampling2D(),
            Conv2D(64, (kernel, kernel), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(n_labels, (1, 1), padding='valid'),
            BatchNormalization(),
            ]

        autoencoder = models.Sequential()
        autoencoder.encoding_layers = encoding_layers

        for l in autoencoder.encoding_layers:
            autoencoder.add(l)
            # print(l.input_shape, l.output_shape, l)

        autoencoder.decoding_layers = decoding_layers
        for l in autoencoder.decoding_layers:
            autoencoder.add(l)

        # autoencoder.add(Reshape((n_labels, img_h * img_w)))
        # autoencoder.add(Permute((2, 1)))
        autoencoder.add(Activation('softmax'))

        # with open(save_path.format(model_name), 'w') as outfile:
            # outfile.write(json.dumps(json.loads(autoencoder.to_json()), indent=2))

        return autoencoder
