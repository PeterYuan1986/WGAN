import tensorflow as tf


class Discriminator(tf.keras.Model):
    def __init__(self, inputshpe=(64, 64, 64, 1), name='Discriminator'):  # img_width, img_height, img_depth
        super(Discriminator, self).__init__(name=name)
        self.inputshpe = inputshpe
        ch_in = 64
        self.leakyrelu1 = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.leakyrelu2 = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.leakyrelu3 = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.leakyrelu4 = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.conv1 = tf.keras.layers.Conv3D(ch_in, 4, strides=2, padding='same', name='conv1')
        self.conv2 = tf.keras.layers.Conv3D(ch_in * 2, 4, strides=2, padding='same', name='conv2')
        self.bn2 = tf.keras.layers.BatchNormalization(axis=-1)
        self.conv3 = tf.keras.layers.Conv3D(ch_in * 4, 4, strides=2, padding='same', name='conv3')
        self.bn3 = tf.keras.layers.BatchNormalization(axis=-1)
        self.conv4 = tf.keras.layers.Conv3D(ch_in * 8, 4, strides=2, padding='same', name='conv4')
        self.bn4 = tf.keras.layers.BatchNormalization(axis=-1)
        self.conv5 = tf.keras.layers.Conv3D(1, 4, strides=1, padding='valid', name='conv5')
        self.do1 = tf.keras.layers.Dropout(0.3, noise_shape=None, seed=None)
        self.do2 = tf.keras.layers.Dropout(0.3, noise_shape=None, seed=None)
        self.do3 = tf.keras.layers.Dropout(0.3, noise_shape=None, seed=None)

    def call(self, input, training=True, mask=None):
        x = input
        x = self.conv1(x)
        x = self.leakyrelu1(x)
        x = self.do1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leakyrelu2(x)
        x = self.do2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.leakyrelu3(x)
        x = self.do3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.leakyrelu4(x)

        x = self.conv5(x)
        return x

    def summary(self):
        x = tf.keras.Input(shape=self.inputshpe)
        model = tf.keras.Model(inputs=[x], outputs=self.call(x), name='Discriminator')
        return model.summary()


class Encoder(tf.keras.Model):
    def __init__(self, inputshpe=(64, 64, 64, 1), noise=1000, name='Encoder'):
        super(Encoder, self).__init__(name=name)
        self.inputshpe = inputshpe
        ch_in = 64
        self.noise = noise

        self.leakyrelu1 = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.leakyrelu2 = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.leakyrelu3 = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.leakyrelu4 = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.conv1 = tf.keras.layers.Conv3D(ch_in, kernel_size=4, strides=2, padding='same', name='conv1')
        self.conv2 = tf.keras.layers.Conv3D(ch_in * 2, kernel_size=4, strides=2, padding='same', name='conv2')
        self.bn2 = tf.keras.layers.BatchNormalization(axis=-1)
        self.conv3 = tf.keras.layers.Conv3D(ch_in * 4, kernel_size=4, strides=2, padding='same', name='conv3')
        self.bn3 = tf.keras.layers.BatchNormalization(axis=-1)
        self.conv4 = tf.keras.layers.Conv3D(ch_in * 8, kernel_size=4, strides=2, padding='same', name='conv4')
        self.bn4 = tf.keras.layers.BatchNormalization(axis=-1)
        self.conv5 = tf.keras.layers.Conv3D(self.noise, kernel_size=4, strides=1, padding='valid', name='conv5')

    def call(self, input, training=True, mask=None):
        x = input

        x = self.conv1(x)
        x = self.leakyrelu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leakyrelu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.leakyrelu3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.leakyrelu4(x)

        x = self.conv5(x)
        x = tf.reshape(x, (-1, self.noise))
        return x

    def summary(self):
        x = tf.keras.Input(shape=self.inputshpe)
        model = tf.keras.Model(inputs=[x], outputs=self.call(x), name='Encoder')
        return model.summary()


class Code_Discriminator(tf.keras.Model):
    def __init__(self, inputshpe=(1000,), noise: int = 1000, name='Code_Discriminator'):
        super(Code_Discriminator, self).__init__(name=name)
        self.inputshpe = inputshpe
        self.noise = noise
        self.fc1 = tf.keras.layers.Dense(self.noise, )
        self.bn1 = tf.keras.layers.BatchNormalization(axis=-1)
        self.leakyrelu1 = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.leakyrelu2 = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.bn2 = tf.keras.layers.BatchNormalization(axis=-1)
        self.fc2 = tf.keras.layers.Dense(4096)
        self.fc3 = tf.keras.layers.Dense(1)

        self.do1 = tf.keras.layers.Dropout(0.3, noise_shape=None, seed=None)

    def call(self, input):
        x = self.fc1(input)
        x = self.bn1(x)
        x = self.leakyrelu1(x)
        x = self.do1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.leakyrelu2(x)

        x = self.fc3(x)
        return x

    def summary(self):
        x = tf.keras.Input(shape=self.inputshpe)
        model = tf.keras.Model(inputs=[x], outputs=self.call(x), name='Code_Discriminator')
        return model.summary()


class Generator(tf.keras.Model):
    def __init__(self, inputshpe=(3000,), noise=3000, channel: int = 64):
        super(Generator, self).__init__()
        self.inputshpe = inputshpe
        ch_in = channel
        self.noise = noise
        self.upsampling1 = tf.keras.layers.UpSampling3D(size=2, data_format=None)
        self.upsampling2 = tf.keras.layers.UpSampling3D(size=2, data_format=None)
        self.upsampling3 = tf.keras.layers.UpSampling3D(size=2, data_format=None)
        self.upsampling4 = tf.keras.layers.UpSampling3D(size=2, data_format=None)
        self.relu1 = tf.keras.layers.ReLU()
        self.relu2 = tf.keras.layers.ReLU()
        self.relu3 = tf.keras.layers.ReLU()
        self.relu4 = tf.keras.layers.ReLU()
        self.bn1 = tf.keras.layers.BatchNormalization(axis=-1)
        self.bn2 = tf.keras.layers.BatchNormalization(axis=-1)
        self.bn3 = tf.keras.layers.BatchNormalization(axis=-1)
        self.bn4 = tf.keras.layers.BatchNormalization(axis=-1)
        self.conv_t = tf.keras.layers.Conv3DTranspose(ch_in * 8, kernel_size=4, strides=1, padding='valid',
                                                      use_bias=False)
        self.conv2 = tf.keras.layers.Conv3D(ch_in * 4, 3, strides=1, padding='same', name='conv2',
                                            use_bias=False)
        self.conv3 = tf.keras.layers.Conv3D(ch_in * 2, 3, strides=1, padding='same', name='conv3',
                                            use_bias=False)
        self.conv4 = tf.keras.layers.Conv3D(ch_in, 3, strides=1, padding='same', name='conv4',
                                            use_bias=False)
        self.conv5 = tf.keras.layers.Conv3D(1, 3, strides=1, padding='same', name='conv5',
                                            use_bias=False)

    def call(self, noise):
        x = tf.reshape(noise, (-1, 1, 1, 1, self.noise))
        x = self.conv_t(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.upsampling1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.upsampling2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.upsampling3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.upsampling4(x)
        x = self.conv5(x)

        x = tf.keras.activations.tanh(x)

        return x

    def summary(self):
        x = tf.keras.Input(shape=self.inputshpe)
        model = tf.keras.Model(inputs=[x], outputs=self.call(x), name='Generator')
        return model.summary()
