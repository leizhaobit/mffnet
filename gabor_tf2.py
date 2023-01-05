import os
import numpy as np
from tensorflow.python.keras.engine.base_layer import Layer
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.keras.utils import conv_utils
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization, Dropout, Dense, Flatten, Activation, Input, concatenate, add
from attention import channel_attention_se, cbam_block
os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'


class Gabor_Conv(tf.keras.layers.Layer):

    def __init__(self,in_channels, out_channels,
                 kernel_size,
                 strides=1,
                 padding='same',
                 dilation_rate=1,
                 **kwargs):
        super(Gabor_Conv, self).__init__(trainable=True, name='conv-gabor', **kwargs)

        self.kernel_size = conv_utils.normalize_tuple(kernel_size, 2, 'kernel_size')
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.strides = strides
        self.padding = padding
        self.dilation_rate = dilation_rate
        self.x0 = np.ceil([self.kernel_size[0] / 2])[0]
        self.y0 = np.ceil([self.kernel_size[1] / 2])[0]
        '''
        self.freq = nn.Parameter(
            (3.14 / 2) * 1.41 ** (-torch.randint(0, 5, (out_channels, in_channels))).type(torch.Tensor))
        self.theta = nn.Parameter((3.14 / 8) * torch.randint(0, 8, (out_channels, in_channels)).type(torch.Tensor))
        self.psi = nn.Parameter(3.14 * torch.rand(out_channels, in_channels))
        self.sigma = nn.Parameter(3.14 / self.freq)
        self.x0 = torch.ceil(torch.Tensor([self.kernel_size[0] / 2]))[0]
        self.y0 = torch.ceil(torch.Tensor([self.kernel_size[1] / 2]))[0]
        '''
    def get_config(self):
        return {"kernel_size": self.kernel_size}

    def kernel_init(self):
        b = (np.pi / 2) * 1.41 ** (-np.random.randint(0, 5, (self.out_channels, self.in_channels)))
        return b

    def build(self, input_shape):
        self.freq = self.add_weight(name="freq",
                                    shape=[self.out_channels, self.in_channels],
                                    initializer=tf.keras.initializers.constant((np.pi / 2) * 1.41 ** (-np.random.randint(0, 5, (self.out_channels, self.in_channels)))),
                                    trainable=True,
                                    dtype='float32')

        self.theta = self.add_weight(name="theta",
                                     shape=[self.out_channels, self.in_channels],
                                     initializer=tf.keras.initializers.constant((np.pi / 8) * (
                                         -np.random.randint(0, 8, (self.out_channels, self.in_channels)))),
                                     trainable=True,
                                     dtype='float32')

        self.psi = self.add_weight(name="psi",
                                   shape=[self.out_channels, self.in_channels],
                                   initializer=tf.keras.initializers.constant(np.pi * (
                                       -np.random.randint(0, 8, (self.out_channels, self.in_channels)))),
                                   trainable=True,
                                   dtype='float32')

        self.sigma = self.add_weight(name="sigma",
                                     shape=[self.out_channels, self.in_channels],
                                     initializer=tf.keras.initializers.constant(np.pi / self.freq),
                                     trainable=True,
                                     dtype='float32')

        self.built = True

    def call(self, inputs, **kwargs):
        # inputs = self.gray(inputs)
        '''
        bs = inputs.shape[0]
        if bs == None:
            bs = 16
        ones = tf.ones(shape=[bs, inputs.shape[1], inputs.shape[2], inputs.shape[3]])
        inputs = inputs * ones
        '''
        y, x = tf.meshgrid(tf.linspace(-self.x0 + 1, self.x0, self.kernel_size[0]),
                               tf.linspace(-self.y0 + 1, self.y0, self.kernel_size[1]))
        # print('y.shape:', y.shape)
        '''weight = tf.Variable(initial_value= tf.zeros((self.out_channels,
                                      self.in_channels, self.kernel_size[0], self.kernel_size[1])), shape= (self.out_channels,
                                      self.in_channels, self.kernel_size[0], self.kernel_size[1]), trainable=True)'''
        temp_weight = list()
        for i in range(self.out_channels):
            for j in range(self.in_channels):
                sigma = tf.broadcast_to(self.sigma[i, j], y.shape)
                freq = tf.broadcast_to(self.freq[i, j], y.shape)
                theta = tf.broadcast_to(self.theta[i, j], y.shape)
                psi = tf.broadcast_to(self.psi[i, j], y.shape)

                #print(x.dtype)
                #print(tf.cos(theta).dtype)
                rotx = tf.cast(x, dtype='float32') * tf.cos(theta) + tf.cast(y, dtype='float32') * tf.sin(theta)
                roty = -tf.cast(x, dtype='float32') * tf.sin(theta) + tf.cast(y, dtype='float32') * tf.cos(theta)

                # g = tf.zeros(y.shape)

                g = tf.exp(-0.5 * ((rotx ** 2 + roty ** 2) / (sigma + 1e-3) ** 2))
                g = g * tf.cos(freq * rotx + psi)
                g = g / (2 * 3.14 * sigma ** 2)
                temp_weight.append(g)
                # weight[i, j].assign(g)
                # weight[i, j] = g
                # self.weight.data[i, j] = g
        # weight.shape=(W, H, C, filters.number)
        # print(weight.shape)
        # print(self.freq[0, 0])
        weight0 = tf.stack(temp_weight)
        weight = tf.expand_dims(weight0, 1, name='expand')
        # print(weight.shape)
        weight = tf.transpose(weight, perm=[2, 3, 1, 0])
        # print(weight.shape)
        '''
        outputs = tf.nn.conv2d(
            inputs,
            weight,
            strides=self.strides,
            padding='SAME',
            data_format=None
        )
        '''
        outputs = K.conv2d(
            inputs,
            weight,
            strides=self.strides,
            padding=self.padding,
            data_format=None,
            dilation_rate=self.dilation_rate
        )

        return outputs


def ResBlock(x, K, att):
    shortcut = x
    conv1 = Conv2D(K, (3, 3), padding='same')(x)

    bn2 = BatchNormalization()(conv1)
    act2 = Activation('swish')(bn2)
    conv2 = Conv2D(K, (3, 3), padding='same')(act2)
    if att == 1:
        conv2 = channel_attention_se(conv2, K, 4)
    elif att == 2:
        conv2 = cbam_block(conv2, 4)
    x = add([conv2, shortcut])
    x = BatchNormalization()(x)
    x = Activation('swish')(x)

    return x


def GaborNN(inputs, g0, in_channels=1, out_channels=16, kernel_size=(8, 8), att=0):
    c1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                activation='swish')
    c2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                activation='swish')
    c3 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                activation='swish')
    c4 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                activation='swish')
    c5 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                activation='swish')
    pool1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same',
                                   activation='swish')
    pool2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same',
                                   activation='swish')
    pool3 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same',
                                   activation='swish')
    pool4 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), padding='same',
                                   activation='swish')
    pool5 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(2, 2), padding='same',
                                   activation='swish', name='texture_final')

    x = g0(inputs)
    x = tf.transpose(x, perm=[0,1,2,3], name='gabor')
    #print(x.shape)
    x = x / 255
    x = BatchNormalization()(x)
    x = Activation('swish')(x)
    x = c1(x)
    #print(x.shape)
    x = ResBlock(x, 32, att)
    x = pool1(x)
    x = c2(x)
    x = ResBlock(x, 64, att)
    x = pool2(x)
    x = c3(x)
    x = ResBlock(x, 128, att)
    x = pool3(x)
    x = c4(x)
    x = ResBlock(x, 256, att)
    x = pool4(x)
    x = c5(x)
    x = ResBlock(x, 512, att)
    x = pool5(x)
    # print(x.shape)
    return x


'''
class GaborNN(tf.keras.Model):
    def __init__(self, in_channels=1, out_channels=16, kernel_size=(8, 8), att=0):
        super(GaborNN, self).__init__()
        self.g0 = Gabor_Conv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.c1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='same', activation='swish')
        self.c2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', activation='swish')
        self.c3 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same', activation='swish')
        self.c4 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same', activation='swish')
        self.c5 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same', activation='swish')
        self.pool1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(2,2), padding='same', activation='swish')
        self.pool2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), padding='same', activation='swish')
        self.pool3 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(2,2), padding='same', activation='swish')
        self.pool4 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(2,2), padding='same', activation='swish')
        self.pool5 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=(2,2), padding='same', activation='swish')
        self.att = att


    def get_config(self):
        return {"kernel_size": 3}

    def ResBlock(self, x, K):
        shortcut = x
        conv1 = Conv2D(K, (3, 3), padding='same')(x)

        bn2 = BatchNormalization()(conv1)
        act2 = Activation('swish')(bn2)
        conv2 = Conv2D(K, (3, 3), padding='same')(act2)
        if self.att == 1:
            conv2 = channel_attention_se(conv2, K, 4)
        elif self.att == 2:
            conv2 = cbam_block(conv2, 4)

        x = add([conv2, shortcut])
        x = BatchNormalization()(x)
        x = Activation('swish')(x)

        return x

    def call(self, inputs, **kwargs):
        # inputs = tf.image.rgb_to_grayscale(inputs)
        x = self.g0(inputs)
        x = x / 255
        x = BatchNormalization()(x)
        x = Activation('swish')(x)
        x = self.c1(x)
        x = self.ResBlock(x, 32)
        x = self.pool1(x)
        x = self.c2(x)
        x = self.ResBlock(x, 64)
        x = self.pool2(x)
        x = self.c3(x)
        x = self.ResBlock(x, 128)
        x = self.pool3(x)
        x = self.c4(x)
        x = self.ResBlock(x, 256)
        x = self.pool4(x)
        x = self.c5(x)
        x = self.ResBlock(x, 512)
        x = self.pool5(x)
        # print(x.shape)
        return x
'''

'''
x = np.random.randn(18, 128, 128, 1)
x2 = np.random.randint(0, 255, [18, 128, 128, 1])
x2 = x2.astype('float32')
net = GaborNN()
out = net(x2)
print(out.shape)
'''
# print(out)
'''
tf.config.experimental_run_functions_eagerly(True)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
test_dir = 'D:\GAN2\data\\test_data'
test_gen = ImageDataGenerator()
test_generator = test_gen.flow_from_directory(test_dir, target_size=(128, 128), batch_size=16, color_mode='grayscale')
model = GaborNN()
model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4), loss=tf.keras.losses.binary_crossentropy, metrics=['accuracy'])
model.fit_generator(test_generator, epochs=3)
'''
# model.fit(test_generator, epochs=3)

# net = _Conv(in_channels=1, out_channels=96, kernel_size=3)

