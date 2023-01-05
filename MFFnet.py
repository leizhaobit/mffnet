# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 20:03:01 2020

@author: ZL
"""
"""
双通道输入
"""

import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import backend as K
from tensorflow.keras.applications import vgg16
from tensorflow.keras import Model
from gabor_tf2 import Gabor_Conv, GaborNN
from attention import channel_attention_se, cbam_block

os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'

'''
def ResBlock(x, K):
    shortcut = x
    conv1 = TimeDistributed(Conv2D(K, (3, 3), padding='same'))(x)

    bn2 = TimeDistributed(BatchNormalization())(conv1)
    act2 = TimeDistributed(Activation('relu'))(bn2)
    conv2 = TimeDistributed(Conv2D(K, (3, 3), padding='same'))(act2)

    x = add([conv2, shortcut])
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Activation('relu'))(x)

    return x
'''
def Unet(inp1, acti):
    concat_axis = 3
    # input
    #inputs = Input(shape=(in_h,in_w,in_c))
    conv1 = Conv2D(64, (3, 3), activation=acti, padding='same')(inp1)
    conv1 = Conv2D(64, (3, 3), activation=acti, padding='same')(conv1)
    pooling1 = MaxPooling2D((2, 2), strides=(2, 2))(conv1)

    # Block 2
    conv2 = Conv2D(128, (3, 3), activation=acti, padding='same')(pooling1)
    conv2 = Conv2D(128, (3, 3), activation=acti, padding='same')(conv2)
    pooling2 = MaxPooling2D((2, 2), strides=(2, 2))(conv2)

    # Block 3
    conv3 = Conv2D(256, (3, 3), activation=acti, padding='same')(pooling2)
    conv3 = Conv2D(256, (3, 3), activation=acti, padding='same')(conv3)
    conv3 = Conv2D(256, (3, 3), activation=acti, padding='same')(conv3)
    pooling3 = MaxPooling2D((2, 2), strides=(2, 2))(conv3)

    # Block 4
    conv4 = Conv2D(512, (3, 3), activation=acti, padding='same')(pooling3)
    conv4 = Conv2D(512, (3, 3), activation=acti, padding='same')(conv4)
    conv4 = Conv2D(512, (3, 3), activation=acti, padding='same')(conv4)
    pooling4 = MaxPooling2D((2, 2), strides=(2, 2))(conv4)

    # Block 5
    conv5 = Conv2D(512, (3, 3), activation=acti, padding='same')(pooling4)
    conv5 = Conv2D(512, (3, 3), activation=acti, padding='same')(conv5)
    conv5 = Conv2D(512, (3, 3), activation=acti, padding='same')(conv5)
    pooling5 = MaxPooling2D((2, 2), strides=(2, 2))(conv5)

    # model = tf.keras.Model(inputs=inp1, outputs=[pooling2, pooling5])
    return pooling2, pooling5


class AdaptiveAvgPool2D(tf.keras.Model):
    def __init__(self, output_size):
        super(AdaptiveAvgPool2D, self).__init__()
        self.output_size = np.array(output_size)
        # def build(self, input_shape):
        #     super(AdaptiveAvgPool2D, self).build(input_shape)

    def get_config(self):
        return {"kernel_size": self.output_size}

    def call(self, inputs, training=None, mask=None):
        input_size = [inputs.shape[1], inputs.shape[2]]
        stride = np.floor((input_size / self.output_size))
        kernel_size = inputs.shape[1:3] - (self.output_size - 1) * stride
        kernel_size = tuple(kernel_size)
        out = AveragePooling2D(pool_size=kernel_size, strides=stride, padding='valid')(inputs)
        # out = tf.nn.avg_pool2d(inputs, ksize=kernel_size, strides=stride, padding='VALID')
        return out

'''
def Timedis(inputs):
    x =TimeDistributed(Conv2D(32, 3, padding="same", activation="relu"))(inputs)
    x = ResBlock(x, 32)
    x = TimeDistributed(Conv2D(32, (3, 3), strides=(2, 2), activation='relu', padding='same'))(x)
    x = TimeDistributed(Conv2D(64, 3, padding="same", activation="relu"))(x)
    x = ResBlock(x, 64)
    x = TimeDistributed(Conv2D(64, (3, 3), strides=(2, 2), activation='relu', padding='same'))(x)
    x = TimeDistributed(Conv2D(128, 3, padding="same", activation="relu"))(x)
    x = ResBlock(x, 128)
    x = TimeDistributed(Conv2D(128, (3, 3), strides=(2, 2), activation='relu', padding='same'))(x)
    x = TimeDistributed(Conv2D(256, 3, padding="same", activation="relu"))(x)
    x = ResBlock(x, 256)
    x = TimeDistributed(Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='same'))(x)
    v0 = x[:,0,:,:,:]
    v1 = x[:,1,:,:,:]
    v2 = x[:,2,:,:,:]
    v3 = x[:,3,:,:,:]

    # x = concatenate([v0, v1, v2, v3], axis=-1)
    # print(x.shape)
    x = tf.keras.layers.add([v0, v1, v2, v3])
    x = AveragePooling2D()(x)
    # print('pooling3:', x.shape)
    # fla = TimeDistributed(Flatten())(x)
    # lstm = LSTM(512, dropout=0.5)(fla)
    return v0, v1, v2, v3, x
'''
'''
x = np.random.randint(0, 255, [16, 4, 128, 128, 1])
x = x.astype('float32')
net = Timedis(x)
exit()
'''


def spatial_attention(input_feature):
    kernel_size = 7

    if K.image_data_format() == "channels_first":
        channel = input_feature.shape[1]
        cbam_feature = Permute((2, 3, 1))(input_feature)
    else:
        channel = input_feature.shape[-1]
        cbam_feature = input_feature

    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    assert avg_pool.shape[-1] == 1
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    assert max_pool.shape[-1] == 1
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    assert concat.shape[-1] == 2
    cbam_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          activation='sigmoid',
                          strides=1,
                          padding='same',
                          kernel_initializer='he_normal',
                          use_bias=False,
                          name='cbam')(concat)

    assert cbam_feature.shape[-1] == 1

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return cbam_feature
'''
x = np.random.randint(0, 255, [16, 4, 4, 1024])
x = x.astype('float32')
net = Attention(in_channels=1024, out_channels=8)
out = net(x)
print(out.shape)
exit()'''


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


def Texture_enhance(inputs, gabor_feature, att, num_features=256, feature_size=28):
    pool3 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same',
                                   activation='swish')
    pool4 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), padding='same',
                                   activation='swish')
    pool5 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(2, 2), padding='same',
                                   activation='swish')
    c3 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                activation='swish')
    c4 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                activation='swish')
    c5 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                activation='swish')

    x = inputs
    x1 = Conv2D(num_features, (3, 3), activation='swish', padding='same', dilation_rate=1)(x)
    x2 = Conv2D(num_features, (3, 3), activation='swish', padding='same', dilation_rate=2)(x)
    x3 = Conv2D(num_features, (3, 3), activation='swish', padding='same', dilation_rate=3)(x)
    x1_ = AdaptiveAvgPool2D(output_size=feature_size)(x1)
    x1_ = tf.image.resize(x1_, (inputs.shape[1], inputs.shape[2]))
    x1 = x1 - x1_

    x2_ = AdaptiveAvgPool2D(output_size=feature_size)(x2)
    x2_ = tf.image.resize(x2_, (inputs.shape[1], inputs.shape[2]))
    x2 = x2 - x2_

    x3_ = AdaptiveAvgPool2D(output_size=feature_size)(x3)
    x3_ = tf.image.resize(x3_, (inputs.shape[1], inputs.shape[2]))
    x3 = x3 - x3_

    x = concatenate([x1, x2, x3], axis=-1)
    x = c3(x)
    x = ResBlock(x, 128, att)
    x = pool3(x)
    x = c4(x)
    x = ResBlock(x, 256, att)
    x = pool4(x)
    x = c5(x)
    x = ResBlock(x, 512, att)
    x = pool5(x)

    x = concatenate([x, gabor_feature], axis=-1)

    # x = add([x, self.gabor_feature])
    return x
'''
x = np.random.randint(0,255,[16,56,56,128])
x = x / 255
Texture_enhance(x,gabor_feature=0, att='swish', num_features=128, feature_size=28)
exit()
'''

x_in = Input(shape=(224,224,3))
base = tf.keras.applications.Xception(include_top=False, input_shape=(224, 224, 3))
feature_model = Model(base.input, base.get_layer(name='block2_pool').output)
#high_pass = high_pass_net(in_channels=1, out_channels=16, kernel_size=(8, 8), shape=(224, 224), att=2)
xx = preprocess_input(x_in)
x = base(xx)
feature = feature_model(xx)
# feature = base.get_layer(name='block2_pool').output
# feature, x = Unet(x_in, 'swish')
#x0 = spatial_attention(x)
#x0 = multiply([x0, x])
#x = Add()([x0, x])

g1 = Gabor_Conv(in_channels=1, out_channels=16, kernel_size=(8, 8))

gray = tf.image.rgb_to_grayscale(x_in)

x1 = GaborNN(inputs=gray, in_channels=1, g0=g1, out_channels=16, kernel_size=(8, 8), att=2)
x1 = Texture_enhance(inputs=feature, att='swish', num_features=128, feature_size=28, gabor_feature=x1)

x = concatenate([x, x1], axis=-1)
x = Flatten()(x)
# print(x.shape)
x = Dense(1024)(x)
x = Dropout(0.5)(x)
x = Dense(1, activation='sigmoid')(x)


model = Model(x_in, x)
model.summary()

tf.config.experimental_run_functions_eagerly(True)



model_name = 'MFFnet-xcep-c40-NE'

target_size = (224, 224)
train_dir = '../datasets/NE/train/'
validation_dir = '../datasets/NE/validation/'
train_gen = ImageDataGenerator()
validation_gen = ImageDataGenerator()
train_generator = train_gen.flow_from_directory(train_dir, target_size=target_size, batch_size=16, class_mode='binary', shuffle=True)
validation_generator = validation_gen.flow_from_directory(validation_dir, target_size=target_size, class_mode='binary', shuffle=True)


os.makedirs('./MFFnet/',exist_ok=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='auto', factor=0.2)
checkpointer = ModelCheckpoint(filepath='./MFFnet/' + model_name + '_best_weights.h5', verbose=1, monitor='val_loss', mode='auto', save_best_only=True) #save at each epoch if the validation decreased
early_stop = EarlyStopping(monitor='val_loss', patience=20)

model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC()])
model.fit(x=train_generator, batch_size=16, epochs=100, validation_data=validation_generator, verbose=1, callbacks=[reduce_lr, checkpointer, early_stop])

# plot_model(model, to_file=model_name + '.jpg', show_layer_names=True, show_shapes=True)
'''
class MFF(tf.keras.Model):
    def __init__(self):
        super(MFF, self).__init__()
        # self.vgg16 = vgg()
        # self.vgg16 = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=(128, 128, 3))
        inp1 = Input(shape=(128, 128, 3))
        self.vgg16 = Unet(inp1)
        self.vgg16.load_weights(r'./vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')

    def cos_loss(self, v0, v1, v2, v3):
        vm = list([v0, v1, v2, v3])
        smax = 0
        alpha = 1
        loss_list = list()
        for i in range(4):
            for j in range(i + 1, 4):
                out = tf.maximum(0, tf.keras.losses.cosine_similarity(vm[i], vm[j]) - smax)
                loss_list.append(out)
        loss = alpha * tf.reduce_mean(loss_list)
        return loss

    def call(self, inputs, training=None, mask=None):
        inp1 = inputs[0]
        inp2 = inputs[1]
        feature, x0 = self.vgg16(inp1)
        v0, v1, v2, v3, x1 = Timedis(inp2)
        loss = self.cos_loss(v0,v1,v2,v3)
        self.add_loss(loss)
        # feature = self.vgg16.get_layer(name='block2_pool')
        
        x2 = Texture_enhance(num_features=128, feature_size=4, gabor_feature=x1)(feature)
        f1 = concatenate([x0, x2], axis=-1)

        att = Attention(in_channels=1024, out_channels=8)(x0)
        matrix = tf.einsum('ijkm,ijkn->imn', x0, att)
        matrix = tf.nn.l2_normalize(matrix, axis=-1)
        matrix = Flatten()(matrix)
        x = Flatten()(f1)
        x = concatenate([x, matrix], axis=-1)

        x = Dense(1024, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(256)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(1, activation='sigmoid')(x)
        return predictions
'''
'''net = MFF()
x = np.random.randint(0, 255, [16, 128, 128, 3])
out = net(x)
print(out.shape)
exit()'''
