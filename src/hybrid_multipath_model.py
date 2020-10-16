import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Lambda, Add, LeakyReLU,  \
                                    MaxPooling2D, concatenate, UpSampling2D,\
                                    Multiply, ZeroPadding2D, Cropping2D
from tensorflow.keras import regularizers
  

def fft_layer(image):
    nchannels = 24
    for ii in range(0,nchannels,2):

        # get real and imaginary portions
        real = Lambda(lambda image: image[:, :, :, ii])(image)
        imag = Lambda(lambda image: image[:, :, :, ii+1])(image)

        image_complex = tf.complex(real, imag)  # Make complex-valued tensor
        kspace_complex = tf.signal.fft2d(image_complex)

        # expand channels to tensorflow/keras format
        real = tf.expand_dims(tf.math.real(kspace_complex), -1)
        imag = tf.expand_dims(tf.math.imag(kspace_complex), -1)
        if ii == 0: 
            kspace = tf.concat([real, imag], -1)
        else:
            kspace = tf.concat([kspace,real, imag], -1)
    return kspace


def ifft_layer(kspace_2channel):
    nchannels = 24
    for ii in range(0,nchannels,2):
        #get real and imaginary portions
        real = Lambda(lambda kspace_2channel : kspace_2channel[:,:,:,ii])(kspace_2channel)
        imag = Lambda(lambda kspace_2channel : kspace_2channel[:,:,:,ii+1])(kspace_2channel)

        kspace_complex = tf.complex(real,imag) # Make complex-valued tensor
        image_complex = tf.signal.ifft2d(kspace_complex)

        # expand channels to tensorflow/keras format
        real = tf.expand_dims(tf.math.real(image_complex),-1)
        imag = tf.expand_dims(tf.math.imag(image_complex),-1)
        if ii == 0: 
            # generate 2-channel representation of image domain
            image_complex_2channel = tf.concat([real, imag], -1)
        else:
            image_complex_2channel = tf.concat([image_complex_2channel, real, imag], -1)
    return image_complex_2channel


def cnn_block(cnn_input, depth, nf, kshape,channels,reg = 0.001):
    """
    :param cnn_input: Input layer to CNN block
    :param depth: Depth of CNN. Disregarding the final convolution block that goes back to
    2 channels
    :param nf: Number of filters of convolutional layers, except for the last
    :param kshape: Shape of the convolutional kernel
    :return: 2-channel, complex reconstruction
    """
    layers = [cnn_input]

    for ii in range(depth):
        # Add convolutional block
        layers.append(Conv2D(nf, kshape, padding='same')(layers[-1]))
        layers.append(LeakyReLU(alpha=0.1)(layers[-1]))
    final_conv = Conv2D(channels, (1, 1), activation='linear')(layers[-1])
    rec1 = Add()([final_conv,cnn_input])
    return rec1


def unet_block(unet_input, kshape=(3, 3),channels = 24):
    """
    :param unet_input: Input layer
    :param kshape: Kernel size
    :return: 2-channel, complex reconstruction
    """

    #fixing channels to default (necessary for encoding and decoding stages to have fixed length)
    if channels > 24:
        unet_input = Conv2D(24, kshape, activation='relu', padding='same')(unet_input)


    conv1 = Conv2D(48, kshape, activation='relu', padding='same')(unet_input)
    conv1 = Conv2D(48, kshape, activation='relu', padding='same')(conv1)
    conv1 = Conv2D(48, kshape, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, kshape, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, kshape, activation='relu', padding='same')(conv2)
    conv2 = Conv2D(64, kshape, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, kshape, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, kshape, activation='relu', padding='same')(conv3)
    conv3 = Conv2D(128, kshape, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, kshape, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, kshape, activation='relu', padding='same')(conv4)
    conv4 = Conv2D(256, kshape, activation='relu', padding='same')(conv4)

    up1 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv3], axis=-1)
    conv5 = Conv2D(128, kshape, activation='relu', padding='same')(up1)
    conv5 = Conv2D(128, kshape, activation='relu', padding='same')(conv5)
    conv5 = Conv2D(128, kshape, activation='relu', padding='same')(conv5)

    up2 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv2], axis=-1)
    conv6 = Conv2D(64, kshape, activation='relu', padding='same')(up2)
    conv6 = Conv2D(64, kshape, activation='relu', padding='same')(conv6)
    conv6 = Conv2D(64, kshape, activation='relu', padding='same')(conv6)

    up3 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv1], axis=-1)
    conv7 = Conv2D(48, kshape, activation='relu', padding='same')(up3)
    conv7 = Conv2D(48, kshape, activation='relu', padding='same')(conv7)
    conv7 = Conv2D(48, kshape, activation='relu', padding='same')(conv7)

    conv8 = Conv2D(channels, (1, 1), activation='linear')(conv7)
    out = Add()([conv8, unet_input])
    return out


def DC_block(rec,mask,sampled_kspace,channels,kspace = False):
    """
    :param rec: Reconstructed data, can be k-space or image domain
    :param mask: undersampling mask
    :param sampled_kspace:
    :param kspace: Boolean, if true, the input is k-space, if false it is image domain
    :return: k-space after data consistency
    """

    if kspace:
        rec_kspace = rec
    else:
        rec_kspace = Lambda(fft_layer)(rec)
    rec_kspace_dc =  Multiply()([rec_kspace,mask])
    rec_kspace_dc = Add()([rec_kspace_dc,sampled_kspace])
    return rec_kspace_dc


def deep_hybrid_multipath(H=218, W=170, Hpad = 3, Wpad = 3, kshape=(3, 3), channels = 24):

    inputs = Input(shape=(H,W,channels))
    mask = Input(shape=(H,W,channels))
    kspace_flag = True
    
    #PARTE DE CIMA

    inputs = Input(shape=(H,W,channels))
    mask = Input(shape=(H,W,channels))
    #I-1
    x = Lambda(ifft_layer)(inputs)
    kspace_flag = False
    x =  ZeroPadding2D(padding=(Hpad,Wpad))(x)
    x =  unet_block(x, kshape,channels)
    x =  Cropping2D(cropping=(Hpad,Wpad))(x)
    x =  DC_block(x,mask,inputs,channels,kspace=kspace_flag)
    x1 =  Lambda(ifft_layer)(x)              

    #K-1
    kspace_flag = True
        # Add CNN block
    x = ZeroPadding2D(padding=(Hpad,Wpad))(inputs)
    x =   unet_block(x, kshape,channels)
    x =   Cropping2D(cropping=(Hpad,Wpad))(x)
    # Add DC block
    x =   DC_block(x,mask,inputs,channels,kspace=kspace_flag)
    x2 =   Lambda(ifft_layer)(x)                
    #out_k1_pos = len(layers
    
    # JUNÇÃO
    up1 = concatenate([x1, x2], axis=-1)

    #PARTE DE BAIXO    
    #I-2
    x = Lambda(ifft_layer)(inputs)
    kspace_flag = False
    x = ZeroPadding2D(padding=(Hpad,Wpad))(x)
    x = unet_block(x, kshape,channels)
    x = Cropping2D(cropping=(Hpad,Wpad))(x)
    x3 = DC_block(x,mask,inputs,channels,kspace=kspace_flag)

    #K-2
    kspace_flag = True
        # Add CNN block
    x = ZeroPadding2D(padding=(Hpad,Wpad))(inputs)
    x = unet_block(x, kshape,channels)
    x = Cropping2D(cropping=(Hpad,Wpad))(x)
    # Add DC block
    x4 = DC_block(x,mask,inputs,channels,kspace=kspace_flag)
                      
    up2 = concatenate([x3, x4], axis=-1)

    
    x = Lambda(ifft_layer)(up1)
    kspace_flag = False
    x = ZeroPadding2D(padding=(Hpad,Wpad))(x)
    x = unet_block(x, kshape,48)
    x = Cropping2D(cropping=(Hpad,Wpad))(x)
    x = DC_block(x,mask,inputs,channels,kspace=kspace_flag)
    x5 = Lambda(ifft_layer)(x)    
                            
                  
    kspace_flag = True
    x = ZeroPadding2D(padding=(Hpad,Wpad))(up2)
    x = unet_block(x, kshape,48)
    x = Cropping2D(cropping=(Hpad,Wpad))(x)
    x = DC_block(x,mask,inputs,channels,kspace=kspace_flag)
    x6 = Lambda(ifft_layer)(x)           
          
    up3 = concatenate([x5, x6], axis=-1)
                  
    kspace_flag = False
    x = ZeroPadding2D(padding=(Hpad,Wpad))(up3)
    x = unet_block(x, kshape,48)
    x = Cropping2D(cropping=(Hpad,Wpad))(x)
    x = DC_block(x,mask,inputs,channels,kspace=kspace_flag)
                  
    out = Lambda(ifft_layer)(x)                 
                
    model = Model(inputs=[inputs,mask], outputs=out)
    return model
