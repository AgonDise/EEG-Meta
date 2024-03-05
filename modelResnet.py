import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, Add, MaxPooling2D, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

def resnet_identity_block(x, filters, kernel_size):
    """
    A block that has no conv layer at shortcut.
    
    Arguments:
    x -- input tensor
    filters -- list of integers, the filters of 3 conv layers at the main path
    kernel_size -- default 3, the kernel size of middle conv layer at main path
    
    Returns:
    x -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """
    
    # Retrieve Filters
    F1, F2 = filters
    
    # Save the input value. You'll need this later to add back to the main path.
    x_shortcut = x
    
    # First component of main path
    x = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1,1), padding='valid')(x)
    x = BatchNormalization(axis=3)(x)
    x = ReLU()(x)
    
    # Second component of main path (≈3 lines)
    x = Conv2D(filters=F2, kernel_size=(kernel_size, kernel_size), strides=(1,1), padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = ReLU()(x)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    x = Add()([x_shortcut, x])
    x = ReLU()(x)
    
    return x

def resnet_conv_block(x, filters, kernel_size, strides):
    """
    A block that has a conv layer at shortcut.
    
    Arguments:
    x -- input tensor
    filters -- list of integers, the filters of 3 conv layers at the main path
    kernel_size -- default 3, the kernel size of middle conv layer at main path
    strides -- integer, specifying the stride to be used
    
    Returns:
    x -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """
    
    # Retrieve Filters
    F1, F2 = filters
    
    # Save the input value
    x_shortcut = x

    # First component of main path 
    x = Conv2D(F1, (1, 1), strides=(strides,strides))(x)
    x = BatchNormalization(axis=3)(x)
    x = ReLU()(x)

    # Second component of main path (≈3 lines)
    x = Conv2D(filters=F2, kernel_size=(kernel_size, kernel_size), strides=(1,1), padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = ReLU()(x)

    # SHORTCUT PATH 
    x_shortcut = Conv2D(F2, (1, 1), strides=(strides,strides), padding='valid')(x_shortcut)
    x_shortcut = BatchNormalization(axis=3)(x_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    x = Add()([x_shortcut, x])
    x = ReLU()(x)

    return x

def ResNet50(input_shape=(128, 256, 1), classes=2):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER
    
    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes
    
    Returns:
    model -- a Model() instance in Keras
    """
    
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)

    # Stage 1
    X = Conv2D(64, (7, 7), strides=(2, 2))(X)
    X = BatchNormalization(axis=3)(X)
    X = ReLU()(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = resnet_conv_block(X, filters=[64, 256], kernel_size=3, strides=1)
    X = resnet_identity_block(X, filters=[64, 256], kernel_size=3)
    X = resnet_identity_block(X, filters=[64, 256], kernel_size=3)

    # Stage 3 (≈4 lines)
    X = resnet_conv_block(X, filters=[128, 512], kernel_size=3, strides=2)
    X = resnet_identity_block(X, filters=[128, 512], kernel_size=3)
    X = resnet_identity_block(X, filters=[128, 512], kernel_size=3)
    X = resnet_identity_block(X, filters=[128, 512], kernel_size=3)

    # Stage 4 (≈6 lines)
    X = resnet_conv_block(X, filters=[256, 1024], kernel_size=3, strides=2)
    X = resnet_identity_block(X, filters=[256, 1024], kernel_size=3)
    X = resnet_identity_block(X, filters=[256, 1024], kernel_size=3)
    X = resnet_identity_block(X, filters=[256, 1024], kernel_size=3)
    X = resnet_identity_block(X, filters=[256, 1024], kernel_size=3)
    X = resnet_identity_block(X, filters=[256, 1024], kernel_size=3)

    # Stage 5 (≈3 lines)
    X = resnet_conv_block(X, filters=[512, 2048], kernel_size=3, strides=2)
    X = resnet_identity_block(X, filters=[512, 2048], kernel_size=3)
    X = resnet_identity_block(X, filters=[512, 2048], kernel_size=3)

    # AVGPOOL
    X = GlobalAveragePooling2D()(X)

    # Output layer
    X = Dense(classes, activation='softmax', name='fc' + str(classes))(X)
    
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='ResNet50')

    return model
