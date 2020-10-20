from Mask.config.config import Config


class MolesConfig(Config):
        ''' 
        MolesConfig:
            Contain the configuration for the dataset + those in Config
        '''
        NAME = "moles"
        GPU_COUNT = 1 # put 2 or more if you are 1 or more gpu
        IMAGES_PER_GPU = 1 # if you are a gpu you are choose how many image to process per gpu
        NUM_CLASSES = 1 + 2  # background + (malignant , benign)
        IMAGE_MIN_DIM = 128
        IMAGE_MAX_DIM = 128

        # hyperparameter
        LEARNING_RATE = 0.001
        STEPS_PER_EPOCH = 100
        VALIDATION_STEPS = 5
    