import os

# PARAMETERS
EPOCHS = 80
BATCH_SIZE = 8
LEARNING_RATE = 0.0001
N_CLASSES = 2
N_SAMPLES = 154
# PATHS
BASE_DATA = 'data/'
BASE_OUTPUT = 'output/'
SCORE_CALCIUM_DATA = os.path.join(BASE_DATA, 'sc_run')
GLAUCOMA_DATA = os.path.join(BASE_DATA, 'originals/glaucoma/')
TRAIN = 'train'
TEST = 'test'

# Models
MODELS = {
    'init_architecture': {
        'vgg16': 'model_init_vgg16.pt',
        'vgg19': 'model_init_vgg19.pt',
        'resnet18': 'model_init_resnet18.pt',
    },
    'glaucoma': 'model_glaucoma.pt',
}
