import os

# PATHS
BASE_DATA = 'data/'
BASE_OUTPUT = 'output/'
BASE_LOGS = 'logs/'
ORIGINAL_SC_DATASET = os.path.join(BASE_DATA, 'originals/sc/')
ORIGINAL_SC_SIZED_DATASET = os.path.join(BASE_DATA, 'originals/sc_org_size/')
SC_MENOS = 'CACSmenos400'
SC_MAS = 'CACSmas400'

BASE_CLINICAL = os.path.join(BASE_DATA, 'clinical/base_clinical_data.xlsx')
CLINICAL_PREDICTION = os.path.join(BASE_DATA, 'clinical/clinical_data_prediction.xlsx')
SCORE_CALCIUM_DATA_ORG = os.path.join(BASE_DATA, 'sc_run_resized')
SCORE_CALCIUM_DATA = os.path.join(BASE_DATA, 'sc_run')
GLAUCOMA_DATA = os.path.join(BASE_DATA, 'originals/glaucoma/')

# FOLD_PREDICTIONS = 'tmp/prediction_data.pckl'
FOLD_PREDICTIONS = 'tmp/discr_preds.pckl'

TRAIN = 'train'
TEST = 'test'
TRAIN_TEST = 'train_test'

# Models
MODELS = {
    'init_architecture': {
        'vgg16': 'model_init_vgg16.pt',
        'vgg19': 'model_init_vgg19.pt',
        'resnet18': 'model_init_resnet18.pt',
    },
    'glaucoma': 'model_glaucoma.pt',
}
