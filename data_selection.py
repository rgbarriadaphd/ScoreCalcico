import logging
import random
import os
import shutil
from distutils.dir_util import copy_tree

OUTPUT_MAS = 'CACSmas400'
OUTPUT_MENOS = 'CACSmenos400'

ORG_BASE = 'data/originals/sc/'
ORG_MAS = os.path.join(ORG_BASE, OUTPUT_MAS)
ORG_MENOS = os.path.join(ORG_BASE, OUTPUT_MENOS)

FOLDS_BASE = 'data/folds/'
FOLD_ID = 'fold_'

RUN_BASE = 'data/sc_run'

class ScoreCalciumSelection:

    def __init__(self, criteria='5-fold'):
        """

        :param criteria:
        """
        logging.info(f'Running experiment based on {criteria} cross validation criteria')

        self._criteria = criteria
        self._short_inputs()
        self._nfolds = 5 if criteria=='5-fold' else 10

        fold_length = int(len(os.listdir(ORG_MENOS)) / self._nfolds)

        # self.create_folds(fold_length, self._nfolds)

    def _short_inputs(self):
        if not self._criteria == 'leave-one-out':
            return

        self._samples = []

        for sample in os.listdir(ORG_MAS):
            self._samples.append(os.path.join(ORG_MAS, sample).split('sc/')[1])
        for sample in os.listdir(ORG_MENOS):
            self._samples.append(os.path.join(ORG_MENOS, sample).split('sc/')[1])

        # mix images in order to not having positive and negative in a row
        random.shuffle(self._samples)


    def generate_run_set(self, test_fold_id=1):

        if os.path.isdir(RUN_BASE):
            shutil.rmtree(RUN_BASE)
        os.mkdir(RUN_BASE)
        os.mkdir(os.path.join(RUN_BASE, 'train'))
        os.mkdir(os.path.join(RUN_BASE, 'train', OUTPUT_MAS))
        os.mkdir(os.path.join(RUN_BASE, 'train', OUTPUT_MENOS))
        os.mkdir(os.path.join(RUN_BASE, 'test'))
        os.mkdir(os.path.join(RUN_BASE, 'test', OUTPUT_MAS))
        os.mkdir(os.path.join(RUN_BASE, 'test', OUTPUT_MENOS))

        if self._criteria == 'leave-one-out':
            test_sample = self._samples[test_fold_id]

            for sample in self._samples:
                org = os.path.join(ORG_BASE, sample)
                if sample == test_sample:
                    dst = os.path.join(RUN_BASE, 'test', sample)
                else:
                    dst = os.path.join(RUN_BASE, 'train', sample)
                shutil.copyfile(org, dst)
        else:
            for fold_id in range(0, self._nfolds):
                org_mas = os.path.join(FOLDS_BASE, FOLD_ID + str(fold_id + 1), OUTPUT_MAS)
                org_menos = os.path.join(FOLDS_BASE, FOLD_ID + str(fold_id + 1), OUTPUT_MENOS)
                if fold_id + 1 == test_fold_id:
                    # Copy test fold
                    dst_mas = os.path.join(RUN_BASE, 'test', OUTPUT_MAS)
                    dst_menos = os.path.join(RUN_BASE, 'test', OUTPUT_MENOS)
                else:
                    # Rest of train folds
                    dst_mas = os.path.join(RUN_BASE, 'train', OUTPUT_MAS)
                    dst_menos = os.path.join(RUN_BASE, 'train', OUTPUT_MENOS)
                copy_tree(org_mas, dst_mas)
                copy_tree(org_menos, dst_menos)



    # def create_folds(self, length, nfolds):
    #     # Create root sc_folds. Remove if exists
    #     if os.path.isdir(FOLD_ROOT):
    #         shutil.rmtree(FOLD_ROOT)
    #     os.mkdir(FOLD_ROOT)
    #
    #     for fold in range(nfolds):
    #         # create fold folder
    #         os.mkdir(os.path.join(FOLD_ROOT,'fold_' + str(fold + 1)))
    #
    #         for img in os.listdir(ORG_MENOS):
    #             print(img)






