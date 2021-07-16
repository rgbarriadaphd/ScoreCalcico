import logging
import random
import os
import shutil
from distutils.dir_util import copy_tree

OUTPUT_MAS = 'CACSmas400'
OUTPUT_MENOS = 'CACSmenos400'

# ORG_BASE = 'data/originals/sc/'
# ORG_MAS = os.path.join(ORG_BASE, OUTPUT_MAS)
# ORG_MENOS = os.path.join(ORG_BASE, OUTPUT_MENOS)

# FOLDS_BASE = 'data/folds/'
FOLD_ID = 'fold_'

# RUN_BASE = 'data/sc_run'

class ScoreCalciumSelection:

    def __init__(self, fold_base, run_base, originals_base, criteria='5-fold'):
        """
        :param criteria:
        """
        logging.info(f'Running experiment based on {criteria} cross validation criteria')

        self._criteria = criteria
        self._fold_base = fold_base
        self._originals_base = originals_base
        self._run_base = run_base

        self._org_mas = os.path.join(self._originals_base, OUTPUT_MAS)
        self._org_menos = os.path.join(self._originals_base, OUTPUT_MENOS)

        if self._criteria == 'leave-one-out':
            self._short_inputs()
        self._nfolds = 5 if criteria=='5-fold' else 10

        fold_length = int(len(os.listdir(self._org_menos)) / self._nfolds)

        # self.create_folds(fold_length, self._nfolds)

    def _short_inputs(self):
        if not self._criteria == 'leave-one-out':
            return

        self._samples = []

        for sample in os.listdir(self._org_mas):
            self._samples.append(os.path.join(self._org_mas, sample).split('sc/')[1])
        for sample in os.listdir(self._org_menos):
            self._samples.append(os.path.join(self._org_menos, sample).split('sc/')[1])

        # mix images in order to not having positive and negative in a row
        random.shuffle(self._samples)


    def generate_run_set(self, test_fold_id=1):

        test_set = {OUTPUT_MAS: [], OUTPUT_MENOS: [] }
        train_set = {OUTPUT_MAS: [], OUTPUT_MENOS: [] }
        if os.path.isdir(self._run_base):
            shutil.rmtree(self._run_base)
        os.mkdir(self._run_base)
        os.mkdir(os.path.join(self._run_base, 'train'))
        os.mkdir(os.path.join(self._run_base, 'train', OUTPUT_MAS))
        os.mkdir(os.path.join(self._run_base, 'train', OUTPUT_MENOS))
        os.mkdir(os.path.join(self._run_base, 'test'))
        os.mkdir(os.path.join(self._run_base, 'test', OUTPUT_MAS))
        os.mkdir(os.path.join(self._run_base, 'test', OUTPUT_MENOS))

        os.mkdir(os.path.join(self._run_base, 'train_test'))
        os.mkdir(os.path.join(self._run_base, 'train_test', OUTPUT_MAS))
        os.mkdir(os.path.join(self._run_base, 'train_test', OUTPUT_MENOS))

        if self._criteria == 'leave-one-out':
            test_sample = self._samples[test_fold_id]

            for sample in self._samples:
                org = os.path.join(self._originals_base, sample)
                if sample == test_sample:
                    dst = os.path.join(self._run_base, 'test', sample)
                else:
                    dst = os.path.join(self._run_base, 'train', sample)
                shutil.copyfile(org, dst)
        else:
            train_folds_mas = []
            train_folds_menos = []
            for fold_id in range(0, self._nfolds):
                org_mas = os.path.join(self._fold_base, FOLD_ID + str(fold_id + 1), OUTPUT_MAS)
                org_menos = os.path.join(self._fold_base, FOLD_ID + str(fold_id + 1), OUTPUT_MENOS)
                if fold_id + 1 == test_fold_id:
                    test_set[OUTPUT_MAS] = [item.split('.')[0] for item in os.listdir(org_mas)]
                    test_set[OUTPUT_MENOS] = [item.split('.')[0] for item in os.listdir(org_menos)]
                    # Copy test fold
                    dst_mas = os.path.join(self._run_base, 'test', OUTPUT_MAS)
                    dst_menos = os.path.join(self._run_base, 'test', OUTPUT_MENOS)
                else:
                    train_folds_mas.append(os.listdir(org_mas))
                    train_folds_menos.append(os.listdir(org_menos))
                    # Rest of train folds
                    dst_mas = os.path.join(self._run_base, 'train', OUTPUT_MAS)
                    dst_menos = os.path.join(self._run_base, 'train', OUTPUT_MENOS)
                copy_tree(org_mas, dst_mas)
                copy_tree(org_menos, dst_menos)

                copy_tree(org_mas, os.path.join(self._run_base, 'train_test', OUTPUT_MAS))
                copy_tree(org_menos, os.path.join(self._run_base, 'train_test', OUTPUT_MENOS))

            train_set[OUTPUT_MAS] = [item.split('.')[0] for sublist in train_folds_mas for item in sublist]
            train_set[OUTPUT_MENOS] = [item.split('.')[0] for sublist in train_folds_menos for item in sublist]

        return train_set, test_set
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






