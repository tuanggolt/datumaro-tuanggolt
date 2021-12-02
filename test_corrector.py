# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
import logging
import sys
import numpy as np

from datumaro.components.annotation import Label, Bbox
from datumaro.components.extractor import DatasetItem
from datumaro.components.dataset import Dataset
from datumaro.components.validator import Validator
from corrector import Corrector

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
    level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Script for copying dataset')
    parser.add_argument('-s', '--source', type=str, help='Path to source')
    parser.add_argument('-d', '--destination', type=str, help='Path to destination')
    parser.add_argument('-f', '--format', type=str, default='coco',
        help='Source format (default: %(default)s)')
    parser.add_argument('-t', '--task', type=str, default='detection',
        help='Task type (default: %(default)s)')
    return parser.parse_args()

def create_dataset_for_test() -> Dataset:
    dataset = Dataset.from_iterable([
        #datasetitem to reproduce MissingAttribute
        DatasetItem(id='1', subset='train',
            image=np.ones((10, 20, 3)),
            annotations=[
                Label(0),
                Bbox(100, 200, 100, 150, label=0),
        ]),
        #datasetitem to reproduce MultiLabelAnnotations
        DatasetItem(id='2', subset='val',
            image=np.ones((10, 20, 3)),
            annotations=[
                Label(0, id=0, attributes={'truncated': 1, 'difficult': 2}),
                Label(1, id=1, attributes={'truncated': 1, 'difficult': 2}),
                Label(2, id=2, attributes={'truncated': 1, 'difficult': 2}),
                Label(3, id=3, attributes={'truncated': 1, 'difficult': 2}),
        ]),
        #datasetitem to reproduce MissingAnnotation
        DatasetItem(id='3', subset='test',
        image=np.ones((10, 20, 3))),
    ], categories=[('cat', '', ['truncated', 'difficult']),
           ('dog', '', ['truncated', 'difficult']),
           ('person', '', ['truncated', 'difficult']),
           ('car', '', ['truncated', 'difficult']),])
    return dataset

def main():
    args = parse_args()

    #dataset = Dataset.import_from(args.source, args.format)
    dataset = create_dataset_for_test()
    validator: Validator = dataset.env.validators[args.task]()
    errors = validator.validate(dataset)

    if errors['validation_reports']:
        print("There are some issues:")
        print("- errors:", errors['summary']['errors'])
        print("- warnings:", errors['summary']['warnings'])

        idx = 0
        ids_to_delete_missing_attribute = set()
        ids_to_delete_multilabel_annotations = set()
        ids_to_delete_missing_annotation = set()
        for idx, error in zip(range(30), errors['validation_reports']):
            print("%s: %s" % (idx, error))

            if error['anomaly_type'] == 'MissingAttribute':
                ids_to_delete_missing_attribute.add(error['item_id'])
            elif error['anomaly_type'] == 'MultiLabelAnnotations':
                ids_to_delete_multilabel_annotations.add(error['item_id'])
            elif error['anomaly_type'] == 'MissingAnnotation':
                ids_to_delete_missing_annotation.add(error['item_id'])

        print("dataset: ", len(dataset))

        dataset_fixed = Corrector.delete_dataset_items(dataset, ids_to_delete_missing_attribute)
        dataset_fixed = Corrector.delete_dataset_items(dataset_fixed, ids_to_delete_multilabel_annotations)
        dataset_fixed = Corrector.delete_dataset_items(dataset_fixed, ids_to_delete_missing_annotation)

        print("dataset_fixed: ", len(dataset_fixed))


if __name__ == "__main__":
    main()
