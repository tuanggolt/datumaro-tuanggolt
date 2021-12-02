# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
import logging
import sys

from distutils.util import strtobool
from datumaro import AnnotationType, Bbox, Label, LabelCategories, Dataset, DatasetItem, Validator

import numpy as np

from datumaro import SourceExtractor
from datumaro.components.extractor import ErrorAction, ErrorPolicy



logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
    level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=str, help='Path to source')
    parser.add_argument('-f', '--format', type=str, default='coco',
        help='Source format (default: %(default)s)')
    parser.add_argument('-t', '--task', type=str, default='detection',
        help='Task type (default: %(default)s)')
    return parser.parse_args()


class MyErrorPolicy(ErrorPolicy):
    def get_action(self, error, supported_actions) -> ErrorAction:
        action = super().get_action(error, supported_actions)
        print(type(error), error, supported_actions, '; action chosen:', action)
        return action

class AskingErrorPolicy(ErrorPolicy):
    def get_action(self, error, supported_actions) -> ErrorAction:
        print("An error found in the dataset:", type(error), error)
        value = None
        while value not in supported_actions:
            value = input(f"Available actions: {', '.join(supported_actions)}: ")
            value = value.strip()
        return value




class MyExtractor(SourceExtractor):
    def __init__(self, url, import_context):
        super().__init__()

        self._import_ctx = import_context

        self._categories = {
            AnnotationType.label: LabelCategories.from_iterable(
                ['cat', 'dog', 'person', 'car'])
        }

    def __iter__(self):
        action = self._import_ctx.report_error('problem 1', ['skip', 'add_label'])
        print('problem 1 - action: ', action)
        if action == 'skip':
            pass
        elif action == 'add_label':
            self._categories[AnnotationType.label].add('_extra1')

        self._import_ctx.report_error('problem 2', ['ignore'])

        yield from [
            DatasetItem(id='1', subset='train',
                image=np.ones((10, 20, 3)),
                annotations=[
                    Label(0),
                    Label(1),
                    Bbox(0, 1, 4, 5, label=2),
                    Bbox(2, 3, 3, 2, label=1),
            ]),

            DatasetItem(id='2', subset='val',
                image=np.ones((10, 20, 3)))
        ]


def import_dataset(*, error_policy=None):
    policy = MyErrorPolicy()
    extractor = MyExtractor('path/', import_context=policy)
    dataset = Dataset.from_extractors(extractor)

    # dataset = Dataset.from_iterable([
    #     DatasetItem(id='1', subset='train',
    #         image=np.ones((10, 20, 3)),
    #         annotations=[
    #             Label(0),
    #             Label(1),
    #             Bbox(0, 1, 4, 5, label=2),
    #             Bbox(2, 3, 3, 2, label=1),
    #     ]),

    #     DatasetItem(id='2', subset='val',
    #         image=np.ones((10, 20, 3))),
    # ], categories=['cat', 'dog', 'person', 'car'])

    import_errors = [
        {'item': ('1', 'train'), 'message': "Annotation #1: Attribute 'z' is not defined in the dataset metainfo"},
        {'item': ('1', 'train'), 'message': "Annotation #2: Undefined label id #6"},
        {'item': ('1', 'train'), 'message': "Annotation #3: Mandatory attribute 'z' has no value"},
        {'item': ('1', 'train'), 'message': "Bounding boxes #3 and #4 overlap"},
        {'item': ('1', 'train'), 'message': "Multiple Label annotations in the item"},
        {'message': "No 'colormap.txt' file is found"},
        {'message': "Label #3 (car) has no annotations"},
        {'item': ('2', 'val'), 'message': "Item has no annotations"},
        {'item': ('2', 'val'), 'message': "Annotation #1: The 'x' coordinate has invalid value (nan)"},
    ]

    return dataset, import_errors

def main():
    args = parse_args()

    print("Importing...")

    dataset = Dataset.import_from(args.source, args.format,
        error_policy=AskingErrorPolicy())

    for item in dataset:
        print(item.annotations)

    ## Batch approach
    # dataset, loading_errors = import_dataset()

    # if loading_errors:
    #     print("There are some loading errors (%s total):" % len(loading_errors))

    #     skip = strtobool(input('Skip [Y/n]? ') or 'y')
    #     if not skip:
    #         for error in loading_errors:
    #             print("  ",
    #                 ("Item %s" % (error['item'], )) if 'item' in error else 'Dataset',
    #                 error['message']
    #             )

    #             skip = 's' == (input('  Skip / try to fix [S/f]? ') or 's').lower()
    #             if skip:
    #                 error['action'] = 'skip'
    #             else:
    #                 error['action'] = 'fix'

    #         print("Reloading with fixes...")
    #         dataset, _ = import_dataset(error_policy=loading_errors)

    print("Dataset is loaded")

    print("Validation...")

    validator: Validator = dataset.env.validators[args.task]()
    errors = validator.validate(dataset)

    if errors['validation_reports']:
        print("There are some problems found:")
        print("- errors:", errors['summary']['errors'])
        print("- warnings:", errors['summary']['warnings'])

        idx = 0
        for idx, error in zip(range(10), errors['validation_reports']):
            print("%s: %s" % (idx, error))
        if len(errors['validation_reports']) - idx:
            print("... (and %s more)" % \
                (len(errors['validation_reports']) - idx))

        skip = strtobool(input('skip [Y/n]?') or 'y')
        if not skip:
            sys.exit(1)

    print("Labels: ")
    for idx, label in enumerate(dataset.categories()[AnnotationType.label]):
        print('  #%s: %s (parent: %s, known attrs: %s)' % (
            idx, label.name, label.parent, ', '.join(label.attributes)
        ))

    new_labels = input("Enter the new labels "
            "[- to skip / <,-separated list, : for parent>]:"
        ) or '-'
    new_labels = new_labels.strip()
    if new_labels != '-':
        new_labels = [
            name.strip().split(':')
            for name in new_labels.split(',')
        ]
        new_labels = LabelCategories.from_iterable(new_labels)

        if new_labels:
            dataset.transform('project_labels', new_labels)

        print("New labels: ")
        for idx, label in enumerate(dataset.categories()[AnnotationType.label]):
            print('  #%s: %s (parent: %s, known attrs: %s)' % (
                idx, label.name, label.parent, ', '.join(label.attributes)
            ))


if __name__ == "__main__":
    main()
