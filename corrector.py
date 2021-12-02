# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

from typing import List

from datumaro.components.dataset import Dataset


class Corrector:

    @staticmethod
    def delete_dataset_items(
        dataset: Dataset, item_ids_to_delete: List[str]
    ) -> Dataset:
        """
        Returns the dataset from which datasetitems of the received item_ids_to_delete have been removed.

        :param dataset: Dataset, which consists of datasetitems
        :param item_ids_to_delete: a list with datasetitem ids to be deleted
        :return Dataset from which items to be deleted have been removed
        """
        if len(item_ids_to_delete) > 0:
            id_subset = {}
            for item in dataset:
                id_subset[item.id] = item.subset

            for id in item_ids_to_delete:
                dataset.remove(id, id_subset[id])

        return dataset
