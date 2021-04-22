import numpy as np

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


class DataLoaderBase(DataLoader):
    """ Challenge Dataloader """

    def __init__(self, dataset_loader: str, batch_size: int, shuffle: bool,
                    validation_split: float, nworkers: int, test_path: list, train_path: list = None):
        """ Constructor
        Args:
            train_path: path to the training dataset
            dataset_loader: dataset loader class
            batch_size: size of the batch
            shuffle: shuffles the data (only if validation data is not created)
            validation_split: decimal for the split of the validation
            nworkers: workers for the dataloader class
            test_path: path to the test dataset(s)
        """
        self.init_kwargs = {
            'batch_size': batch_size,
            'num_workers': nworkers,
            'shuffle': shuffle
        }

        self.test_path = test_path

        if not train_path:
            return self.get_test()

        self.train_dataset = self.dataset_loader(train_path[0])
        self.valid_dataset = self.dataset_loader(train_path[0])

        self.train_sampler = None
        self.valid_sampler = None

        if validation_split:
            self._split(validation_split)
            self.init_kwargs.pop('shuffle')

        super().__init__(self.train_dataset, sampler=self.train_sampler, **self.init_kwargs)

    def _split(self, validation_split: float):
        """ Creates a sampler to extract training and validation data
        Args:
            validation_split: decimal for the split of the validation
        """
        # random indices based off the validation split
        num_train = len(self.train_dataset)
        train_indices = np.array(range(num_train))
        validation_indices = np.random.choice(train_indices, int(
            num_train * validation_split), replace=False)

        train_indices = np.delete(train_indices, validation_indices)

        # subset the dataset
        train_idx, valid_idx = train_indices, validation_indices
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        self.train_sampler = train_sampler
        self.valid_sampler = valid_sampler

    def split_validation(self) -> DataLoader:
        """ Returns the validation data """
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(self.valid_dataset, sampler=self.valid_sampler, **self.init_kwargs)

    def get_test(self) -> list:
        """ Returns the test data """
        test_data = []
        for path in self.test_path:
            test_data.append(
                (path, DataLoader(self.dataset_loader(path), **self.init_kwargs)))
        return test_data
