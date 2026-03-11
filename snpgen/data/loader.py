import h5py
import numpy as np
import torch

from typing import Callable
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from snpgen.utils import is_list_like
from .utils import create_pad_mask, fast_assign, fast_onehot

class SplitDataset():
    def __init__(
            self,
            file_path,
            val_ratio=0.2,
            test_ratio=0.2,
            onehot=True,
            data_dtype=np.float32,
            channel_first=True,
            cast_targets=False,
            metadata=True,
            x_key='data',
            y_key='labels',
            seed=None,
            verbose=True
        ):
        """
        Initializes a SplitDataset object.

        Args:
            file_path (str): The file path to the dataset.
            val_ratio (float, optional): The ratio of validation data to the training data. Defaults to 0.2.
            test_ratio (float, optional): The ratio of test data to the total data. Defaults to 0.2.
            onehot (bool, optional): Whether to convert the **data** to one-hot encoding. Defaults to True.
            data_dtype (np.dtype or str, optional): The data type to cast the data to (None to keep the original type). Defaults to np.float32.
            channel_first (bool, optional): Whether to transpose the data to channel first format (i.e. batch_size, channels, num_classes) for one-hot encoding. Defaults to True.
            cast_targets (bool, optional): Whether to cast the targets to a float tensor. Defaults to False.
            metadata (bool or list, optional): Whether to load metadata. If a list is provided, only the specified metadata keys will be loaded. Defaults to True.
            seed (int, optional): The random seed for data splitting. Defaults to None.
            verbose (bool, optional): Whether to print detailed information during dataset initialization. Defaults to True.
        """
        self.file_path = file_path
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.x_key = x_key
        self.y_key = y_key
        self.onehot = onehot
        self.channel_first = channel_first
        
        print("Building dataset...")
        
        print("  Loading data...")
        
        self.data, self.targets, self.metadata = self._load_data(metadata=metadata)

        print("   Data loaded successfully.")

        if data_dtype is not None:
            if isinstance(data_dtype, str):
                if data_dtype.startswith('np.'):
                    data_dtype = data_dtype[3:]
                data_dtype = np.dtype(getattr(np, data_dtype))
        else:
            data_dtype = self.data.dtype
        
        print("  Preprocessing data...")            
        if onehot:
            # Convert data to one-hot encoding with correct dtype
            print("    Converting data to one-hot encoding...")
            self.data = fast_onehot(self.data, dtype=data_dtype)
            print("     Data converted to one-hot encoding.")
            if channel_first:
                self.data = np.transpose(self.data, (0, 2, 1)) # b, c, L
        else:
            # Cast data to the desired dtype if necessary
            if data_dtype != self.data.dtype:
                self.data = self.data.astype(data_dtype)

        # if len(self.targets.shape) == 1:
        #     self.targets = self.targets[:, np.newaxis]

        # If the targets are of any type of integer, convert them to int32
        # Useful for example to convert np.int8 to np.int32 in order to get a torch.tensor
        # of type torch.int32 and not torch.CharTensor
        if np.issubdtype(self.targets.dtype, np.integer):
            self.targets = self.targets.astype(np.int32)

        if cast_targets and self.targets.dtype.kind != 'f':
            self.targets = self.targets.astype(np.float32)

        print("   Data preprocessed successfully.")

        metadata_to_split = {}
        if self.metadata is not None:
            for k, v in self.metadata.items():
                # Split only metadata entries that have the same number of samples as data
                if isinstance(v, np.ndarray) and v.shape[0] == self.data.shape[0]:
                    metadata_to_split[k] = v
        metadata_train, metadata_val, metadata_test = None, None, None

        print("  Creating splits...")

        X_train_val, X_test, y_train_val, y_test, *metadata_split = train_test_split(
            self.data, self.targets, *list(metadata_to_split.values()), test_size=self.test_ratio, random_state=seed, stratify=self.targets)

        metadata_train_val = []
        if len(metadata_split) > 0:
            metadata_train_val = metadata_split[0::2]
            metadata_test = metadata_split[1::2]

        # Update stratify for second split
        X_train, X_val, y_train, y_val, *metadata_split = train_test_split(
            X_train_val, y_train_val, *metadata_train_val, test_size=self.val_ratio, random_state=seed, stratify=y_train_val)
        
        print("   Splits created successfully.")
        
        if len(metadata_split) > 0:
            metadata_train = metadata_split[0::2]
            metadata_val = metadata_split[1::2]
       
        self.train_data = X_train
        self.val_data = X_val
        self.test_data = X_test

        self.train_labels = y_train
        self.val_labels = y_val
        self.test_labels = y_test

        self.train_metadata = {k: v for k, v in zip(metadata_to_split.keys(), metadata_train)} if metadata_train else metadata_train
        self.val_metadata = {k: v for k, v in zip(metadata_to_split.keys(), metadata_val)} if metadata_val else metadata_val
        self.test_metadata = {k: v for k, v in zip(metadata_to_split.keys(), metadata_test)} if metadata_test else metadata_test

        # Keep them, could be useful
        self._train_val_data = X_train_val
        self._train_val_labels = y_train_val
        self._train_val_metadata = {k: v for k, v in zip(metadata_to_split.keys(), metadata_train_val)} if len(metadata_train_val)>0 else None

        if verbose:
            print(f"  Original target range: min={self.targets.min()}, max={self.targets.max()}")

        print("Dataset built successfully.")
        
    def _load_data(self, metadata=False):
        """
        Loads data, labels and metadata from the file.

        Returns:
            tuple: A tuple containing data, labels and metadata.
        """
        _metadata = None
        with h5py.File(self.file_path, 'r') as f:
            data = f[self.x_key][:]
            labels = f[self.y_key][:]
            if isinstance(metadata, bool) and metadata:
                # load any additional content as metadata
                _metadata = {k: f[k][:] for k in list(f.keys()) if k not in [self.x_key, self.y_key]}
            elif is_list_like(metadata):
                # load only the specified metadata keys
                _metadata = {k: f[k][:] for k in metadata if (k in f.keys() and k not in [self.x_key, self.y_key])}

        return data, labels, _metadata

    def get_split(self, split, metadata=False):
        """
        Returns the specified split of the data, labels and metadata.

        Args:
            split (str): The split to retrieve. Must be one of ['train', 'val', 'test', 'train_val', 'full'].
            metadata (bool, optional): Whether to return metadata. Default is False.

        Returns:
            tuple: A tuple containing the data and labels of the specified split, with metadata if requested.
        
        Raises:
            AssertionError: If the split is not one of ['train', 'val', 'test', 'train_val', 'full'].
        """
        assert split in ['train', 'val', 'test', 'train_val', 'full'], "split must be one of ['train', 'val', 'test', 'train_val, 'full']"
        
        if split == 'train':
            if metadata:
                return self.train_data, self.train_labels, self.train_metadata
            else:
                return self.train_data, self.train_labels
        elif split == 'val':
            if metadata:
                return self.val_data, self.val_labels, self.val_metadata
            else:
                return self.val_data, self.val_labels
        elif split == 'test':
            if metadata:
                return self.test_data, self.test_labels, self.test_metadata
            else:
                return self.test_data, self.test_labels
        elif split == 'train_val':
            if metadata:
                return self._train_val_data, self._train_val_labels, self._train_val_metadata
            else:
                return self._train_val_data, self._train_val_labels
        elif split == 'full':
            if metadata:
                return self.data, self.targets, self.metadata
            else:
                return self.data, self.targets
        else:
            return None
        
    def get_metadata(self, split, keys=None):
        """
        Retrieves metadata for a specified split and optional keys.
        The method returns metadata based on the specified split ('train', 'val', 'test', 'train_val', 'full')
        and optionally filters the metadata by the provided keys.
        
        Args:
            split (str): The data split to retrieve metadata for. Must be one of:
                'train', 'val', 'test', 'train_val', 'full'
            keys (str or list, optional): Key(s) to filter metadata by. Defaults to None.
                
        Returns:
            Various: If keys is None, returns the complete metadata for the specified split.
            If a single key is provided, returns the value for that key.
            If multiple keys are provided, returns a dictionary with the specified key-value pairs.
            Returns None if the split is invalid or no metadata exists for the specified split.
        Raises:
            AssertionError: If split is not one of the valid options.
        """

        assert split in ['train', 'val', 'test', 'train_val', 'full'], "split must be one of ['train', 'val', 'test', 'train_val, 'full']"
        
        if isinstance(keys, str):
            keys = [keys]
            
        metadata = None
        
        if split == 'train':
            metadata = self.train_metadata
        elif split == 'val':
            metadata = self.val_metadata
        elif split == 'test':
            metadata = self.test_metadata
        elif split == 'train_val':
            metadata = self._train_val_metadata
        elif split == 'full':
            metadata = self.metadata
        else:
            metadata = None
            
        if metadata is not None and keys is not None:
            if len(keys) == 1 and keys[0] in metadata:
                metadata = metadata[keys[0]]
            else:
                metadata = {k: metadata[k] for k in keys if k in metadata}
        
        return metadata
        

class SNPDataset(Dataset):
    """
    A custom dataset class for SNP data.

    Args:
        x (np.ndarray or tuple): The input data. If a tuple, it should contain two np.ndarray objects, 
                                where the first one represents the input data and the second one represents the targets.
        y (np.ndarray, optional): The target data. Only required if `x` is a single np.ndarray.
        seq_len (int, optional): The maximum sequence length to consider. Default is None.
        as_dict (bool, optional): Whether to return the data as a dictionary. Default is False.

    Raises:
        ValueError: If the input data is invalid.
    """

    def __init__(self, x, y=None, seq_len=None, as_dict=False, channel_first=True, **kwargs):
        self.as_dict = as_dict
        self.channel_first = channel_first
        if y is not None:
            self.data = self._truncate_array(x, seq_len)
            self.targets = y
        elif isinstance(x, (list, tuple)) and len(x) == 2 and isinstance(x[0], np.ndarray) and isinstance(x[1], np.ndarray):
            self.data = self._truncate_array(x[0], seq_len)
            self.targets = x[1]
        else:
            raise ValueError("Invalid input data.")
        
    def get_labels(self):
        return self.targets
    
    def _truncate_array(self, arr, seq_len):
        if self.channel_first:
            return arr[..., :seq_len]
        else:
            return arr[:, :seq_len, ...]
    
    def get_seq_len(self):
        if self.channel_first:
            return self.data.shape[-1]
        else:
            return self.data.shape[1]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        if self.as_dict:
            return {'x': x, 'y': y}
        else:
            return x, y
        
class SNPSequenceDataset(SNPDataset):
    """
    A dataset class for handling SNP sequence data.
    
    Args:
        x (numpy.ndarray, list, or tuple): Input data. Can be a single numpy array or a list/tuple containing 
            (data, targets) or (data, block_ids) or (data, targets, block_ids).
        y (numpy.ndarray, optional): Target labels. Required if `x` is a single array or a tuple of (data, block_ids).
        block_ids (numpy.ndarray, optional): Block IDs. Required if `x` is a single array or a tuple of (data, targets).
        seq_len (int, optional): Sequence length to truncate the data to.
        as_dict (bool, optional): If True, returns data as a dictionary. Default is False.
    
    Raises:
        ValueError: If input configurations are invalid or if the lengths of data, targets, and block_ids do not match.

    """
    def __init__(self, x, y=None, block_ids=None, seq_len=None, as_dict=False, patch_size=None, pad_value=None, channel_first=True, **kwargs):
        
        if patch_size is None:
            raise ValueError("Patch size must be provided")

        self.seq_len = seq_len
        self.as_dict = as_dict
        self.patch_size = patch_size
        self.pad_value = pad_value
        self.channel_first = channel_first
        
        # Unpack input data
        self.data, self.targets, self.block_ids = self._unpack_input(x, y, block_ids)
        
        # Truncate data if necessary
        self.data = self._truncate_array(self.data, seq_len)
        self.block_ids = self._truncate_array(self.block_ids, seq_len)
        
        if channel_first and self.data.ndim == 3:
            if self.data.shape[1] > self.data.shape[2]:
                print(f"Are you sure the data is in channel first format? Data shape: {self.data.shape}")
        
        # Pad data to make block lengths a multiple of patch_size
        self._already_padded = False
        self._pad_data()
            
        
    def _unpack_input(self, x, y, block_ids):
        # Convert tuple input to list for easier handling and do some basic validation
        if isinstance(x, (list, tuple)):
            x = list(x)
            if not all(isinstance(arr, np.ndarray) for arr in x):
                raise ValueError("All elements in input tuple/list must be numpy arrays")
            
            if len(x) not in (2, 3):
                raise ValueError("Input tuple/list must have 2 or 3 elements")
        
        # Case 1: x is just data, both y and block_ids provided as arguments
        if not isinstance(x, (list, tuple)):
            if y is None or block_ids is None:
                raise ValueError("If x is a single array, both y and block_ids must be provided")
            data = x
            targets = y
            block_ids = block_ids
            
        # Case 2: x is (data, block_ids) and y is provided
        elif len(x) == 2 and y is not None:
            data = x[0]
            block_ids = x[1]
            targets = y
            
        # Case 3: x is (data, targets) and block_ids is provided
        elif len(x) == 2 and block_ids is not None:
            data = x[0]
            targets = x[1]
            block_ids = block_ids
            
        # Case 4: x is (data, targets, block_ids)
        elif len(x) == 3:
            if y is not None or block_ids is not None:
                raise ValueError("If providing (data, targets, block_ids) tuple, y and block_ids arguments must be None")
            data = x[0]
            targets = x[1]
            block_ids = x[2]
            
        else:
            raise ValueError("Invalid input configuration")
            
        # Validate shapes
        if len(block_ids) not in data.shape[1:]:
            raise ValueError("Data, targets, and block_ids must have the same length")
            
        return data, targets, block_ids
  
    
    def _pad_data(self):
        """
        Pad each block in self.data to make its length a multiple of patch_size.

        Parameters:
        data (numpy.ndarray): The input data array. Can be either one-hot encoded (3D) or regular (2D).
                            Input shape:
                                - Regular data: (batch_size, seq_len)
                                - One-hot encoded data: (batch_size, seq_len, num_classes)
        block_ids (numpy.ndarray): 1D array of block IDs corresponding to the input data.
        patch_size (int): The desired patch size to pad the data to.
        pad_value (optional, int or float): The value to use for padding in the case of regular data. 
                                            In the case of one-hot encoded data, the padding value will be set to num_classes + 1.
                                            If None, the padding value will be set to one more than the maximum value in the data.
        
        Updates the instance variables:
        - self.data: The padded data.
        - self.block_ids: The padded block IDs.
        - self.pad_mask: A boolean mask indicating the padding positions.
        - self._already_padded: Flag indicating that the data has been padded.
        """
         
        if self._already_padded:
            return

        print("Start padding data...")
        pad_mask, padded_block_ids = create_pad_mask(self.block_ids, self.patch_size)
        
        # Check if input is one-hot encoded (has 3 dimensions)
        is_onehot = len(self.data.shape) == 3
               
        if is_onehot:
            if self.channel_first:
                # Transpose data to channel last:
                # (batch_size, num_classes, seq_len) -> (batch_size, seq_len, num_classes)
                self.data = np.transpose(self.data, (0, 2, 1))
                
            batch_size, seq_len, n_classes = self.data.shape
            # Create padded array with an extra class dimension for padding
            # Initialize with zeros
            result = np.zeros((batch_size, pad_mask.shape[0], n_classes + 1), dtype=self.data.dtype)
            # Copy original data for non-padding positions
            # result[:, ~pad_mask, :n_classes] = self.data # very slow if result is large
            result = fast_assign(x=result, y=self.data, idx=~pad_mask, third_axis_idx=np.arange(n_classes))
            # Set the padding positions to one-hot vector with 1 in the new class position
            # result[:, pad_mask, -1] = 1 # very slow if result is large
            result = fast_assign(x=result, y=1, idx=pad_mask, third_axis_idx=-1)
            # Set the padding value to None for one-hot encoded data
            self.pad_value = None
            
            if self.channel_first:
                # Transpose data back to channel first:
                # (batch_size, seq_len, num_classes) -> (batch_size, num_classes, seq_len)
                result = np.transpose(result, (0, 2, 1))
            
        else:
            # Handle regular data (non one-hot)
            pad_value = self.pad_value
            if pad_value is None:
                pad_value = np.max(self.data) + 1
            result = np.full((self.data.shape[0], pad_mask.shape[0]), pad_value, dtype=self.data.dtype)
            # result[:, ~pad_mask] = self.data # very slow if result is large
            result = fast_assign(x=result, y=self.data, idx=~pad_mask)
            
        self.data = result
        self.block_ids = padded_block_ids
        self.pad_mask = pad_mask
        self._already_padded = True
        
        print("Data padded successfully.")


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset

    Arguments:
        dataset: a torch.utils.data.Dataset object
        strategy: a strategy to calculate weights
        indices: a list of indices
        num_samples: number of samples to draw
        callback_get_label: a callback-like function which takes one argument: dataset
        seed: a random seed to be used. None to use the global random seed.
    """

    def __init__(
        self,
        dataset,
        strategy: str = 'inverse_freq',
        labels: list = None,
        indices: list = None,
        num_samples: int = None,
        callback_get_label: Callable = None,
        seed: int = None
    ):
        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        # distribution of classes in the dataset
        labels = self._get_labels(dataset) if labels is None else labels
        unique_labels, unique_counts = np.unique(labels, return_counts=True) # Get unique labels and their counts
        counts = unique_counts[np.searchsorted(unique_labels, labels)] # get the counts for each label in the original order

        weights = self._get_weights(counts, strategy=strategy)

        self.weights = torch.DoubleTensor(weights.tolist())

        self.gen = None
        if seed is not None:
            self.gen = torch.Generator()
            self.gen.manual_seed(seed)

    def _get_weights(self, counts, strategy):
        if strategy == 'inverse_freq':
            #https://github.com/ufoym/imbalanced-dataset-sampler/blob/01cb129677348824a20905baea112d501e3bf642/torchsampler/imbalanced.py#L43
            weights = 1.0 / counts
        elif strategy == 'effective_num':
            # https://github.com/zzw-zwzhang/LDAM-DRW/blob/3193f05c1e6e8c4798c5419e97c5a479d991e3e9/utils.py#L31
            beta = 0.9999
            effective_num = 1.0 - np.power(beta, counts)
            weights = (1.0 - beta) / np.array(effective_num)
        else:
            raise NotImplementedError(f"Strategy {strategy} not implemented")

        return weights

    def _get_labels(self, dataset):
        if self.callback_get_label:
            return self.callback_get_label(dataset)
        elif isinstance(dataset, torch.utils.data.Dataset):
            return dataset.get_labels()
        else:
            raise NotImplementedError

    def __iter__(self):
        rand_tensor = torch.multinomial(self.weights, self.num_samples, replacement=True, generator=self.gen)
        return (self.indices[i] for i in rand_tensor)

    def __len__(self):
        return self.num_samples