import torch
from abc import abstractmethod
from torch.utils.data import IterableDataset
import importlib

def beta_scheduler(n_timestep=1000, linear_start=1e-4, linear_end=2e-2):
    betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
    return betas


class Txt2ImgIterableBaseDataset(IterableDataset):
    '''
    Define an interface to make the IterableDatasets for text2img data chainable
    '''
    def __init__(self, num_records=0, valid_ids=None, size=256):
        super().__init__()
        self.num_records = num_records
        self.valid_ids = valid_ids
        self.sample_ids = valid_ids
        self.size = size

        print(f'{self.__class__.__name__} dataset contains {self.__len__()} examples.')

    def __len__(self):
        return self.num_records

    @abstractmethod
    def __iter__(self):
        pass


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)