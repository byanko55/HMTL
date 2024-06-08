from .imgset import *
from pipeline.op import *
    

class MultiSet(Dataset):
    """
    # In
      - data_sources: dictionary of torch.utils.data.Dataset
      - num_samples: number of samples will be chosen from each data sources
    """
    def __init__(self, data_sources:dict, num_samples:int) -> None:
        self.ns = num_samples
        self.sn = list(data_sources.keys()) # data source names
        self.nds = len(self.sn)
        self.x = {}; self.y = {}; self.indices = {}

        for ds_name, raw_data in data_sources.items():
            train_indices = np.random.choice(len(raw_data), num_samples, replace=True)
            x_, y_ = raw_data[train_indices]
            self.x[ds_name] = np.array(x_)
            self.y[ds_name] = np.array(y_)
            self.indices[ds_name] = train_indices

    def rawdata(self) -> tuple:
        return self.x, self.y, self.indices

    def __len__(self) -> int:
        return self.nds

    def __getitem__(self, item:list) -> tuple:
        imgs, labels = zip(*[(self.x[k][item], self.y[k][item]) for k in self.sn])

        return imgs, labels


class FusionLoader():
    """
    # In
      - data_sources: dictionary of torch.utils.data.Dataset
    """
    def __init__(self, data_sources:dict, cuda:bool = True) -> None:
        self.rds = data_sources # original(raw) data samples
        self.cuda = cuda
        self.sn = list(data_sources.keys()) # data source names
        self.nt_all = 0 # number of (all) training samples
        self.nt = [] # number of training samples from each data sources
        self.lds = {'train':{}, 'eval':{}, 'test':{}} # 

    """
    Load sample data from different data sources,
    and then split each data set according to train/val/test ratio 
    (test_ratio = 1.0 - train_ratio - val_ratio)
    
    # In
      - train_ratio, val_ratio: 
    """
    def load(self, train_ratio:float = 0.7, val_ratio:float = 0.1) -> None:
        assert train_ratio + val_ratio < 1.0

        for k,v in self.rds.items():
            num_samples = len(v)
            train_samples = int(train_ratio * num_samples)
            val_samples = int(val_ratio * num_samples)

            self.nt_all += train_samples
            self.nt.append(train_samples)
            indices = np.random.shuffle(np.arange(num_samples))

            self.lds['train'][k] = indices[:train_samples]
            self.lds['eval'][k] = indices[train_samples:train_samples+val_samples]
            self.lds['test'][k] = indices[train_samples+val_samples:]


    """
    Build a training batch

    # Note
    this batch is temporary,
    which means the one used in previous training epoch will be discarded subsequently;
    In every epochs, the 'FusionSet' creates new batch consisting arbitrary set of sample data.
    
    # In
      - minibatch_size: 
      - sampling_rate: you want to adjust the full-batch size according to this argument
    # Out
      - tb: DataLoader 
    """
    def trainbatch(self, minibatch_size:int = 256, sampling_rate:float = 1.0) -> DataLoader:
        if not self.lds['train']:
            raise FusionException("[fuset: trainbatch]\'load\' the dataset first!!")

        k = int(self.nt * sampling_rate)
        fs = FusionSet(data_sources=self.lds['train'], num_samples=k)

        tb = DataLoader(
            dataset=fs, 
            batch_size=minibatch_size,
            pin_memory=self.cuda
        )

        return tb

    """
    Build a evaluation/test batch

    # In
      - ds_names: specify what data sources should be evaluated.
      - minibatch_size: 
      - eval: if true, the generated batch will be used in eval-mode (otherwise, test-mode). 
    # Out
      - tb: dictionary of torch.utils.data.DataLoader 
    """
    def evalbatch(self, ds_names:list, minibatch_size:int = 256, eval:bool = True) -> dict:
        if not self.lds['eval']:
            raise FusionException("[fuset: evalbatch]\'load\' the dataset first!!")
        
        eb = {}

        for ds_name in ds_names:
            if eval :
                eb[ds_name] = DataLoader(
                    dataset=self.lds['eval'][ds_name], 
                    batch_size=minibatch_size,
                    pin_memory=self.cuda
                )
            else :
                eb[ds_name] = DataLoader(
                    dataset=self.lds['test'][ds_name], 
                    batch_size=minibatch_size,
                    pin_memory=self.cuda
                )

        return eb

    def dsweight(self) -> list:
        return self.nt/self.nt_all

    """
    Return name of each data sources
    """
    def dslist(self) -> list:
        return self.sn