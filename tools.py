import numpy as np
import copy
import argparse
import h5py
import torch



## use argparse to configure experiment
# Experiment name is mandatory

parser = argparse.ArgumentParser(description='Train a PPO agent for the CarRacing-v0')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.99)')
parser.add_argument('--L2', type=float, default=0, help='L2 regularisation factor (1e-3 way too high, think 1e-6)')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--action-repeat', type=int, default=8, metavar='N',
                    help='repeat action in N frames (default: 4 used to be 8)')
parser.add_argument('--episode-length', type=int, default=500, metavar='N',
                    help='maximum steps per episode (old default: 1000)')
parser.add_argument('--img-stack', type=int, default=4, metavar='N', help='stack N image in a state (default: 4)')
parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
parser.add_argument('--buffer-size', type=int, default=2000, help='replay buffer capacity (default: 2000)')
parser.add_argument('--batch-size', type=int, default=128, help='batch size (default: 128)')
parser.add_argument('--fixed-speed', type=float, default=.1, help='Fixed velocity, 0 for disable (default: .1)')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument('--checkpoint', action='store_true', help='Continue from last model')
parser.add_argument('--vis', action='store_true', help='plot with visdom')
parser.add_argument("--force", action="store_true", help="overwrite name")
parser.add_argument("--domain-rand", action="store_true", help="Enable domain randomisation")

parser.add_argument("name", type=str, help="Experiment name") # Required

args = parser.parse_args()


# USAGE:
batch_size = args.batch_size
# do something with batch size



### Create a tensorboard with experiment `args.name`, and if `args.force` is set it will overwrite existing experiments.
# also logs all arguments as text to the tensorboard
def get_tensorboard(args):
    logdir_base = "logs"
    if args.name == "test":
        logdir_base = "/tmp/tensorboard"

    logdir = logdir_base + "/" + args.name + "/"

    if args.force:
        try:
            shutil.rmtree(logdir)
        except FileNotFoundError as e:
            pass

    if not os.path.exists(logdir_base):
        os.mkdir(logdir_base)

    prev = set(os.listdir(logdir_base))
    if args.name in prev:
        raise NameError("Experiment already exists")

    writer = SummaryWriter(logdir)
    writer.add_text("Info", str(vars(args)))
    return writer

# USAGE
# pass to tensorborad function to get a tensorboard writer
writer = get_tensorboard(args)

# log information to the tensorboard
writer.add_scalar('Reward/reward', score, training_step_or_epoch)


""" Class that saves data and labels as a .hdf5 file. """
class HDF5DatasetExtendable:
    VERSION = "1.0.0"

    def __init__(self, filename, data_type=np.float32, label_type=np.int, compression=None):
        """
        Set initial parameters for auto-resizable hdf5 dataset
        :param filename: The filenamae
        :param data_type: Datatype to use for storage, such as np.float32
        :param label_type: Datatype to use for storage, such as np.float32
        :param compression: None or "gzip"
        """
        assert ".hdf5" in filename, "Filename ust be .hdf5 file"
        self.filename = filename
        self.data_type = data_type
        self.label_type = label_type
        self.compression = compression
        self.initialized = False

    def __enter__(self):
        self.file = h5py.File(self.filename, "w")
        return self

    def add_metadata(self, info):
        """
        Add metedata to the dataset attributes. This data can be displayed when starting training, and
        informs the user of what this dataset contains. Be descriptive!
        Tip: a good starting point is just pass vars(args), such that all commandline options are logged.
        :param info: A dictionary with user information such as {"augmentation":"shifted"}
        """
        self.dataset.attrs["version"] = self.VERSION
        for k, v in info.items():
            self.dataset.attrs[k] = str(v)

    def _init(self, data, labels):
        """
        Initialize the dataset objects with the first batch of data.
        Do not call this function, always call append.
        :param data: numpy array containing the data, where the first axis is the sample index
        :param labels: numpy array containing the labels, where the first axis is the sample index
        """
        self.dataset = self.file.create_dataset(
            "data", np.shape(data), self.data_type, maxshape=(None,) + np.shape(data)[1:],
            data=data, chunks=True,
            compression=self.compression
        )
        self.labelset = self.file.create_dataset(
            "labels", np.shape(labels), self.label_type, maxshape=(None,) + np.shape(labels)[1:],
            data=labels, chunks=True,
            compression=self.compression
        )
        self.initialized = True

    def append(self, data, labels):
        """
        Add data to the dataset. If not initialized, this will copy the shapes from the first call and initialze.
        :param data: numpy array containing the data, where the first axis is the sample index
        :param labels: numpy array containing the labels, where the first axis is the sample index
        """
        if not self.initialized:
            self._init(data, labels)
            return

        shape = np.array(self.dataset.shape)
        shape[0] += data.shape[0]
        self.dataset.resize(shape)
        self.dataset[-data.shape[0]:, ...] = data

        shape = np.array(self.labelset.shape)
        shape[0] += labels.shape[0]
        self.labelset.resize(shape)
        self.labelset[-labels.shape[0]:, ...] = labels

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()


class DataGeneratorHDF5(torch.utils.data.Dataset):
    'Generates data for Keras'

    def __init__(self, filename, batch_size=32, dim=(16, 7, 2048), shuffle=True, verbose=True):
        'Initialization'
        self.batch_size = batch_size
        self.dim = dim
        self.shuffle = shuffle
        self.verbose = verbose

        self.filename = filename
        assert ".hdf5" in filename, "Not hdf5 file"

    def __enter__(self):
        self.file = h5py.File(self.filename, "r")

        self.length = len(self.file["data"])
        if self.verbose:
            print("Using dataset", self.filename)
            print("  Number of samples", self.length)
            print("  Metadata:")
            for k, v in self.file["data"].attrs.items():
                print("   ", k, v)

        self.X = self.file["data"]
        self.y = self.file["labels"]

        # Note: if split into training and test sets, these may not  be the same shape
        self.indexes = np.arange(len(self.X))

        self.on_epoch_end()

        return self

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.indexes) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        indexes = sorted(indexes)  # required for hdf5 indexing

        X = torch.from_numpy(self.X[indexes])
        y = torch.from_numpy(self.y[indexes])

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def split(self, factor=0.1):
        """ Split into training and validation sets, probably very not thread safe """
        split = int(len(self.indexes) * (1 - factor))
        train_indices, test_indices = self.indexes[:split], self.indexes[split:]
        test = copy.deepcopy(self)

        self.indexes = train_indices
        test.indexes = test_indices
        return self, test

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()

    @staticmethod
    def from_multiple():
        raise NotImplementedError()
        file_names_to_concatenate = ['1.h5', '2.h5', '3.h5', '4.h5']
        entry_key = 'data'  # where the data is inside of the source files.

        sources = []
        total_length = 0
        for i, filename in enumerate(file_names_to_concatenate):
            with h5py.File(file_names_to_concatenate[i], 'r') as activeData:
                vsource = h5py.VirtualSource(activeData[entry_key])
                total_length += vsource.shape[0]
                sources.append(vsource)

        layout = h5py.VirtualLayout(shape=(total_length,),
                                    dtype=np.float)

        offset = 0
        for vsource in sources:
            length = vsource.shape[0]
            layout[offset: offset + length] = vsource
            offset += length

        with h5py.File("VDS_con.h5", 'w', libver='latest') as f:
            f.create_virtual_dataset(entry_key, layout, fillvalue=0)


# Example
if __name__ == '__main__':
    with HDF5DatasetExtendable("File.h5") as ds:
        for i in range(10):
            # append N 100x100 3-channel images to the dataset with random labels of 0 or 1
            N = 10
            X = np.random.random(size=(N, 3, 100, 100))
            y = np.random.randint(0,1,N)
            ds.append(X, y)

    with DataGeneratorHDF5("File.h5") as ds:
        print(ds[0])  # ds acts as a numpy array
