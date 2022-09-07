import numpy
from onnxruntime.quantization import CalibrationDataReader
from pathlib import Path


class Gpt2InputReader(CalibrationDataReader):
    def __init__(self, data_folder: str):
        self.batch_id = 0
        self.input_folder = Path(data_folder)

        if not self.input_folder.is_dir():
            raise RuntimeError(
                f"Can't find input data directory: {str(self.input_folder)}"
            )
        data_file = self.input_folder / f"batch_{self.batch_id}.npz"
        if not data_file.exists():
            raise RuntimeError(f"No data files found under '{self.input_folder}'")

    def get_next(self):
        self.input_dict = None
        data_file = self.input_folder / f"batch_{self.batch_id}.npz"
        if not data_file.exists():
            return None
        self.batch_id += 1

        self.input_dict = {}
        npy_file = numpy.load(data_file)
        for name in npy_file.files:
            self.input_dict[name] = npy_file[name]

        return self.input_dict

    def rewind(self):
        self.batch_id = 0
