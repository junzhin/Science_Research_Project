from ffcv.writer import DatasetWriter
from ffcv.fields import NDArrayField, FloatField

writer = DatasetWriter(write_path, {
    'covariate': NDArrayField(shape=(d,), dtype=np.dtype('float32')),
    'label': FloatField(),

}, num_workers=16)