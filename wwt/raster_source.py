import numpy as np

import rastervision as rv

from rastervision.core import Box
from rastervision.data import RasterSource
from rastervision.data import IdentityCRSTransformer

class NumpyRasterSource(RasterSource):
    def __init__(self, arr, extent=None, crs_transformer=None):
        self.arr = arr
        if extent is None:
            self.extent = Box(0, 0, arr.shape[0], arr.shape[1])
        else:
            self.extent = extent
        if crs_transformer is None:
            self.crs_transformer = IdentityCRSTransformer()
        else:
            self.crs_transformer = crs_transformer

    def get_extent(self):
        """Return the extent of the RasterSource.

        Returns:
            Box in pixel coordinates with extent
        """
        return self.extent

    def get_dtype(self):
        """Return the numpy.dtype of this scene"""
        return np.uint8

    def get_crs_transformer(self):
        """Return the associated CRSTransformer."""
        return self.crs_transformer

    def _get_chip(self, window):
        """Return the chip located in the window.

        Args:
            window: Box

        Returns:
            [height, width, channels] numpy array
        """
        return self.arr[window.ymin:window.ymax, window.xmin:window.xmax, :]
