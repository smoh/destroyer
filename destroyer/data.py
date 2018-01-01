import os
from glob import glob
import numpy as np
import fitsio

_datadir = os.path.dirname(os.path.dirname(__file__)[:-1])+'/data'

__all__ = [
    "load_lookup",
    "load_lamost",
    "GradientSpectra",
    "Spectra"
]

def load_lookup():
    """
    Get the array of spectra filenames.
    This is row-by-row match to LAMOST catalog.
    """
    return np.load(_datadir+"/filename.npz")['filename']


def load_lamost():
    """
    Returns the LAMOST catalog in ndarray.
    """
    return fitsio.read(_datadir+"/lamost_DR_stellar.fits")


class GradientSpectra(object):
    """
    class containing Ting gradient spectra
    """

    def __init__(self):
        d = np.load(_datadir+"/GDh_kurucz_spectra.npz")
        self.wave = d['wavelength'][:-32]
        self.gradient = d['gradient_spectra'][:,:-32]/0.2    # [X/H] / dex


class Spectra(object):
    def __init__(self):
        filenames = []
        spec, ivar = [], []
        for fn in glob(_datadir+"/lamost/*.fits.gz"):
            filenames.append(os.path.basename(fn).replace("Ho_normalized_",""))
            spec.append(fitsio.read(fn)[0])
            ivar.append(fitsio.read(fn)[1])
        self.filenames = np.array(filenames)
        self.spec = np.array(spec)
        self.ivar = np.array(ivar)
