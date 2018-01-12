import os
from glob import glob
import numpy as np
import pandas as pd
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
    return fitsio.read(_datadir+"/lamost_DR4_stellar.fits")


class GradientSpectra(object):
    """
    class containing Ting gradient spectra
    """

    def __init__(self):
        d = np.load(_datadir+"/GDh_kurucz_spectra.npz")
        self.wave = d['wavelength'][:-32]
        self.gradient = d['gradient_spectra'][:,:-32]/0.2    # [X/H] / dex


class Spectra(object):
    def __init__(self, data):
        """
        Load Ting's normalized LAMOST spectra in arrays

        Attributes
        ----------
        wave : 1d-array, (Nwave, )
            wavelength array
        flux : 2d-array, (Nstar, Nwave)
            flux array
        ivar : 2d-array, (Nstar, Nwave)
            inverse variance array,
        mask : 2d-array, (Nstar, Nwave)
            mask array, `True` for masked pixels
        idx : 1d-array, (Nstar, )
            index array into LAMOST catalog
        """
        self.wave = data['wave']
        self.flux = data['flux']
        self.ivar = data['ivar']
        self.filenames = data['filenames']
        self.idx = data['idx']
        self._make_masks()
        self._exclude_constant_spectra()

    @classmethod
    def from_directory(cls, datadir):
        """
        Read in all spectra fits files in a directory
        """
        d = np.load(_datadir+"/GDh_kurucz_spectra.npz")
        wave = d['wavelength'][:-32]
        filenames, flux, ivar = [], [], []
        for fn in glob(datadir+"/*.fits.gz"):
            filenames.append(os.path.basename(fn).replace("Ho_normalized_",""))
            flux.append(fitsio.read(fn)[0])
            ivar.append(fitsio.read(fn)[1])
        filenames = np.array(filenames)
        flux = np.array(flux)
        ivar = np.array(ivar)

        lookup = load_lookup()
        lookup = pd.DataFrame({"filename":lookup}).reset_index()
        merged = pd.merge(pd.DataFrame({'filename':filenames}),
                          lookup)
        idx = merged['index'].values
        spectra = cls(dict(wave=wave, flux=flux, ivar=ivar, filenames=filenames,
                           idx=idx))
        return spectra


    def to_npz(self, filename):
        """
        Save spectra to npz file
        """
        with open(filename, 'wb') as f:
            np.savez(f, wave=self.wave, flux=self.flux, ivar=self.ivar,
                     filenames=self.filenames, idx=self.idx)


    @classmethod
    def from_npz(cls, filename):
        """
        Initialize Spectra from npz file.
        """
        return cls(np.load(filename))


    def _make_masks(self):
        """
        Make mask array for LAMOST spectra
        """
        bad_flux = ~np.isfinite(self.flux)
        bad_ivar = (~np.isfinite(self.ivar) | (self.ivar <= 0))

        spread = 3 # due to redshift
        skylines = np.array([4046, 4358, 5460, 5577, 6300, 6363, 6863])
        bad_pix_skyline = np.zeros(self.wave.size, dtype=bool)
        for skyline in skylines:
            badmin = skyline-spread
            badmax = skyline+spread
            bad_pix_temp = np.logical_and(self.wave > badmin, self.wave < badmax)
            bad_pix_skyline[bad_pix_temp] = True
        # 34 pixels

        self.mask = bad_flux | bad_ivar | bad_pix_skyline

    def _exclude_constant_spectra(self):

        import itertools
        # count consecutive values: ex) [1, 1, 0, 2.3, 5.3] -> [2, 1, 1, 1]
        counts = []
        for flux in self.flux:
            counts.append(
                np.max([len(list(v)) for _, v in itertools.groupby(flux)]))
        counts = np.array(counts)
        # threshold counts
        bad_idx = np.where(counts > 50)[0]
        print("Trashing {:d} spectra: weird constant flux spectra".format(
            bad_idx.size))
        boolidx = np.ones(self.flux.shape[0], dtype=np.bool)
        boolidx[bad_idx] = False
        self.flux = self.flux[boolidx]
        self.ivar = self.ivar[boolidx]
        self.mask = self.mask[boolidx]
        self.filenames = self.filenames[boolidx]
        self.idx = self.idx[boolidx]
