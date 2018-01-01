import os
import numpy as np
import pandas as pd

from astropy.table import Table
import astropy.units as u
import astropy.constants as c
import astropy.coordinates as coords

__all__ = [
    "alphaM",
    "RockDerivative"
]

_datadir = os.path.dirname(os.path.dirname(__file__)[:-1])+'/data'

# mass in convective zone in earthmasses
alphaM = (0.02*u.solMass).to(u.earthMass).value


class RockDerivative(object):

    def __init__(self,):
        self.df = self._load_table()

    def show_missing_abundances(self):
        """Returns a table of elements missing in either solar or earth composition"""
        return self.df.loc[self.df.isnull()[['photosphere','bulk']].values.any(axis=1)]

    def _load_table(self):
        """Returns cleaned table of abundances"""
        solar = pd.read_csv(_datadir+"/asplund2009.csv", skipinitialspace=True, comment='#')
        atomic = pd.read_csv(_datadir+"/atomicmass.csv", skipinitialspace=True, comment='#')
        earth = pd.read_csv(_datadir+"/mcdonough2003.csv", skipinitialspace=True, comment='#')

        tb = pd.merge(solar[['Z','element','photosphere']], atomic[['element','Name','weight']], how='left', on='element')
        tb = pd.merge(tb, earth, how='outer', on='element')
        tb = tb[['Z', 'element' , 'Name', 'weight', 'photosphere', 'bulk']]
        #NOTE: missing elements filled in arbitrarily
        tb['f_photo'] = tb.weight*10**tb.photosphere.fillna(0)/(tb.weight*10**tb.photosphere.fillna(0)).sum()
        tb['f_rock'] = tb.bulk.fillna(0)/1e6
        return tb

    def XH(self, m):
        return np.log10(1+ self.df.f_rock.values/self.df.f_photo.values * m/alphaM)

    def dXHdm_exact(self, m):
        return self.df.f_rock.values / (self.df.f_photo.values + self.df.f_rock.values * m / alphaM) / alphaM / np.log(10)

    def dXHdm_avg(self, m1, m2):
        return (self.XH(m2)-self.XH(m1))/(m2-m1)

