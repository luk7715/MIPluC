#%%
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
import os

#%%
cwd = os.getcwd()


def normalize(data, lower=0, upper=255):
    '''This is used to normalize the fits file to jepg format'''
    return ((data - data.min()) / (data.max() - data.min())) * (upper - lower) + lower


def crop(filename, center, shape):
    #fname = get_pkg_data_filename(filename)
    fhdul = fits.open(filename)[0]
    image = fhdul.data

    position = center
    shape = shape
    cutout = Cutout2D(image, position, shape)

    data = cutout.data
    normalized_fit = normalize(data)
    hdul = fits.PrimaryHDU(normalized_fit)
    newp = Image.fromarray(normalized_fit)
    newp = newp.convert("L")
    newp = newp.transpose(Image.FLIP_LEFT_RIGHT)
    
    coordx = str(center[0])
    coordy = str(center[1])

    f'{cwd}/pluto_{coordx}_{coordy}.fits'
    hdul.writeto(f'{cwd}/FITS/pluto_{coordx}_{coordy}.fits',overwrite=True)
    newp.save(f'{cwd}/IMG/pluto_{coordx}_{coordy}.jpg')

    return


# %%
f = './mp1_0299178832_0x530_sci_14_MPAN.fit'
c = (3856, 4556)
s = (512,512)
crop(f,c,s)

# %%
