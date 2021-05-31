"""The utils module contains utility funtions for Phydrus.

"""

import logging
from logging import handlers

logger = logging.getLogger(__name__)

import numpy as np
from io import StringIO
import pandas as pd

from numpy import sqrt, log, cos, pi, sin, exp, maximum, clip

# from .utils import extraterrestrial_r, daylight_hours, solar_declination, \
#     day_of_year, relative_distance, sunset_angle, extraterrestrial_r_hour

from numpy import tan, cos, pi, sin, arccos
from pandas import to_numeric

def show_versions():
    """
    Method to print the version of dependencies.

    Examples
    --------
    >>> import phydrus as ps
    >>> ps.show_versions()

    Python version: 3.8.2 (default, Mar 25 2020, 11:22:43)
    [Clang 4.0.1 (tags/RELEASE_401/final)]
    Numpy version: 1.19.2
    Pandas version: 1.2.1
    Phydrus version: 0.1.0
    Matplotlib version: 3.3.2

    """
    from phydrus import __version__ as ps_version
    from pandas import __version__ as pd_version
    from numpy import __version__ as np_version
    from matplotlib import __version__ as mpl_version
    from sys import version as os_version

    msg = (
        f"Python version: {os_version}\n"
        f"Numpy version: {np_version}\n"
        f"Pandas version: {pd_version}\n"
        f"Phydrus version: {ps_version}\n"
        f"Matplotlib version: {mpl_version}"
    )

    return print(msg)


def _initialize_logger(logger=None, level=logging.INFO):
    """
    Internal method to create a logger instance to log program output.

    Parameters
    -------
    logger : logging.Logger
        A Logger-instance. Use ps.logger to initialise the Logging instance
        that handles all logging throughout Phydrus,  including all sub modules
        and packages.

    """
    if logger is None:
        logger = logging.getLogger('phydrus')

    logger.setLevel(level)
    remove_file_handlers(logger)
    set_console_handler(logger)


def set_console_handler(logger=None, level=logging.INFO,
                        fmt="%(levelname)s: %(message)s"):
    """
    Method to add a console handler to the logger of Phydrus.

    Parameters
    -------
    logger : logging.Logger
        A Logger-instance. Use ps.logger to initialise the Logging instance
        that handles all logging throughout Phydrus, including all sub modules
        and packages.

    """
    if logger is None:
        logger = logging.getLogger('phydrus')
    remove_console_handler(logger)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    formatter = logging.Formatter(fmt=fmt)
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def set_log_level(level):
    """
    Set the log-level for which to log Phydrus messages.

    Parameters
    ----------
    level: str
        String with the level to log messages to the screen for. Options
        are: "INFO", "WARNING", and "ERROR".

    Examples
    --------
    >>> import phydrus as ps
    >>> ps.set_log_level("ERROR")

    """
    set_console_handler(level=level)


def remove_console_handler(logger=None):
    """
    Method to remove the console handler to the logger of Phydrus.

    Parameters
    ----------
    logger : logging.Logger
        A Logger-instance. Use ps.logger to initialise the Logging instance
        that handles all logging throughout Phydrus, including all sub modules
        and packages.

    """
    if logger is None:
        logger = logging.getLogger('phydrus')

    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            logger.removeHandler(handler)


def add_file_handlers(logger=None, filenames=('info.log', 'errors.log'),
                      levels=(logging.INFO, logging.ERROR), maxBytes=10485760,
                      backupCount=20, encoding='utf8', datefmt='%d %H:%M',
                      fmt='%(asctime)s-%(name)s-%(levelname)s-%(message)s'):
    """
    Method to add file handlers in the logger of Phydrus.

    Parameters
    -------
    logger : logging.Logger
        A Logger-instance. Use ps.logger to initialise the Logging instance
        that handles all logging throughout Phydrus, including all sub modules
        and packages.

    """
    if logger is None:
        logger = logging.getLogger('phydrus')
    # create formatter
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    # create file handlers, set the level & formatter, and add it to the logger
    for filename, level in zip(filenames, levels):
        fh = handlers.RotatingFileHandler(filename, maxBytes=maxBytes,
                                          backupCount=backupCount,
                                          encoding=encoding)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)


def remove_file_handlers(logger=None):
    """
    Method to remove any file handlers in the logger of Phydrus.

    Parameters
    -------
    logger : logging.Logger
        A Logger-instance. Use ps.logger to initialise the Logging instance
        that handles all logging throughout Phydrus, including all sub modules
        and packages.
    """
    if logger is None:
        logger = logging.getLogger('phydrus')
    for handler in logger.handlers:
        if isinstance(handler, handlers.RotatingFileHandler):
            logger.removeHandler(handler)
            
            
def z_loop(z, r1 = 10, r2 = 20):
    if z > -r1:
        return 1
    elif z < -(r1 + r2):
        return 0
    else:
        return(z+(r1+r2))/r2
    
def z_loop_vG(z, top=0, bot=-100, Lr = 20):
    bot = np.asarray(bot) - top
    bot = np.asarray(bot)
    z = np.asarray(z) - top
    top  = top - top
    if z > (top - 0.2 * Lr):
        return (5 / 3) / Lr
    elif z < top - Lr:
        return 0
    else:
        return (25 / 12) / Lr * (1 - (top - z / Lr))

def ihead(depth, elements, GWT):
    z = np.linspace(0, depth, elements+1, dtype=None)
    H = np.full(elements+1, GWT)
    hw = H - z #pressure head
    return hw

def read_nodinf(path, ml, proflength=False):
    num_lines = sum(1 for line in open(f'{path}/NOD_INF.OUT'))
    with open(f'{path}/NOD_INF.OUT') as fo:
        f = fo.readlines() 
    sidx = 9
    if proflength==True:
        proflen = len(pd.read_csv(f'{path}PROFILE.OUT', skiprows=6, skipfooter=0, delim_whitespace=True))
        rows = proflen + 2
    else:
        rows = len(ml.profile) + 3 # elements + 4
    new = sidx + rows
    s = StringIO('\n'.join(f[sidx:new]))
    d = {}
    d[0] = pd.read_csv(s, skiprows=[1,2], delim_whitespace=True).astype(float)
    if proflength==True:
        timesteps = int(np.floor(num_lines/(proflen+sidx))-1)
    else:
        timesteps = int(np.floor(num_lines/(len(ml.profile)+sidx))-1)
    for i in range(timesteps):
        idx = new + 6
        new = idx + rows
        s = StringIO('\n'.join(f[idx:new]))
        d[i+1] = pd.read_csv(s, delim_whitespace=True).drop([0]).astype(float)
    return d

def get_gwt(nodinf):
    """
    Extract the location of the groundwater table where the pressure head = 0 (bottom up)
    
    INPUT:
    nodinf - Dictionary of the NOD_INF.OUT file with the timesteps as keys. 
             The data in the dictionary contains a DataFrame for each timestep 
             with the column 'Depth' and 'Head'.
    OUTPUT:
    gwl - Numpy array with the location of the groundwater table.
        """
    gwl = []
    for i in list(nodinf.keys()):
        j = 0
        while np.flip(nodinf[i]['Head'].values)[j] > 0 and j < len(nodinf[i])-1:
            j += 1
        idx = len(nodinf[0]['Head']) - j
        if idx == 0:
            gwl_new = nodinf[i]['Head'][idx]
        elif idx == len(nodinf[i]['Head']):
            gwl_new = nodinf[i]['Depth'].iloc[-1]
        else:
            gwl_new = (0-nodinf[i]['Head'].iloc[idx-1])*(nodinf[i]['Depth'].iloc[idx]-nodinf[i]['Depth'].iloc[idx-1])/(nodinf[i]['Head'].iloc[idx]-nodinf[i]['Head'].iloc[idx-1]) + nodinf[i]['Depth'].iloc[idx-1]
        gwl.append(gwl_new)
    return np.array(gwl)

def create_nonlinear_profile(top=0, bot=-1, gwt_i=-100, dx_min=0.5, dx_max=1, r1=10, r2=20, lay=1, ah=1.0, ak=1.0, ath=1.0, temp=20.0, conc=None, sconc=None):
    if not isinstance(bot, list):
        bot = [bot]
    depth = bot[-1]
    rz = r1 + r2 #total length of rootzone
    
    dx_1 = np.arange(top, top-rz, -dx_min) # rootzone discretized with dx_min
    dx_2 = dx_1[-1] - np.cumsum(np.linspace(dx_min, dx_max, 40)) # increase step from dx_min to dx_max
    if depth >= dx_2[-1]:
        print(f'Depth should be lower than {dx_2[-1]}')
    dx_3 = np.flip(np.arange(depth, dx_2[-1], dx_max)) # extend total length to with dx_max to total depth
    grid = np.concatenate((dx_1, dx_2, dx_3)) # create total grid
    if np.max(np.diff(grid[1:])/np.diff(grid[1:])) > 1.5:
        print(f'The ratio of the sizes of two neighboring elements is not recommended to exceed about 1.5. Currently {np.max(np.diff(grid[1:])/np.diff(grid[1:]))}')
    if len(grid) > 900: #should not exceed 1000
        print('GRID TOO LONG, try a other values for dx_min and dx_max')
    
    data = pd.DataFrame(columns=["x", "h", "Mat", "Lay", "Beta", "Axz", "Bxz", "Dxz", "Temp", "Conc", "SConc"])
    data["x"] = grid
    data['h'] = -(grid - gwt_i)
    if len(bot) == 1:
        data['Mat'] = np.full(len(grid), 1)
    else:
        data['Mat'] = np.full(len(grid), 1)
        layidx = []
        for i in range(len(bot)):
            layidx = np.append(layidx, np.argmin(np.abs(data['x'] - bot[i])))
        for j in range(len(layidx)):
            data.loc[int(layidx[j])+1:, 'Mat'] = j + 2
    data.index = data.index + 1
    data["Lay"] = np.full(len(grid), lay)
    data["Axz"] = np.full(len(grid), ah)
    data["Bxz"] = np.full(len(grid), ak)
    data["Dxz"] = np.full(len(grid), ath)
    data["Temp"] = np.full(len(grid), temp)
    data["Beta"] = data.apply(lambda row: z_loop_vG(row["x"], top=top, bot=depth, Lr = rz), axis=1) #Hoffman and van Genuchten, 1983
#     data["Beta"] = data.apply(lambda row: z_loop_vG(row["x"], r1 = r1 r2 = r2), axis=1)
    data = data.replace(np.nan, "")
    print(f'Profile length: {len(data)}')
    return data

def partitioning_grass(P, ET, a=0.45, ch=5, k = 0.463, return_LAI=False):
    
    """
    Partitioning according to equation 2.75 in the Manual v4.0 and 
    Sutanto, Wenninger, Coenders and Uhlenbrook [2021]
    
    INPUT:
    P - precipitation [cm]
    ET - potential evapotranspiration Penman Monteith [cm]
    a - constant [cm]
    ch - cropheight (5-15 for clipped grass) [cm]
    k - radiation extinction by canopy (rExtinct) (0.463) [-]
    
    VARIABLES:
    LAI - Leaf Area Index (0.24 * ch) [cm/cm]
    SCF - Soil Cover Fraction (b) [-]
   
    OUTPUT:
    Pnet - Net Precipitation (P - I) [cm]
    I - Interception [cm]
    Et,p - Potential Transpiration [cm]
    Es,p - Potential Soil Evaporation [cm]
    """
    
    LAI = 0.24 * ch
    SCF = 1 - np.exp(-k * LAI)
    I = a * LAI * (1 - 1 / (1 + SCF * P / (a * LAI)))
    Pnet = np.maximum(P - I, 0)
    Etp = ET * SCF
    Esp = ET * (1 - SCF)
#     print(SCF)
    if return_LAI==True:
        return Pnet, I, Etp, Esp, LAI
    else:
        return Pnet, I, Etp, Esp
    
def get_recharge(nodinf, recharge_depth):
    """
    INPUT:
    nodinf - Dictionary of the NOD_INF.OUT file with the timesteps as keys. 
             The data in the dictionary contains a DataFrame for each timestep 
             with the column 'Depth' and 'Flux'.
    recharge_depth - Depth at which the recharge is extracted.

    OUTPUT:
    recharge - Numpy array with the recharge at a defined depth.
    """
    # recharge depth has to be negative
    idx = np.argmin(abs(nodinf[0]['Depth'].values-recharge_depth))
    recharge = []
    for i in list(nodinf.keys()):
        recharge.append(nodinf[i]['Flux'].iloc[idx])
    return np.array(recharge)