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
    gwl = [0]
    for i in range(len(nodinf)):
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
    return np.array(gwl[2:])

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
    # recharge depth has to be negative
    idx = np.argmin(abs(nodinf[0]['Depth'].values-recharge_depth))
    rech = [0]
    for i in range(len(nodinf)):
        rech_new = nodinf[i]['Flux'].loc[idx]
        rech.append(rech_new)
    return np.array(rech[2:])

def penman(wind, elevation, latitude, solar=None, net=None, sflux=0, tmax=None,
           tmin=None, rhmax=None, rhmin=None, rh=None, n=None, nn=None,
           rso=None, a=2.6, b=0.536):
    """
    Returns evapotranspiration calculated with the Penman (1948) method.

    Based on equation 6 in Allen et al (1998).

    Parameters
    ----------
    wind: pandas.Series
        mean day wind speed [m/s]
    elevation: float/int
        the site elevation [m]
    latitude: float/int
        the site latitude [rad]
    solar: pandas.Series, optional
        incoming measured solar radiation [MJ m-2 d-1]
    net: pandas.Series, optional
        net radiation [MJ m-2 d-1]
    sflux: pandas.Series/int, optional
        soil heat flux [MJ m-2 d-1]
    tmax: pandas.Series, optional
        maximum day temperature [°C]
    tmin: pandas.Series, optional
        minimum day temperature [°C]
    rhmax: pandas.Series, optional
        maximum daily relative humidity [%]
    rhmin: pandas.Series, optional
        mainimum daily relative humidity [%]
    rh: pandas.Series, optional
        mean daily relative humidity [%]
    n: pandas.Series/float, optional
        actual duration of sunshine [hour]
    nn: pandas.Series/float, optional
        maximum possible duration of sunshine or daylight hours [hour]
    rso: pandas.Series/float, optional
        clear-sky solar radiation [MJ m-2 day-1]
    a: float/int, optional
        wind coefficient [-]
    b: float/int, optional
        wind coefficient [-]

    Returns
    -------
        pandas.Series containing the calculated evapotranspiration

    Examples
    --------
    >>> penman_et = penman(wind, elevation, latitude, solar=solar, tmax=tmax,
    >>>                    tmin=tmin, rh=rh)

    """
    ta = (tmax + tmin) / 2
    pressure = press_calc(elevation, ta)
    gamma = psy_calc(pressure)
    dlt = vpc_calc(ta)
    lambd = lambda_calc(ta)

    ea = ea_calc(tmax, tmin, rhmax, rhmin, rh)
    es = es_calc(tmax, tmin)
    if net is None:
        rns = shortwave_r(solar=solar, n=n, nn=nn)  # in #  [MJ/m2/d]
        rnl = longwave_r(solar=solar, tmax=tmax, tmin=tmin, rhmax=rhmax,
                         rhmin=rhmin, rh=rh, rso=rso, elevation=elevation,
                         lat=latitude, ea=ea)  # in #  [MJ/m2/d]
        net = rns - rnl

    w = a * (1 + b * wind)

    den = lambd * (dlt + gamma)
    num1 = (dlt * (net - sflux) / den)
    num2 = (gamma * (es - ea) * w / den)
    pet = (num1 + num2)
    return pet


def pm_fao56(wind, elevation, latitude, solar=None, net=None, sflux=0,
             tmax=None, tmin=None, rhmax=None, rhmin=None, rh=None, n=None,
             nn=None, rso=None):
    """
    Returns reference evapotranspiration using the FAO-56 Penman-Monteith
    equation (Monteith, 1965; Allen et al, 1998).

    Based on equation 6 in Allen et al (1998).

    Parameters
    ----------
    wind: Series
        mean day wind speed [m/s]
    elevation: float/int
        the site elevation [m]
    latitude: float/int
        the site latitude [rad]
    solar: pandas.Series, optional
        incoming measured solar radiation [MJ m-2 d-1]
    net: pandas.Series, optional
        net radiation [MJ m-2 d-1]
    sflux: Series/float/int, optional
        soil heat flux [MJ m-2 d-1]
    tmax: pandas.Series, optional
        maximum day temperature [°C]
    tmin: pandas.Series, optional
        minimum day temperature [°C]
    rhmax: pandas.Series, optional
        maximum daily relative humidity [%]
    rhmin: pandas.Series, optional
        mainimum daily relative humidity [%]
    rh: pandas.Series, optional
        mean daily relative humidity [%]
    n: Series/float, optional
        actual duration of sunshine [hour]
    nn: Series/float, optional
        maximum possible duration of sunshine or daylight hours [hour]
    rso: Series/float, optional
        clear-sky solar radiation [MJ m-2 day-1]

    Returns
    -------
        pandas.Series containing the calculated evapotranspiration

    Examples
    --------
    >>> pm_fao56_et = pm_fao56(wind, elevation, latitude, solar=solar,
    >>>                        tmax=tmax, tmin=tmin, rh=rh)

    """
    ta = (tmax + tmin) / 2
    pressure = press_calc(elevation, ta)
    gamma = psy_calc(pressure)
    dlt = vpc_calc(ta)

    gamma1 = (gamma * (1 + 0.34 * wind))

    ea = ea_calc(tmax, tmin, rhmax=rhmax, rhmin=rhmin, rh=rh)
    es = es_calc(tmax, tmin)
    if net is None:
        rns = shortwave_r(solar=solar, n=n, nn=nn)  # in [MJ/m2/d]
        rnl = longwave_r(solar=solar, tmax=tmax, tmin=tmin, rhmax=rhmax,
                         rhmin=rhmin, rh=rh, rso=rso, elevation=elevation,
                         lat=latitude, ea=ea)  # in [MJ/m2/d]
        net = rns - rnl

    den = (dlt + gamma1)
    num1 = (0.408 * dlt * (net - sflux))
    num2 = (gamma * (es - ea) * 900 * wind / (ta + 273))
    return (num1 + num2) / den


def pm_asce(wind, elevation, latitude, solar=None, net=None, sflux=0,
            tmax=None, tmin=None, rhmax=None, rhmin=None, rh=None, n=None,
            nn=None, rso=None, lai=None, croph=None, rs=1, ra=1, rl=100):
    """
    Returns evapotranspiration calculated with the ASCE Penman-Monteith
    (Monteith, 1965; ASCE, 2005) method.

    Parameters
    ----------
    wind: pandas.Series
        mean day wind speed [m/s]
    elevation: float/int
        the site elevation [m]
    latitude: float/int
        the site latitude [rad]
    solar: pandas.Series, optional
        incoming measured solar radiation [MJ m-2 d-1]
    net: pandas.Series, optional
        net radiation [MJ m-2 d-1]
    sflux: Series/float/int, optional
        soil heat flux [MJ m-2 d-1]
    tmax: pandas.Series, optional
        maximum day temperature [°C]
    tmin: pandas.Series, optional
        minimum day temperature [°C]
    rhmax: pandas.Series, optional
        maximum daily relative humidity [%]
    rhmin: pandas.Series, optional
        minimum daily relative humidity [%]
    rh: pandas.Series, optional
        mean daily relative humidity [%]
    n: pandas.Series/float, optional
        actual duration of sunshine [hour]
    nn: pandas.Series/float, optional
        maximum possible duration of sunshine or daylight hours [hour]
    rso: pandas.Series/float, optional
        clear-sky solar radiation [MJ m-2 day-1]
    lai: pandas.Series/float, optional
        measured leaf area index [-]
    croph: float/int/pandas.series, optional
        crop height [m]
    rs: int, optional
        1 => rs = 70
        2 => rs = rl/LAI; rl = 200
    ra: int, optional
        1 => ra = 208/wind
        2 => ra is calculated based on equation 36 in FAO (1990), ANNEX V.

    Returns
    -------
        pandas.Series containing the calculated evapotranspiration

    Examples
    --------
    >>> pmasce = pm_asce(wind, elevation, latitude, rs=solar, tmax=tmax,
    >>>                  tmin=tmin, rh=rh)

    """
    ta = (tmax + tmin) / 2
    lambd = lambda_calc(ta)
    pressure = press_calc(elevation, ta)
    gamma = psy_calc(pressure)
    dlt = vpc_calc(ta)
    cp = 1.013 * 10 ** (-3)
    r_a = aero_r(wind, method=ra, croph=croph)
    r_s = surface_r(method=rs, lai=lai, rl=rl)
    gamma1 = gamma * (1 + r_s / r_a)

    ea = ea_calc(tmax=tmax, tmin=tmin, rhmax=rhmax, rhmin=rhmin, rh=rh)
    es = es_calc(tmax, tmin)
    rho_a = calc_rhoa(pressure, ta, ea)
    if net is None:
        rns = shortwave_r(solar=solar, n=n, nn=nn)
        rnl = longwave_r(solar=solar, tmax=tmax, tmin=tmin, rhmax=rhmax,
                         rhmin=rhmin, rh=rh, rso=rso, elevation=elevation,
                         lat=latitude, ea=ea)
        net = rns - rnl
    kmin = 86400
    den = (lambd * (dlt + gamma1))
    num1 = (dlt * (net - sflux) / den)
    num2 = (rho_a * cp * kmin * (es - ea) / r_a / den)
    return num1 + num2


def pm_corrected(wind, elevation, latitude, solar=None, net=None, sflux=0, tmean=None,
                 tmax=None, tmin=None, rhmax=None, rhmin=None, rh=None, n=None,
                 nn=None, rso=None, lai=None, croph=None, r_s=None, rs=1, ra=1, a_s=1,
                 a_sh=1, rl=100, a=1.35, b=-0.35, co2=300, srs=0.0009, laieff=0, flai=1,
                 freq="D"):
    """
    Returns evapotranspiration calculated with the upscaled corrected
    Penman-Monteith equation from Schymanski (Schymanski, 2017).

    Parameters
    ----------
    wind: pandas.Series
        mean day wind speed [m/s]
    elevation: float/int
        the site elevation [m]
    latitude: float/int
        the site latitude [rad]
    solar: pandas.Series, optional
        incoming measured solar radiation [MJ m-2 d-1]
    net: pandas.Series, optional
        net radiation [MJ m-2 d-1]
    sflux: Series/float/int, optional
        soil heat flux [MJ m-2 d-1]
    tmax: pandas.Series, optional
        maximum day temperature [°C]
    tmin: pandas.Series, optional
        minimum day temperature [°C]
    rhmax: pandas.Series, optional
        maximum daily relative humidity [%]
    rhmin: pandas.Series, optional
        minimum daily relative humidity [%]
    rh: pandas.Series, optional
        mean daily relative humidity [%]
    n: pandas.Series/float, optional
        actual duration of sunshine [hour]
    nn: pandas.Series/float, optional
        maximum possible duration of sunshine or daylight hours [hour]
    rso: pandas.Series/float, optional
        clear-sky solar radiation [MJ m-2 day-1]
    lai: pandas.Series/float, optional
        measured leaf area index [-]
    croph: float/int/pandas.series, optional
        crop height [m]
    rs: int, optional
        1 => rs = 70
        2 => rs = rl/LAI; rl = 200
    ra: int, optional
        1 => ra = 208/wind
        2 => ra is calculated based on equation 36 in FAO (1990), ANNEX V.
    a_s: int, optional
        Fraction of one-sided leaf area covered by stomata (1 if stomata are 1
        on one side only, 2 if they are on both sides)
    a_sh: int, optional
        Fraction of projected area exchanging sensible heat with the air (2)

    Returns
    -------
        pandas.Series containing the calculated evapotranspiration

    """
    if "D" in freq:
        tmean = (tmax + tmin) / 2
        es = es_calc(tmax, tmin)
    else:
        es = e0_calc(tmean)
    lambd = lambda_calc(tmean)
    pressure = press_calc(elevation, tmean)
    gamma = psy_calc(pressure)
    dlt = vpc_calc(tmean)
    cp = 1.013 * 10 ** (-3)
    r_a = aero_r(wind, method=ra, croph=croph)
    ea = ea_calc(tmax=tmax, tmin=tmin, rhmax=rhmax, rhmin=rhmin, rh=rh)
    rho_a = calc_rhoa(pressure, tmean, ea)
    if net is None:
        rns = shortwave_r(solar=solar, n=n, nn=nn)
        rnl = longwave_r(solar=solar, tmax=tmax, tmin=tmin, rhmax=rhmax,
                         rhmin=rhmin, rh=rh, rso=rso, elevation=elevation,
                         lat=latitude, ea=ea, a=a, b=b, freq=freq)
        net = rns - rnl * a_sh
    kmin = 86400
    if "H" in freq:
        kmin = 3600
    if r_s is None:
        r_s = surface_r(method=rs, lai=lai, rl=rl, co2=co2, srs=srs, laieff=laieff, 
                        flai=flai)

    gamma1 = gamma * a_sh / a_s * (1 + r_s / r_a)
    den = (lambd * (dlt + gamma1))
    num1 = (dlt * (net - sflux) / den)
    num2 = (rho_a * cp * kmin * (es - ea) * a_sh / r_a / den)
    return num1 + num2


def pm_fao1990(wind, elevation, latitude, solar=None, tmax=None, tmin=None,
               rh=None, croph=None, ra=2, rs=60, n=None, nn=None):
    """
    Returns evapotranspiration calculated with the FAO Penman-Monteith
    (Monteith, 1965; FAO, 1990) method.

    Based on equation 30 (FAO, 1990).

    Parameters
    ----------
    wind: pandas.Series
        mean day wind speed [m/s]
    elevation: float/int
        the site elevation [m]
    latitude: float/int
        the site latitude [rad]
    solar: pandas.Series, optional
        incoming measured solar radiation [MJ m-2 d-1]
    tmax: pandas.Series, optional
        maximum day temperature [°C]
    tmin: pandas.Series, optional
        minimum day temperature [°C]
    rh: pandas.Series, optional
        mean daily relative humidity [%]
    croph: float/int/pandas.series, optional
        crop height [m]

    Returns
    -------
        pandas.Series containing the calculated evapotranspiration

    Examples
    --------
    >>> pm_fao1990_et = pm_fao1990(wind, elevation, latitude, solar=solar,
    >>>                            tmax=tmax, tmin=tmin, rh=rh, croph=0.6)

    """
    # aeroterm
    ta = (tmax + tmin) / 2.
    lambd = lambda_calc(ta)
    pressure = press_calc(elevation, ta)
    gamma = psy_calc(pressure=pressure, lambd=lambd)
    eamax = e0_calc(tmax)
    eamin = e0_calc(tmin)

    raa = aero_r(wind, method=ra, croph=croph)
    eadew = ed_calc(tmax, tmin, rh)  # OK
    aerodyn = raa * wind
    aerotcff = 0.622 * 3.486 * 86400. / aerodyn / 1.01
    lai = croph * 24
    rs = 200/lai
    gamma1 = gamma * (1 + rs / raa)
    dlt = vpc_calc(tmin=tmin, tmax=tmax, method=1)

    gm_dl = gamma / (dlt + gamma1)
    eamean = (eamax + eamin) / 2
    etaero = gm_dl * aerotcff / (ta + 273.) * wind * (eamean - eadew)


    dl_dl = dlt / (dlt + gamma)
    # rad term
    rso = rs_calc(solar.index, latitude)  # OK
    rns = shortwave_r(solar=solar, n=n, nn=nn)
    rnl = longwave_r(solar, tmax=tmax, tmin=tmin, rh=rh, rso=rso,
                     elevation=elevation, lat=latitude, ea=eadew)
    net = rns - rnl

    radterm = dl_dl * (net) / lambd
    pm = (etaero + radterm)
    return pm, radterm, etaero, rnl, rns


def priestley_taylor(wind, elevation, latitude, solar=None, net=None,
                     tmax=None, tmin=None, rhmax=None, rhmin=None, rh=None,
                     rso=None, n=None, nn=None, alpha=1.26):
    """
    Returns evapotranspiration calculated with the Penman-Monteith
    (FAO,1990) method.

    Based on equation 6 in Allen et al (1998).

    Parameters
    ----------
    wind: pandas.Series
        mean day wind speed [m/s]
    elevation: float/int
        the site elevation [m]
    latitude: float/int
        the site latitude [rad]
    solar: pandas.Series
        incoming measured solar radiation [MJ m-2 d-1]
    net: pandas.Series
        net radiation [MJ m-2 d-1]
    tmax: pandas.Series
        maximum day temperature [°C]
    tmin: pandas.Series
        minimum day temperature [°C]
    rhmax: pandas.Series
        maximum daily relative humidity [%]
    rhmin: pandas.Series
        mainimum daily relative humidity [%]
    rh: pandas.Series
        mean daily relative humidity [%]
    n: Series/float
        actual duration of sunshine [hour]
    nn: Series/float
        maximum possible duration of sunshine or daylight hours [hour]
    rso: Series/float
        clear-sky solar radiation [MJ m-2 day-1]
    alpha: Series/float
        calibration coefficient

    Returns
    -------
        pandas.Series containing the calculated evapotranspiration

    Examples
    --------
    >>> pm = priestley_taylor(wind, elevation, latitude, solar=solar,
    >>>                       tmax=tmax, tmin=tmin, rh=rh, croph=0.6)

    """
    ta = (tmax + tmin) / 2
    lambd = lambda_calc(ta)
    pressure = press_calc(elevation, ta)
    gamma = psy_calc(pressure=pressure, lambd=None)
    dlt = vpc_calc(temperature=ta, method=0)

    ea = ea_calc(tmax, tmin, rhmax=rhmax, rhmin=rhmin, rh=rh)
    if net is None:
        rns = shortwave_r(solar=solar, n=n, nn=nn)  # in [MJ/m2/d]
        rnl = longwave_r(solar=solar, tmax=tmax, tmin=tmin, rhmax=rhmax,
                         rhmin=rhmin, rh=rh, rso=rso, elevation=elevation,
                         lat=latitude, ea=ea)  # in [MJ/m2/d]
        net = rns - rnl

    return (alpha * dlt * net) / (lambd * (dlt + gamma))


def makkink(tmax, tmin, rs, elevation, f=1):
    """
    Returns evapotranspiration calculated with the Makkink (1957) method.

    Parameters
    ----------
    tmax: pandas.Series
        maximum day temperature [°C]
    tmin: pandas.Series
        minimum day temperature [°C]
    rs: pandas.Series
        incoming measured solar radiation [MJ m-2 d-1]
    elevation: float/int
        the site elevation [m]
    f: float/int, optional
        crop coefficient [-]

    Returns
    -------
        Series containing the calculated evapotranspiration

    Examples
    --------
    >>> mak = makkink(tmax, tmin, rs, elevation)

    """
    ta = (tmax + tmin) / 2
    pressure = press_calc(elevation, ta)
    gamma = psy_calc(pressure=pressure, lambd=None)
    dlt = vpc_calc(temperature=ta, method=0)

    return f / 2.45 * 0.61 * rs * dlt / (dlt + gamma) - 0.12


##% Utility functions (TODO: Make private?)


def longwave_r(solar, tmax=None, tmin=None, rhmax=None, rhmin=None,
               rh=None, rso=None, elevation=None, lat=None, ea=None,
               a=1.35, b=-0.35, freq="D"):
    """
    Net outgoing longwave radiation.

    Based on equation 39 in Allen et al (1998).
    Parameters
    ----------
    solar: Series
        incoming measured solar radiation [MJ m-2 d-1]
    elevation: float/int
        the site elevation [m]
    lat: float/int
        the site latitude [rad]
    tmax: Series
        maximum day temperature [°C]
    tmin: Series
        minimum day temperature [°C]
    rhmax: Series
        maximum daily relative humidity [%]
    rhmin: Series
        mainimum daily relative humidity [%]
    rh: Series
        mean daily relative humidity [%]
    rso: Series/float
        clear-sky solar radiation [MJ m-2 day-1]
    ea: Series
        actual vapour pressure.
    Returns
    -------
        pandas.Series containing the calculated net outgoing radiation
    """
    if ea is None:
        ea = ea_calc(tmin=tmin, tmax=tmax, rhmin=rhmin, rhmax=rhmax, rh=rh)
    if "H" in freq:
        steff = 2.042 * 10 ** (-10)  # MJm-2K-4h-1
        if rso is None:
            ra = extraterrestrial_r_hour(solar.index, lat)
            rso = rso_calc(ra, elevation)
        solar_rat = solar / rso
        solar_rat = clip(solar_rat, 0.3, 1)
        tmp1 = steff * (ta + 273.2) ** 4
    else:
        steff = 4.903 * 10 ** (-9)  # MJm-2K-4d-1
        if rso is None:
            ra = extraterrestrial_r(solar.index, lat)
            rso = rso_calc(ra, elevation)
        solar_rat = clip(solar / rso, 0.3, 1)
        tmp1 = steff * ((tmax + 273.2) ** 4 + (tmin + 273.2) ** 4) / 2

    tmp2 = 0.34 - 0.139 * sqrt(ea)  # OK
    tmp2 = clip(tmp2, 0.05, 1)
    tmp3 = a * solar_rat + b  # OK
    return tmp1 * tmp2 * tmp3


def vpc_calc(temperature=None, tmin=None, tmax=None, method=0):
    """
    Slope of saturation vapour pressure curve at air Temperature.

    Parameters
    ----------
    temperature: Series
        mean day temperature [degC].

    Returns
    -------
        Series of Saturation vapour pressure [kPa degC-1].

    Notes
    -----
    if method is 0:
        Based on equation 13. in Allen et al 1998. The slope of the vapour
        pressure curve is in the FAO-56 method calculated using mean air
        temperature
    if method is 1:
        From FAO (1990), ANNEX V, eq. 3

    """
    if method == 0:
        ea = e0_calc(temperature)
        return 4098 * ea / (temperature + 237.3) ** 2
    elif method == 1:
        eamax = e0_calc(tmax)
        eamin = e0_calc(tmin)
        return round((2049. * eamax / (tmax + 237.3) ** 2) +
                     (2049. * eamin / (tmin + 237.3) ** 2), 8)
    elif method == 2:
        return 2503 * exp((17.27 * temperature) / (temperature + 237.3)) / (
                temperature + 237.3) ** 2


def e0_calc(temperature):
    """
    saturation vapour pressure at the air temperature T.

    Based on equations 11 in ALLen et al (1998).
    Parameters
    ----------Saturation Vapour Pressure  (es) from air temperature
    temperature: pandas.Series
         temperature [degC]
    Returns
    -------
        pandas.Series of saturation vapour pressure at the air temperature
        T [kPa]

    """
    return 0.6108 * exp((17.27 * temperature) / (temperature + 237.3))


def es_calc(tmax, tmin):
    """
    saturation vapour pressure at the air temperature T.

    Based on equations 11 in Allen et al (1998).

    Parameters
    ----------Saturation Vapour Pressure  (es) from air temperature
    tmax: pandas.Series
        maximum day temperature [°C]
    tmin: pandas.Series
        minimum day temperature [°C]

    Returns
    -------
        pandas.Series of saturation vapour pressure at the air temperature
        T [kPa]

    """
    eamax = e0_calc(tmax)
    eamin = e0_calc(tmin)
    return (eamax + eamin) / 2


def ea_calc(tmax, tmin, rhmax=None, rhmin=None, rh=None):
    """Actual Vapour Pressure (ea) from air temperature.

    Based on equations 17, 18, 19, in ALLen et al (1998).
    Parameters
    ----------
    tmax: Series
        maximum day temperature [degC]
    tmin: Series
        minimum day temperature [degC]
    rhmax: Series
        maximum daily relative humidity [%]
    rhmin: Series
        mainimum daily relative humidity [%]
    rh: pandas.Series/int
        mean daily relative humidity [%]
    Returns
    -------
        Series of saturation vapour pressure at the air temperature
        T [kPa]
    """
    eamax = e0_calc(tmax)
    eamin = e0_calc(tmin)
    if rhmax is not None and rhmin is not None:  # eq. 17
        return (eamin * rhmax / 200) + (eamax * rhmin / 200)
    elif rhmax is not None and rhmin is None:  # eq.18
        return eamin * rhmax / 100
    elif rhmax is None and rhmin is not None:  # eq. 48
        return eamin
    elif rh is not None:  # eq. 19
        return rh / 200 * (eamax + eamin)
    else:
        print("error")


def rso_calc(ra, elevation):
    """
    Actual Vapour Pressure (ea) from air temperature.

    Based on equations 37 in ALLen et al (1998).
    Parameters
    ----------
    ra: Series
        extraterrestrial radiation [MJ m-2 day-1]
    Returns
    -------
        Series of clear-sky solar radiation [MJ m-2 day-1]

    """
    return (0.75 + (2 * 10 ** -5) * elevation) * ra


def psy_calc(pressure, lambd=None):
    """
    Psychrometric constant [kPa degC-1].

    Parameters
    ----------
    pressure: int/real
        atmospheric pressure [kPa].
    lambd: float,m optional
        Divide the pressure by this value.

    Returns
    -------
        pandas.series of Psychrometric constant [kPa degC-1].

    Notes
    -----
    if lambd is none:
        From FAO (1990), ANNEX V, eq. 4
    else:
        Based on equation 8 in Allen et al (1998).

    """
    if lambd is None:
        return 0.000665 * pressure
    else:
        return 0.0016286 * pressure / lambd


def press_calc(elevation, temperature):
    """
    Atmospheric pressure.

    Based on equation 7 in Allen et al (1998).
    Parameters
    ----------
    elevation: int/real
        elevation above sea level [m].
    Returns
    -------
        int/real of atmospheric pressure [kPa].

    """
    return 101.3 * (((273.16 + temperature) - 0.0065 * elevation) /
                    (273.16 + temperature)) ** (9.807 / (0.0065 * 287))


def shortwave_r(solar=None, meteoindex=None, lat=None, alpha=0.23, n=None,
                nn=None):
    """
    Net solar or shortwave radiation.

    Based on equation 38 in Allen et al (1998).

    Parameters
    ----------
    meteoindex: pandas.Series.index
    solar: Series
        incoming measured solar radiation [MJ m-2 d-1]
    lat: float/int
        the site latitude [rad]
    alpha: float/int
        albedo or canopy reflection coefficient, which is 0.23 for the
        hypothetical grass reference crop [-]
    n: float/int
        actual duration of sunshine [hour]
    nn: float/int
        daylight hours [-]

    Returns
    -------
        Series containing the calculated net outgoing radiation
    """
    if solar is not None:
        return (1 - alpha) * solar
    else:
        return (1 - alpha) * in_solar_r(meteoindex, lat, n=n, nn=nn)


def in_solar_r(meteoindex, lat, a_s=0.25, b_s=0.5, n=None, nn=None):
    """
    Incoming solar radiation.
    Based on eq. 35 from FAO56.
    """
    ra = extraterrestrial_r(meteoindex, lat)
    if n is None:
        n = daylight_hours(meteoindex, lat)
    return (a_s + b_s * n / nn) * ra


def lai_calc(method=1, croph=None):
    if method == 1:
        return 0.24 * croph


def surface_r(lai=None, method=1, laieff=0, rl=100, co2=300, srs=0.0009, flai=1):
    if method == 1:
        return 70
    elif method == 2:
        lai_eff = calc_laieff(lai=lai, method=laieff)
        return rl / lai_eff
    elif method == 3:
        lai_eff = calc_laieff(lai=lai, method=laieff)
        fco2 = (1 + srs * (co2 - 300))
        return rl / lai_eff * fco2
    elif method == 4:
        lai_eff = calc_laieff(lai=lai, method=laieff)
        flai1 = lai_eff/lai_eff.max() * flai
        fco2 = (1 + srs * (co2 - 300))
        return rl / lai_eff * fco2 * flai1

def calc_laieff(lai=None, method=0):
    if method == 0:
        return 0.5 * lai
    if method == 1:
        return lai/(0.3*lai+1.2)
    if method == 2:
        laie = lai.copy()
        laie[(lai>2)&(lai<4)] = 2
        laie[lai>4] = 0.5 * lai
        return laie
    if method == 3:
        laie = lai.copy()
        laie[lai>4] = 4
        return laie*0.5


def lambda_calc(temperature):
    """
    From ASCE (2001), eq. B.7
    """
    return 2.501 - 0.002361 * temperature


def calc_rhoa(pressure, ta, ea):
    tkv = (273.16 + ta) * (1 - 0.378 * ea / pressure) ** (-1)
    return 3.486 * pressure / tkv


def aero_r(wind, croph=None, zw=2, zh=2, method=1):
    """
    The aerodynamic resistance, applied for neutral stability conditions
     from ASCE (2001), eq. B.2

    Parameters
    ----------
    wind: Series
         wind speed at height z [m/s]
    zw: float
        height of wind measurements [m]
    zh: float
         height of humidity and or air temperature measurements [m]

    Returns
    -------
        pandas.Series containing the calculated aerodynamic resistance.

    """
    if method == 1:
        return 208 / wind
    elif method == 2:
        d = 0.667 * croph
        zom = 0.123 * croph
        zoh = 0.0123 * croph
        return (log((zw - d) / zom)) * \
               (log((zh - d) / zoh) /
                (0.41 ** 2) / wind)


def cloudiness_factor(rs, rso, ac=1.35, bc=-0.35):
    """
    Cloudiness factor f
    From FAO (1990), ANNEX V, eq. 57
    """
    return ac * rs / rso + bc


def rs_calc(meteoindex, lat, a_s=0.25, b_s=0.5):
    """
    Nncoming solar radiation rs
    From FAO (1990), ANNEX V, eq. 52
    """
    ra = extraterrestrial_r(meteoindex, lat)
    nn = 1
    return (a_s + b_s * nn) * ra


def ed_calc(tmax, tmin, rh):
    """
    Actual Vapour Pressure (ed).
    From FAO (1990), ANNEX V, eq. 11
    """
    eamax = e0_calc(tmax)
    eamin = e0_calc(tmin)
    return rh / (50. / eamin + 50. / eamax)


def calc_rns(solar=None, meteoindex=None, lat=None, alpha=0.23):
    """
    Net Shortwave Radiation Rns
    From FAO (1990), ANNEX V, eq. 51
    """
    if solar is not None:
        return (1 - alpha) * solar
    else:
        return (1 - alpha) * rs_calc(meteoindex, lat)


def calc_rnl(tmax, tmin, ea, cloudf, longa=0.34, longb=-0.139):
    """
    Net Longwave Radiation Rnl from FAO (1990), ANNEX V, eq. 56

    Parameters
    ----------
    tmax: Series
        maximum day temperature [°C]
    tmin: Series
        minimum day temperature [°C]

    Returns
    -------
        pandas.Series containing the calculated net outgoing radiation.

    """
    sigma = 0.00000000245 * ((tmax + 273.16) ** 4 + (tmin + 273.16) ** 4)
    emiss = longa + longb * round(sqrt(ea), 8)
    return sigma * cloudf * emiss


def day_of_year(meteoindex):
    """
    Return day of the year (1-365) based on pandas.series.index

    Parameters
    ----------
    meteoindex: pandas.Series.index

    Returns
    -------
        array of with ints specifying day of year.

    """
    return to_numeric(meteoindex.strftime('%j'))


def daylight_hours(meteoindex, lat):
    """
    Daylight hours
    Based on eq. 34 from FAO56.
    """
    j = day_of_year(meteoindex)
    sol_dec = solar_declination(j)
    sangle = sunset_angle(sol_dec, lat)
    return round(24 / pi * sangle, 1)


def sunset_angle(sol_dec, lat):
    """
    Sunset hour angle from latitude and solar declination [rad].

    Based on equations 25 in ALLen et al (1998).
    Parameters
    ----------
    sol_dec: pandas.Series
        solar declination [rad]
    lat: float/int
        the site latitude [rad]
    Returns
    -------
        pandas.Series of sunset hour angle [rad].

    """
    return arccos(-tan(sol_dec) * tan(lat))


def sunset_angle_hour(meteoindex, lz, lm, lat, sol_dec):
    """
    Sunset hour angle from latitude and solar declination [rad].

    Based on equations 25 in ALLen et al (1998).
    Parameters
    ----------
    sol_dec: pandas.Series
        solar declination [rad]
    lat: float/int
        the site latitude [rad]
    Returns
    -------
        pandas.Series of sunset hour angle [rad].

    """
    j=day_of_year(meteoindex)
    b = 2*pi*(j-81)/364
    sc = 0.1645*sin(2*b) - 0.1255*cos(b) - 0.025*sin(b)
    t = meteoindex.hour + 0.5
    sol_t = t+0.06667*(lz-lm)+sc-12
    omega = pi/12
    omega1 = omega-pi/24
    omega2 = omega+pi/24
    omegas = arccos(-tan(lat)*tan(sol_dec))
    omega1 = clip(omega1,-omegas, omegas)
    omega2 = clip(omega2,-omegas, omegas)
    omega1 = maximum(omega1,omega1, )
    omega1 = clip(omega1, -100000000, omega2)

    return omega2, omega1


def solar_declination(j):
    """
    Solar declination [rad] from day of year [rad].

    Based on equations 24 in ALLen et al (1998).
    Parameters
    ----------
    j: array.py
        day of the year (1-365)
    Returns
    -------
        array.py of solar declination [rad].

    """
    return 0.4093 * sin(2. * 3.141592654 / 365. * j - 1.39)


def relative_distance(j):
    """
    Inverse relative distance between earth and sun from day of the year.

    Based on equation 23 in Allen et al (1998).

    Parameters
    ----------
    j: array.py
        day of the year (1-365)
    Returns
    -------
        array.py specifyng day of year.
    """
    return 1 + 0.033 * cos((2. * 3.141592654 / 365.) * j)


def extraterrestrial_r(meteoindex, lat):
    """
    Returns Extraterrestrial Radiation (Ra).

    Based on equation 21 in Allen et al (1998).
    Parameters
    ----------
    meteoindex: Series.index
    lat: float/int
        the site latitude [rad]
    Returns
    -------
        Series of solar declination [rad].

    """
    j = day_of_year(meteoindex)
    dr = relative_distance(j)
    sol_dec = solar_declination(j)

    omega = sunset_angle(lat, sol_dec)
    xx = sin(sol_dec) * sin(lat)
    yy = cos(sol_dec) * cos(lat)
    return 118.08 / 3.141592654 * dr * (omega * xx + yy * sin(omega))


def extraterrestrial_r_hour(meteoindex, lat, lz=0, lm=0):
    """
    Returns Extraterrestrial Radiation (Ra).

    Based on equation 21 in Allen et al (1998).
    Parameters
    ----------
    meteoindex: Series.index
    lat: float/int
        the site latitude [rad]
    Returns
    -------
        Series of solar declination [rad].

    """
    j = day_of_year(meteoindex)
    dr = relative_distance(j)
    sol_dec = solar_declination(j)

    omega2, omega1 = sunset_angle_hour(meteoindex, lz=lz, lm=lm,
                                       lat=lat, sol_dec=sol_dec)
    xx = sin(sol_dec) * sin(lat)
    yy = cos(sol_dec) * cos(lat)
    gsc = 4.92
    return 12 / pi * gsc * dr * \
           ((omega2-omega1) * xx + yy *
            (sin(omega2)-sin(omega1)))