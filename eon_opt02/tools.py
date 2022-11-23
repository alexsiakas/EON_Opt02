
__all__ = ['conver_to_jd', 'get_sensor', 'detrend', 'periodogram', 'get_range_phase']

import numpy as np
import datetime
from PyAstronomy.pyTiming import pyPeriod
from PyAstronomy.pyTiming import pyPDM
import ephem

class Scanner:

    def __init__(self, listVal, mode="period"):
        """
          The Scanner class is used to define the period/frequency range and steps used in the \
          PDM analysis. It is iteratable.

          Parameters:
            - `minVal` - float, Minimum value,
            - `maxVal` - float, Maximum value,
            - `dVal`   - float, Delta value,
            - `mode`   - string, optional, Either "period" or "frequency" (default = "period").

          .. Note: Whether min, max, and dval refer to period or frequency depends on the ``mode'' parameter.
          .. Note: Time units have match the time axis given to the PDM analysis class.
        """
        self.minVal = min(listVal)
        self.maxVal = max(listVal)
        self.dVal = None
        self.listVal = listVal
        self.mode = mode

    def __iter__(self):
        ii = 0
        jj = len(self.listVal)
        while ii < jj:
            yield self.listVal[ii]
            ii += 1


pyPDM.Scanner = Scanner


def conver_to_jd(datetime_in):
    time_dif = datetime_in - datetime.datetime(1600, 1, 1, 0, 0, 0, 0)
    return time_dif.days + (time_dif.seconds + time_dif.microseconds / 1000000.0) / 60.0 / 60.0 / 24.0 + 2305447.5


def get_sensor(OBSCODE):

    #     TODO - implement search in the sensor book
    #     Should return:
    #     observatory (string)
    #     telescope (string)
    #     sensor (string)
    #     filter
    #     latitude (deg)
    #     longitude (deg)
    #     elevation (m)

    return 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan'


def moving_poly(x,y,w,n):
    mp = []
    for index in range(w,len(x)-w):
        trend = np.poly1d(np.polyfit(x[index-w:index+w+1], y[index-w:index+w+1], n))
        mp.append(trend(x[index]))
    return np.array(mp)


def detrend(times, mag, half_window=10, poly_deg=1, limit_to_single_winow=5, single_window_poly_deg=2):

    if np.max(times) - np.min(times) < limit_to_single_winow * half_window:

        trend = np.poly1d(np.polyfit(times , mag, single_window_poly_deg))(times)
        detrended_times = times
        detrended_mag = mag - trend

    else:

        half_window = int(half_window / np.median(times[1:] - times[:-1]))

        if len(times) < limit_to_single_winow * half_window:

            trend = np.poly1d(np.polyfit(times , mag, single_window_poly_deg))(times)
            detrended_times = times
            detrended_mag = mag - trend

        else:

            trend = moving_poly(times, mag, half_window, poly_deg)
            detrended_times = times[half_window + 1: - half_window + 1]
            detrended_mag = mag[half_window + 1: - half_window + 1]  - trend

    return trend, detrended_times, detrended_mag


def periodogram(jd, mag,
                period_max=2.0, period_min=0.5, period_step=0.01, fap_limit=0.001, long_period_peak_ratio=0.9,
                cleaning_max_power_ratio=0.2, cleaning_alliase_proximity_ratio =0.2,
                pdm_bins = 20):

    periods = np.arange(
        np.log10(period_min),
        np.log10(period_max * (np.max(jd) - np.min(jd))),
        period_step)

    periods = 10**(periods)
    freqences = 1 / periods
    freqences = freqences[::-1]

    clp = pyPeriod.Gls((jd, mag), freq = freqences)
    plevel = max(clp.powerLevel(fap_limit), cleaning_max_power_ratio * np.max(clp.power))

    periodogram = [clp.freq, clp.power]
    peaks = []
    signals = []

    if np.sum(clp.power > plevel):

        output = np.where(clp.power > plevel)[0]
        section_starts = np.append(output[0], output[1:][np.where(output[1:] - output[:-1] > 1)])
        section_ends = np.append(output[np.where(output[1:] - output[:-1] > 1)], output[-1])
        sections = np.swapaxes([section_starts, section_ends], 0, 1)

        for section in sections:

            clp = pyPeriod.Gls((jd, mag),
                               freq=np.linspace(
                                   periodogram[0][max(0, section[0] - 1)],
                                   periodogram[0][min(len(periodogram[0]) - 1, section[1] + 1)],
                                   1000)
                               )
            frequency = clp.freq
            power = clp.power
            ifmax = np.argmax(clp.power)
            pmax = clp.power[ifmax]
            fmax = clp.freq[ifmax]
            amax = np.sqrt(clp._a[ifmax]**2 + clp._b[ifmax]**2)
            phase = np.arctan2(clp._a[ifmax], clp._b[ifmax]) / (2.*np.pi)
            off = clp._off[ifmax] + clp._Y

            if section[0] == 0 and power[0] > long_period_peak_ratio * np.max(power):
                long_period = True
            elif 1./fmax > max(jd) - min(jd):
                long_period = True
            else:
                long_period = False

            peaks.append([pmax, 1./fmax, amax, phase, off, long_period])

    #         cleaning weak peaks
    if len(peaks) > 0:
        power = np.array([peak[0] for peak in peaks], dtype = float)
        peaks = np.delete(peaks,  np.where(power/np.max(power) < cleaning_max_power_ratio)[0], axis=0)


    #         cleaning alliaces
    if len(peaks) > 1:

        peaks = sorted(peaks, key=lambda x :-x[0])

        cleaning = True

        ratios_to_clean = np.arange(2.0, 9.0, 1.0)

        while cleaning and len(peaks) > 1:

            restart = False
            #                 print(peaks)

            for i in range(len(peaks)):
                #                     print(i)
                for j in range(len(peaks)):
                    #                         print(j)
                    if i != j:
                        test = max(peaks[i][1] / peaks[j][1], peaks[j][1] / peaks[i][1])
                        if min(np.abs(test - ratios_to_clean)) <  cleaning_alliase_proximity_ratio:
                            to_clean = [i,j][int(peaks[i][0] > peaks[j][0])]
                            if not peaks[to_clean][5]:
                                #                                     print('cleaning... ', to_clean)
                                peaks = np.delete(peaks, [to_clean], axis=0)
                                #                                     print(peaks)
                                restart = True
                                break
                if restart:
                    break

            if not restart:
                cleaning = False


    #         models ans PDM
    lt_signal = np.ones_like(jd) * np.median(mag)
    long_period = False
    periods = []
    signals = []

    for peak in peaks:
        if peak[5]:
            long_period = True
            lt_signal = peak[2]*np.sin((2*np.pi/peak[1])*(jd - (min(jd) - peak[3]*peak[1]))) + peak[4]
        else:
            periods.append(list(peak))
            signals.append(peak[2]*np.sin((2*np.pi/peak[1])*(jd - (min(jd) - peak[3]*peak[1]))))

    total_signal = lt_signal + np.sum(signals, 0)

    for period_idx in range(len(periods)):

        isolated_signal = mag - total_signal + signals[period_idx]

        S = pyPDM.Scanner(listVal = np.array([periods[period_idx][1]*ff for ff in
                                              range(1,
                                                    min(7, 1 + int((max(jd) - min(jd))/periods[period_idx][1]))
                                                    )
                                              ]),
                          mode = 'period' )
        P = pyPDM.PyPDM(jd, isolated_signal)
        pdm_per, thetas = P.pdmEquiBin(pdm_bins, S)
        periods[period_idx].append(np.argmin(thetas) + 1)


    return np.array(periodogram), np.array(periods), long_period, np.array(total_signal), np.array(lt_signal), np.array(signals)


def get_range_phase(tle_line1, tle_line2, dates, sensor_lat, sensor_lon, sensor_elev):
    ###########################################################################
    ## Calculate the range and phase angle of a satellite defined by a TLE
    ## from a given sensor.
    ## requires: PyEphem
    ###########################################################################
    ## INPUT
    ## tle_line1    string with the first TLE line
    ## tle_line2    string with the second TLE line
    ## dates        list of dates in JD
    ## sensor_lat   sensor's latitude [rad]
    ## sensor_lat   sensor's longitude [rad]
    ## sensor_elev  sensor's elevations [m]
    ###########################################################################
    ## OUTPUT
    ## range        list of satellite-sensor distance for each date
    ## phase        list of satellite phase angle for each date
    ###########################################################################
    jd_offset = -2415020.0 # PyEphem actually represents dates as the number of days since noon on 1899 December 31
    AU = 149597871 # PyEphem returns sun-earth distance in AU
    sensor = ephem.Observer() #sensor object
    sensor.lon = np.deg2rad(sensor_lon) #longitude [rad]
    sensor.lat = np.deg2rad(sensor_lat) #latitude [rad]
    sensor.elevation = sensor_elev #elevation [m]
    line0 = "0 (00000)"
    line1 = tle_line1
    line2 = tle_line2
    sat = ephem.readtle(line0, line1, line2) #satellite object
    outrange = np.zeros(len(dates)) #output array
    outphase = np.zeros(len(dates)) #output array
    for i,date in enumerate(dates):
        sensor.date = date + jd_offset #update observer date
        sat.compute(sensor) #compute satellite observer ephemeris
        rsat = sat.range/1000.0 # satellite range [km]
        sun = ephem.Sun(sensor) # compute sun observer ephemeris
        rsun = sun.earth_distance*AU # earth-sun distance [km]
        psi = np.arccos(np.sin(sun.dec)*np.sin(sat.dec)+np.cos(sun.dec)*np.cos(sat.dec)*np.cos(sun.ra-sat.ra)) # solar elongation
        phase = np.arctan2(rsun * np.sin(psi),(rsat - rsun * np.cos(psi))) #phase angle
        outrange[i] = rsat # add to output list
        outphase[i] = phase # add to output list
    return outrange,outphase



