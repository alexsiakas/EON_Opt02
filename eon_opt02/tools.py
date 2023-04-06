
__all__ = ['conver_to_jd', 'get_sensor', 'detrend', 'periodogram', 'get_range_phase', 'build_model', 'FalsePositive', 'test_trend','full_PDM']

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
    if str(OBSCODE) == 'AUTH1':
        observatory = 'Noesis'
        telescope = 'Rasa8'
        sensor = 'QHY268'
        filter = 'none'
        latitude = 40.562694
        longitude = 22.995556
        elevation = 63
    else:
        observatory = 'nan'
        telescope = 'nan'
        sensor = 'nan'
        filter = 'nan'
        latitude = 'nan'
        longitude = 'nan'
        elevation = 'nan'
    return observatory, telescope, sensor, filter, latitude, longitude, elevation


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
        trend_type = 'polynomial'

    else:

        half_window = int(half_window / np.median(times[1:] - times[:-1]))

        if len(times) < limit_to_single_winow * half_window:

            trend = np.poly1d(np.polyfit(times , mag, single_window_poly_deg))(times)
            detrended_times = times
            detrended_mag = mag - trend
            trend_type = 'polynomial'

        else:

            trend = moving_poly(times, mag, half_window, poly_deg)
            detrended_times = times[half_window + 1: - half_window + 1]
            detrended_mag = mag[half_window + 1: - half_window + 1]  - trend
            trend_type = 'moving_poly'

    return trend, detrended_times, detrended_mag, trend_type

def test_trend(trend, jd, mag, peaks, fake_peaks, harmonic_peaks, trend_type,limit = 0.2,
                period_max=2.0, period_min=0.5, period_step=0.01, fap_limit=0.001, long_period_peak_ratio=0.9,
                cleaning_max_power_ratio=0.2, cleaning_alliase_proximity_ratio =0.2,
                pdm_bins = 20): 
    ###########################################################################
    ## This function tests the effectivness of the trend. The first test removes any 
    ## period detected in trend that is also detected in signal and is not considered fake.
    ## The second test forces trend to follow any relativelly long periods (|T/Duration  - 1| < 0.2 )  
    ## present in raw lightcurve and that is considered fake. 
    ## 
    ## This function exploits the periodogram function and model_signals function
    ## and can only exist after detrend function
    ###########################################################################
    trend_periodogram = None
    if trend_type == 'moving_poly':
    # find periods in trend 
        (trend_periodogram, periods, Long_p, _, _,_) = periodogram_trend(jd, trend, period_max=period_max, period_min=period_min,
                                      period_step=period_step, fap_limit=fap_limit,
                                      long_period_peak_ratio=long_period_peak_ratio,
                                      cleaning_max_power_ratio=cleaning_max_power_ratio,
                                      cleaning_alliase_proximity_ratio=cleaning_alliase_proximity_ratio,
                                      pdm_bins=pdm_bins)
        
        # test if any of the peaks found in trend is real
        #print(Long_p)
        #print('test periodogram periods')
        #print(periods)
        #print('And Peaks')
        if len(peaks) > 0 and len(harmonic_peaks) > 0:
            real_peaks = np.concatenate((peaks, harmonic_peaks))
        elif len(peaks) > 0:
            real_peaks = peaks
        else:
            real_peaks = harmonic_peaks
        
        #print(real_peaks)
        
        if len(periods)>0:
            true_period_in_trend = []

            for period_idx in range(len(periods)):
                for peak_idx in range(len(real_peaks)):
                    if abs(real_peaks[peak_idx][1] - periods[period_idx][1])/ real_peaks[peak_idx][1] < limit:
                        true_period_in_trend.append(period_idx)

            periods = np.array(periods)
            true_periods = periods[true_period_in_trend]
            _,_,_,_,trend_signals = model_signals(jd,trend,true_periods)
            trend_out = trend - np.sum(trend_signals,0)
            jd_out = jd
            mag_out = mag + np.sum(trend_signals,0)
        else:
            trend_out = trend
            jd_out = jd
            mag_out = mag

    else:
        if len(fake_peaks) > 0 and len(peaks) < 1:
            hw = len(jd)
            for fp in fake_peaks:
                if abs(fp[1]/(max(jd) - min(jd)) - 1) < limit:
                    step = np.median(jd[1:]-jd[:-1])
                    num_steps_in_peak = int(fp[1]/step)
                    window = int(num_steps_in_peak/7)
                    hw = int((window - 1)/2 if window%2 != 0 else window/2)

                    mag = mag + trend
                    trend_out = moving_poly(jd, mag, hw, 1)
                    jd_out = jd[hw + 1: - hw + 1]
                    mag_out = mag[hw + 1: - hw + 1]  - trend_out
                else:
                    trend_out = trend
                    jd_out = jd
                    mag_out = mag
        else:
            trend_out = trend
            jd_out = jd
            mag_out = mag


    return trend_out, jd_out, mag_out, trend_periodogram

def periodogram_trend(jd, mag,
                period_max=2.0, period_min=0.5, period_step=0.01, fap_limit=0.001, long_period_peak_ratio=0.9,
                cleaning_max_power_ratio=0.2, cleaning_alliase_proximity_ratio =0.2,
                pdm_bins = 20):
    ######################################################################################################
    ## This function is used to test trend. The main and only difference from the normal periodogramm 
    ## function lies in the fact that all peaks (>20% maximum) are accepted. No aliaces or harmonics are 
    ## removed
    ##########################
    ## Inputs same as periodogram function
    ######################################################################################################

    periods = np.arange(
        np.log10(max(period_min,2*np.median(jd[1:] - jd[:-1]))),
        np.log10(period_max * (np.max(jd) - np.min(jd))),
        period_step)

    periods = 10**(periods)
    freqences = 1 / periods
    freqences = freqences[::-1]

    clp = pyPeriod.Gls((jd, mag), freq = freqences)
    plevel = clp.powerLevel(fap_limit) #max(clp.powerLevel(fap_limit), cleaning_max_power_ratio * np.max(clp.power))

    periodogram = [clp.freq, clp.power]
    peaks = []
    signals = []

    if np.sum(clp.power > plevel):

        output = np.where(clp.power > plevel)[0]
        section_starts = np.append(output[0], output[1:][np.where(output[1:] - output[:-1] > 1)])
        section_ends = np.append(output[np.where(output[1:] - output[:-1] > 1)], output[-1])
        sections = np.swapaxes([section_starts, section_ends], 0, 1)
        #print(f'there are {len(sections)} sections')
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

 
    periods, long_period, total_signal, lt_signal, signals  = model_signals(jd, mag, peaks)

    return np.array(periodogram), np.array(periods), long_period, np.array(total_signal), np.array(lt_signal), np.array(signals)


def periodogram(jd, mag,
                period_max=2.0, period_min=0.5, period_step=0.01, fap_limit=0.001, long_period_peak_ratio=0.9,
                cleaning_max_power_ratio=0.2, cleaning_alliase_proximity_ratio =0.2,
                pdm_bins = 20):

    periods = np.arange(
        np.log10(max(period_min,2*np.median(jd[1:] - jd[:-1]))),
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

    low_power_peaks = []
    #         cleaning weak peaks
    if len(peaks) > 0:
        power = np.array([peak[0] for peak in peaks], dtype = float)
        for pow_idx in range(len(power)):
            if (power[pow_idx]/np.max(power)) < cleaning_max_power_ratio:
                low_power_peaks.append(peaks[pow_idx])
        peaks = np.delete(peaks,  np.where(power/np.max(power) < cleaning_max_power_ratio)[0], axis=0)

    #     not yet    cleaning harmonics
    if len(peaks) > 1:

        peaks = sorted(peaks, key=lambda x :-x[0])

        '''cleaning = True

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
                cleaning = False'''







    #         models ans PDM
    periods, long_period, total_signal, lt_signal, signals  = model_signals(jd, mag, peaks)
    low_power_periods, _, low_power_total_signal,_,low_power_signals = model_signals(jd,mag,low_power_peaks)
    if len(low_power_periods)>0:
        total_signal = total_signal+np.sum(low_power_signals,0)
    

    #pdm_peaks = []
    #pdm_peak_thetas = []

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
        #pdm_peaks.append(pdm_per)
        #pdm_peak_thetas.append(thetas)


    return np.array(periodogram), np.array(periods), long_period, np.array(total_signal), np.array(lt_signal), np.array(signals), np.array(low_power_periods),np.array(low_power_signals)

def full_PDM(jd, mag, dominant_periods,harmonic_periods, period_max=2.0, period_min=0.5, period_step=0.001, pdm_bins = 20 ):
    periods = np.arange(
        np.log10(max(period_min, 2*np.median(jd[1:] - jd[:-1]))),
        np.log10(period_max * (np.max(jd) - np.min(jd))),
        period_step)

    periods = 10**(periods)

    S = pyPDM.Scanner(listVal = periods, mode='period')
    P = pyPDM.PyPDM(jd,mag)
    pdm_per, thetas = P.pdmEquiBin(pdm_bins,S)


    pdm_dominant_period = []

    pdm_min = pdm_per[np.where(thetas== np.min(thetas))]
    #print(pdm_min)
    if len(pdm_min) > 1:
        #print('There are 2 with min value')
        if np.abs(pdm_min[0] - pdm_min[1])/pdm_min[0] < 0.05:
            pdm_min = pdm_min[0]
        else:
            print('Two peaks in pdm, checking./n')
            per_min = 1
            for pdmin in pdm_min:
                extra_periods = np.arange(max(0,pdmin-0.1*pdmin),pdmin+0.1*pdmin, ((pdmin+0.01*pdmin)-(pdmin-0.1*pdmin))/21)
                S2 = pyPDM.Scanner(listVal = extra_periods, mode='period')
                pdm_test, theta_test = P.pdmEquiBin(pdm_bins, S2)

                if per_min > pdm_test[np.argmin(theta_test)]:
                    per_min = pdm_test[np.argmin(theta_test)]
            pdm_min = per_min
    if len(dominant_periods) > 0:
        for period in dominant_periods:
            if np.abs(pdm_min - period[1])/period[1] < 0.15:
                pdm_dominant_period = period
        for period in harmonic_periods:
            if np.abs(pdm_min - period[1])/period[1] < 0.15:
                pdm_dominant_period = period

        if len(pdm_dominant_period) < 1:
            test_period = dominant_periods[0][1]*dominant_periods[0][6]
            for period in harmonic_periods:
                if np.abs(test_period - period[1])/period[1] < 0.2:
                    pdm_dominant_period = period

    if len(pdm_dominant_period) < 1:
        pdm_dominant_period = None
        print('pdm_was_None')

    return pdm_per, thetas, pdm_dominant_period


def model_signals (jd,mag,peaks):
    ###########################################################################
    ## Calculate the sinusoidal model describing the signal, based on the 
    ## peaks of the periodogram
    ## 
    ## This function is combined with the results of the periodogram function
    ###########################################################################
    ## INPUT
    ## jd        list of time instances 
    ## mag       list of recorded magnitude
    ## peaks     array of peaks returned by the periodogram, each line contains
    ##        [power, period, apmplitude, phase, offset, long period statement] 
    ###########################################################################
    ## OUTPUT
    ## periods          array of peaks (long period excluded)
    ## long_period      bool True if any of the peaks is considered a 
    ##                  potentialy long period
    ## total_signal     list combination of all signals
    ## lt_signal        long period signal
    ## signals          array containing all signals separately 
    
    ###########################################################################
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
    
    return periods, long_period, total_signal, lt_signal, signals


def FalsePositive(jd,mag, peaks, signals, false_limit = 0.2,
                period_max=2.0, period_min=0.5, period_step=0.01, fap_limit=0.001, long_period_peak_ratio=0.9,
                cleaning_max_power_ratio=0.2, cleaning_alliase_proximity_ratio =0.2,
                pdm_bins = 20):
    ###########################################################################
    ## This function removes the detcted signal from the light curve, then 
    ## performs a periodogramm to the resulting time series and compares the 
    ## resulting peaks. If any of the new peaks matches an old one the second 
    ## is considered a false positive
    ## 
    ## This function exploits the periodogram function and model_signals function
    ###########################################################################
    ## INPUT
    ## jd           list of time instances 
    ## mag          list of recorded magnitude
    ## peaks        array of peaks returned by the periodogram, each line contains
    ##              [power, period, apmplitude, phase, offset, long period statement] 
    ## total_signal the detected signal
    ## 
    ## rest of inputs same as periodogram function
    ###########################################################################
    ## OUTPUT
    ## periods_out          array of (true positive) peaks
    ## total_signal_out     list combination of all (true) signals
    ## signals_out          array containing all (true) signals separately 
    ## 
    ###########################################################################
    
    # This is the normal procedure
    tot_sig = np.sum(signals, 0)
    #harm_tot_sig = np.sum(harmonic_signals,0)
    #tot_sig = tot_sig-harm_tot_sig
    # remove detected signals from the list
    clean_mag = np.ones_like(mag) * mag - tot_sig
    
    #compute periods on the "cleaned" list
    (_, periods, _, _, _,_,_,_) = periodogram(jd, clean_mag, period_max=period_max, period_min=period_min,
                                      period_step=period_step, fap_limit=fap_limit,
                                      long_period_peak_ratio=long_period_peak_ratio,
                                      cleaning_max_power_ratio=cleaning_max_power_ratio,
                                      cleaning_alliase_proximity_ratio=cleaning_alliase_proximity_ratio,
                                      pdm_bins=pdm_bins)
    
    # check if any of the new peaks is close enough to the old ones
    false_pos_index = []
    if len(periods) > 0:    
        for peak_idx in range(len(peaks)):
            for period_idx in range(len(periods)):
                if np.abs(peaks[peak_idx][1] - periods[period_idx][1])/peaks[peak_idx][1] < false_limit: # |T_old - T_new|/T_old < limit
                    false_pos_index.append(peak_idx)

    # altervative checking one peack at a time
    '''false_pos_index = []
    h_false_pos_index = []
    for peak_idx in range(len(peaks)):
        clean_mag = mag - signals[peak_idx]
        (_, periods, _, _, _,_,_) = periodogram(jd, clean_mag, period_max=period_max, period_min=period_min,
                                      period_step=period_step, fap_limit=fap_limit,
                                      long_period_peak_ratio=long_period_peak_ratio,
                                      cleaning_max_power_ratio=cleaning_max_power_ratio,
                                      cleaning_alliase_proximity_ratio=cleaning_alliase_proximity_ratio,
                                      pdm_bins=pdm_bins)
        
        for period_idx in range(len(periods)):
            if np.abs(peaks[peak_idx][1] - periods[period_idx][1])/peaks[peak_idx][1] < false_limit: # |T_old - T_new|/T_old < limit
                false_pos_index.append(peak_idx)'''


    peaks_out = np.delete(peaks, false_pos_index, axis=0)
    fake_peaks = peaks[false_pos_index]
    
    
    harmonic_peaks = []

    if len(peaks_out) > 1:

        peaks_out = sorted(peaks_out, key=lambda x :-x[0])

        cleaning = True

        ratios_to_clean = np.arange(2.0, 9.0, 1.0)
        ratios_to_clean = np.append(ratios_to_clean, np.array([3/2, 5/2, 7/2]))
        ratios_to_clean = np.append(ratios_to_clean, np.array([4/3, 5/3, 7/3, 8/3]))
        
        #print(peaks)
        while cleaning and len(peaks_out) > 1:

            restart = False
            #                 print(peaks)

            for i in range(len(peaks_out)):
                #                     print(i)
                for j in range(len(peaks_out)):
                    #                         print(j)
                    if i != j:
                        test = max(peaks_out[i][1] / peaks_out[j][1], peaks_out[j][1] / peaks_out[i][1])
                        idx = np.argmin(np.abs(test - ratios_to_clean))
                        if np.abs(test - ratios_to_clean[idx]) <  cleaning_alliase_proximity_ratio+(ratios_to_clean[idx]-1)*0.03:
                            to_clean = [i,j][int(peaks[i][0] > peaks[j][0])]
                            if not peaks[to_clean][5]:
                                #print('cleaning... ', to_clean)
                                harmonic_peaks.append(peaks_out[to_clean])
                                peaks_out = np.delete(peaks_out, [to_clean], axis=0)
                                #print(peaks)
                                restart = True
                                break
                if restart:
                    break

            if not restart:
                cleaning = False
    # delete false positives

    # create new models 
    periods_out, _, total_signal_out, _, signals_out = model_signals(jd,mag, peaks_out)
    
    h_periods_out, _,h_total_signal,_,h_signals_out = model_signals(jd,mag,harmonic_peaks) 
    total_signal_out = total_signal_out+np.sum(h_signals_out,axis=0)

    return periods_out, total_signal_out, signals_out, fake_peaks, h_periods_out ,h_signals_out,


def comp_tr_lp(x,y, std_limit):
    if len(x) == len(y):
        differ = np.array(x) - np.array(y)
    else:
        hw = int((len(y) - len(x))/2)
        differ = np.array(x) - np.array(y[hw + 1: - hw + 1])
    md = np.mean(differ) 
    stdd = np.std(differ) 
    res = False
    if np.abs(md) < stdd and stdd < std_limit:    # if mean value < 0.1 (close to 0) and std < std_lc , trend == sin model 
        res = True
    return res


def build_model(mag, trend, periods, harmonics, low_power): 
    model = trend 
    if len(periods) > 0:
        model = model + np.sum(periods, 0)

        if len(harmonics) > 0:
            model = model + np.sum(harmonics, 0)

        if len(low_power) > 0:
            model = model + np.sum(low_power, 0)
    model_plot = model
    if len(model) < len(mag):
        hw = int((len(mag) - len(model))/2)
        addnan = np.empty(hw)
        addnan[:] = np.nan
        model = np.append(addnan,model)
        model = np.append(model,addnan)
    return model, model_plot



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



