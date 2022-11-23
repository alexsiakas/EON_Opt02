
__all__ = ['LightCurve']

import numpy as np
import datetime
import os
import matplotlib.pyplot as plt

from eon_opt02.tools import *


class LightCurve:

    def __init__(self, file_path, analyse=False, export=None, show=False):

        self.name = os.path.split(file_path)[1].split('.')[0]

        #         Load file

        tdm = open(file_path).read()

        #         Load meta-data, they can be acessed by directly from LightCurve object
        #         e.g. lightgurve.OBSCODE

        for line in tdm.split('DATA_START')[0].split('\n'):
            line = line.replace('COMMENT', '')
            if '=' in line:
                columns = line.split('=')
                try:
                    setattr(self, columns[0].replace(' ', ''), float(columns[1]))
                except ValueError:
                    setattr(self, columns[0].replace(' ', ''), str(columns[1]))

        #         Load data, calculate jd and sort by time, they can be acessed by directly from LightCurve object
        #         e.g. lightgurve.MAG
        #         Add also UTC, JD and DT attributes (DT is the time since the start of the observation in seconds)

        data = {}

        for line in tdm.split('DATA_START')[1].split('DATA_STOP')[0].split('\n'):
            if 'MAG' in line:
                utc = line.split()[2]
                year, month, day = utc.split('T')[0].split('-')
                hour, minute, second = utc.split('T')[1].split(':')
                if '.' in second:
                    second, microseconds = second.split('.')
                else:
                    microseconds = '0'
                while len(microseconds) > 6:
                    microseconds = str(round(float(microseconds)/10))
                microseconds = microseconds.ljust(6, '0')
                dt_obj = datetime.datetime(int(year), int(month), int(day), int(hour), int(minute), int(second),
                                           int(microseconds))
                data[utc] = {'JD': conver_to_jd(dt_obj), 'DATETIME': dt_obj}

        time_series_keys = ['JD', 'DATETIME']
        for line in tdm.split('DATA_START')[1].split('DATA_STOP')[0].split('\n'):
            line = line.replace('COMMENT', '')
            if '=' in line:
                columns = [ff.replace(' ', '') for ff in line.replace('=', ' ').split()]
                try:
                    data[columns[1]][columns[0]] = float(columns[2])
                except ValueError:
                    data[columns[1]][columns[0]] = str(columns[2])
                if columns[0] not in time_series_keys:
                    time_series_keys.append(columns[0])

        self.UTC = np.array(sorted(list(data.keys()), key=lambda x: data[x]['JD']))

        for time_series_key in time_series_keys:
            setattr(self, time_series_key, np.array([data[ff][time_series_key] for ff in self.UTC]))

        self.DT = np.array([(ff-self.DATETIME[0]).total_seconds() for ff in self.DATETIME])

        del tdm, data

        #         Load sensor data

        (self.observatory, self.telescope, self.sensor, self.filter, self.latitude,
         self.longitude, self.elevation) = get_sensor(self.OBSCODE)

        #         Analyse

        if analyse:
            self.analyse(export=export, show=show)

    def plot(self, directory=None):

        fig = plt.figure(figsize=(9, 10))

        fig.suptitle(self.name)

        ax1 = fig.add_subplot(4, 2, 1)
        ax1.plot(self.DT, self.MAG, 'ko', ms=1, label='Data', zorder=1)
        ax1.plot(self.detrended_dt, self.trend, 'r-', ms=1, label='Trend', zorder=2)
        ax1.plot(self.DT, self.total_signal1, 'g-', label='Sin model', zorder=3)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Magnitude')
        ax1.set_title('RAW LC')
        ax1.legend(bbox_to_anchor=(0.3, 1.5))

        ax2 = fig.add_subplot(4, 2, 3)
        ax2.set_xlabel("Period (s)")
        ax2.set_ylabel("Power")
        ax2.plot(1/self.periodogramm1[0], self.periodogramm1[1], 'b', zorder=1)
        for period in self.periods1:
            ax2.axvline(period[1], ls='--', c='r', zorder=2)
        ax2.set_xscale('log')
        ax2.set_title('RAW LC / Periodogramm')

        if self.long_period:
            ylim = ax2.get_ylim()
            ax2.fill_between([np.max(self.DT) - np.min(self.DT), np.max(1/self.periodogramm1[0])],
                             -1, 2, color='r', alpha=0.1)
            ax2.set_ylim(ylim)

        if len(self.periods1) > 0:
            cleaned_mag = self.MAG - self.total_signal1 + self.signals1[0]
            cleaned_signal = self.signals1[0]
            phase = self.DT / self.periods1[0][1]
            phase -= np.int_(phase)
            phase *= self.periods1[0][1]
        else:
            phase = self.DT
            cleaned_mag = self.MAG - self.lt_signal1
            cleaned_signal = self.lt_signal1 * 0

        phase, cleaned_mag, cleaned_signal = np.swapaxes(
            sorted(np.swapaxes([phase, cleaned_mag, cleaned_signal], 0, 1), key=lambda x: x[0]),
            0, 1)

        ax3 = fig.add_subplot(4, 2, 5)
        ax3.plot(phase, cleaned_mag, 'ko', ms=1, zorder=1)
        ax3.plot(phase, cleaned_signal, 'g-', zorder=2)
        ax3.set_xlabel('Time in phase of dominant period (s)')
        ax3.set_ylabel('Magnitude')
        ax3.set_title('RAW LC / Signal of dominant period')

        if len(self.periods1) > 0:
            cleaned_mag = self.MAG - self.total_signal1 + self.signals1[0]
            cleaned_signal = self.signals1[0]
            phase = self.DT / self.periods1[0][1] / self.periods1[0][6]
            phase -= np.int_(phase)
            phase *= self.periods1[0][1]*self.periods1[0][6]
        else:
            phase = self.DT
            cleaned_mag = self.MAG - self.lt_signal1
            cleaned_signal = self.lt_signal1 * 0

        phase, cleaned_mag, cleaned_signal = np.swapaxes(
            sorted(np.swapaxes([phase, cleaned_mag, cleaned_signal], 0, 1), key=lambda x: x[0]),
            0, 1)

        ax4 = fig.add_subplot(4, 2, 7)
        ax4.plot(phase, cleaned_mag, 'ko', ms=1, zorder=1)
        ax4.plot(phase, cleaned_signal, 'g-', zorder=2)
        ax4.set_xlabel('Time in phase of dominant period (s)')
        ax4.set_ylabel('Magnitude')
        ax4.set_title('RAW LC / Signal of dominant period in PDM')

        #         Detrended

        ax5 = fig.add_subplot(4, 2, 2)
        ax5.plot(self.detrended_dt, self.detrended_mag, 'ko', ms=1, zorder=1)
        ax5.plot(self.detrended_dt, self.total_signal2, 'g-', zorder=2)
        ax5.set_xlim(ax1.get_xlim())
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Magnitude')
        ax5.set_title('DT LC')

        ax6 = fig.add_subplot(4, 2, 4)
        ax6.set_xlabel("Period (s)")
        ax6.set_ylabel("Power")
        ax6.plot(1/self.periodogramm2[0], self.periodogramm2[1], 'b', zorder=1)
        for period in self.periods2:
            ax6.axvline(period[1], ls='--', c='r', zorder=2)
        ax6.set_xlim(ax2.get_xlim())
        ax6.set_xscale('log')
        ax6.set_title('DT LC / Periodogramm')

        if self.long_period2:
            ylim = ax6.get_ylim()
            ax6.fill_between([np.max(self.detrended_dt) - np.min(self.detrended_dt), np.max(1/self.periodogramm2[0])],
                             -1, 2, color='r', alpha=0.1)
            ax6.set_ylim(ylim)

        if len(self.periods2) > 0:
            cleaned_mag = self.detrended_mag - self.total_signal2 + self.signals2[0]
            cleaned_signal = self.signals2[0]
            phase = self.detrended_dt / self.periods2[0][1]
            phase -= np.int_(phase)
            phase *= self.periods2[0][1]
        else:
            phase = self.detrended_dt
            cleaned_mag = self.detrended_mag - self.lt_signal2
            cleaned_signal = self.lt_signal2 * 0

        phase, cleaned_mag, cleaned_signal = np.swapaxes(
            sorted(np.swapaxes([phase, cleaned_mag, cleaned_signal], 0, 1), key=lambda x: x[0]),
            0, 1)

        ax7 = fig.add_subplot(4, 2, 6)
        ax7.plot(phase, cleaned_mag, 'ko', ms=1, zorder=1)
        ax7.plot(phase, cleaned_signal, 'g-', zorder=2)
        ax7.set_xlabel('Time in phase of dominant period (s)')
        ax7.set_ylabel('Magnitude')
        ax7.set_title('DT LC / Signal of dominant period')

        if len(self.periods2) > 0:
            cleaned_mag = self.detrended_mag - self.total_signal2 + self.signals2[0]
            cleaned_signal = self.signals2[0]
            phase = self.detrended_dt / self.periods2[0][1] / self.periods2[0][6]
            phase -= np.int_(phase)
            phase *= self.periods2[0][1]*self.periods2[0][6]
        else:
            phase = self.detrended_dt
            cleaned_mag = self.detrended_mag - self.lt_signal2
            cleaned_signal = self.lt_signal2 * 0

        phase, cleaned_mag, cleaned_signal = np.swapaxes(
            sorted(np.swapaxes([phase, cleaned_mag, cleaned_signal], 0, 1), key=lambda x: x[0]),
            0, 1)

        ax8 = fig.add_subplot(4, 2, 8)
        ax8.plot(phase, cleaned_mag, 'ko', ms=1, zorder=1)
        ax8.plot(phase, cleaned_signal, 'g-', zorder=2)
        ax8.set_xlabel('Time in phase of dominant period (s)')
        ax8.set_ylabel('Magnitude')
        ax8.set_title('DT LC / Signal of dominant period in PDM')

        fig.subplots_adjust(bottom=0.05, top=0.9, left=0.1, right=0.99, hspace=0.5, wspace=0.25)

        if directory:

            fig.savefig(os.path.join(directory, self.name + '.pdf'))
            plt.close('all')
            del fig

        else:

            fig.show()

    def csv(self, directory=None):

        csv = [
            '-----------------------START METADATA-------------------------',
            self.observatory,
            'Telescope: {0}'.format(self.telescope),
            'Sensor: {0}'.format(self.sensor),
            'Latitude [deg]: {0} '.format(self.latitude),
            'Longitude [deg]: {0}'.format(self.longitude),
            'Elevation [m]: {0}'.format(self.elevation),
            'Reduction data procedure: {0}'.format(None),
            'Uncertainty: {0}'.format('From TDM'),
            'Magnitude type: {0}'.format('From TDM'),
            'Angular measurements: {0}'.format('From TDM'),
            'First Epoch MJDutc: {0}'.format(self.JD[0] - 2400000.5),
            'First Epoch ISOutc: {0}'.format(self.UTC[0]),
            'Target NORADID: {0}'.format(int(self.NORADID)),
            'Phase Angle [rad]: {0}'.format(self.phase[0]),
            'Distance from observatory [km]: {0}'.format(self.distance[0]),
            'Filter used: {0}'.format(self.filter),
            'Exposure time [sec]: {0}'.format(
                round(np.median(np.int_(self.DT[1:]*1000000)-np.int_(self.DT[:-1]*1000000))/1000000, 6)
            ),
            'Periodicity class: {0}'.format(self.physical_class),
            'Period [s]: {0}'.format(self.period),
            'Additional periods [s]: {0}'.format(','.join([str(ff) for ff in self.additional_periods])),
            'Periodicity classification comments:\n{0}'.format(
                '\n'.join(['            ' + ff for ff in self.comments])
            ),
            '-----------------------END METADATA----------------------------',
        ]

        if self.ANGLE_TYPE == 'RADEC':
            csv.append('OBSid,TimeSinceFirstEpoch[sec],FOVra[deg],FOVdec[deg],SigPlate[deg],'
                       'Mag,SigMag,PhaseAngle,Distance,STMag')
        else:
            csv.append('OBSid,TimeSinceFirstEpoch[sec],FOVaz[deg],FOVel[deg],SigPlate[deg],'
                       'Mag,SigMag,PhaseAngle,Distance,STMag')

        for entry in range(len(self.JD)):
            csv.append(','.join([
                str(float(entry + 1)),
                str(self.DT[entry]),
                str(self.ANGLE_1[entry]),
                str(self.ANGLE_2[entry]),
                str(self.PLATE_SOLUTION_RMS[entry]),
                str(self.RAW_MAG[entry]),
                str(self.PHOTOMETRIC_RMS[entry]),
                str(self.phase[entry]),
                str(self.distance[entry]),
                str(self.stmag[entry]),
            ]))

        if directory:

            w = open(os.path.join(directory, self.name + '.csv'), 'w')
            w.write('\n'.join(csv))
            w.close()

        else:

            print('\n'.join(self.comments))

    def export(self, directory):

        self.plot(directory)
        self.csv(directory)

    def show(self):

        self.plot()
        self.csv()

    def analyse(self,
                period_max=2.0, period_min=0.5, period_step=0.01, fap_limit=0.001, long_period_peak_ratio=0.9,
                cleaning_max_power_ratio=0.2, cleaning_alliase_proximity_ratio=0.2, pdm_bins=20,
                half_window=10, poly_deg=1, limit_to_single_winow=5, single_window_poly_deg=2,
                export=None, show=False):

        #         Calculate disctance, phase and standard magnitude
        try:
            self.distance, self.phase = get_range_phase(
                self.FIRSTLINE, self.SECONDLINE, self.JD,
                self.latitude, self.longitude, self.elevation)
            self.stmag = self.MAG - 5 * np.log10(np.pi/(np.sin(self.phase)+(np.pi-self.phase)*np.cos(self.phase)))
            self.stmag = self.stmag + 5 - 5 * np.log10(self.distance/6378)
            self.RAW_MAG, self.MAG = self.MAG, self.stmag
        except:
            self.distance, self.phase = np.ones_like(self.JD) * np.nan, np.ones_like(self.JD) * np.nan
            self.stmag = np.ones_like(self.JD) * np.nan
            self.RAW_MAG, self.MAG = self.MAG, self.MAG

        #         raw lightcurve analysis

        #         periodogramm

        (self.periodogramm1, self.periods1, self.long_period, self.total_signal1, self.lt_signal1,
         self.signals1) = periodogram(self.DT, self.MAG, period_max=period_max, period_min=period_min,
                                      period_step=period_step, fap_limit=fap_limit,
                                      long_period_peak_ratio=long_period_peak_ratio,
                                      cleaning_max_power_ratio=cleaning_max_power_ratio,
                                      cleaning_alliase_proximity_ratio=cleaning_alliase_proximity_ratio,
                                      pdm_bins=pdm_bins)

        #         test of repetitions to clean residuals

        self.residuals_test1 = []

        test_mag = np.ones_like(self.MAG) * self.MAG - self.total_signal1
        for i in range(200):
            (periodogramm, periods, long_period, total_signal, lt_signal,
             signals) = periodogram(self.DT, test_mag, period_max=period_max, period_min=period_min,
                                    period_step=period_step, fap_limit=fap_limit,
                                    long_period_peak_ratio=long_period_peak_ratio,
                                    cleaning_max_power_ratio=cleaning_max_power_ratio,
                                    cleaning_alliase_proximity_ratio=cleaning_alliase_proximity_ratio,
                                    pdm_bins=pdm_bins)
            if len(periods) > 0:
                self.residuals_test1.append(periods[0])
                test_mag = test_mag - total_signal
            else:
                break

        #        detrend raw lightcurve

        self.trend, self.detrended_dt, self.detrended_mag = detrend(self.DT, self.MAG, half_window=half_window,
                                                                    poly_deg=poly_deg,
                                                                    limit_to_single_winow=limit_to_single_winow,
                                                                    single_window_poly_deg=single_window_poly_deg)

        #        compute periodogramm for detrended lightcurve

        (self.periodogramm2, self.periods2, self.long_period2, self.total_signal2, self.lt_signal2,
         self.signals2) = periodogram(self.detrended_dt, self.detrended_mag, period_max=period_max,
                                      period_min=period_min,
                                      period_step=period_step, fap_limit=fap_limit,
                                      long_period_peak_ratio=long_period_peak_ratio,
                                      cleaning_max_power_ratio=cleaning_max_power_ratio,
                                      cleaning_alliase_proximity_ratio=cleaning_alliase_proximity_ratio,
                                      pdm_bins=pdm_bins)

        self.residuals_test2 = []

        test_mag = np.ones_like(self.detrended_mag) * self.detrended_mag - self.total_signal2
        for i in range(200):
            (periodogramm, periods, long_period, total_signal, lt_signal,
             signals) = periodogram(self.detrended_dt, test_mag, period_max=period_max, period_min=period_min,
                                    period_step=period_step, fap_limit=fap_limit,
                                    long_period_peak_ratio=long_period_peak_ratio,
                                    cleaning_max_power_ratio=cleaning_max_power_ratio,
                                    cleaning_alliase_proximity_ratio=cleaning_alliase_proximity_ratio,
                                    pdm_bins=pdm_bins)
            if len(periods) > 0:
                self.residuals_test2.append(periods[0])
                test_mag = test_mag - total_signal
            else:
                break

        #         periodicity classification

        self.period = None
        self.additional_periods = []

        if not self.long_period:
            if len(self.periods1) == 0 and len(self.periods2) == 0:
                self.statistical_class = 1
                self.physical_class = 'Non Variable'
            elif len(self.periods1) > 0 and len(self.periods2) == 0:
                self.statistical_class = 3
                self.physical_class = 'Periodic variable'
                self.period = self.periods1[0][1] * self.periods1[0][6]
                for period in self.periods1[1:]:
                    self.additional_periods.append(period[1] * period[6])
            elif len(self.periods1) > 0 and len(self.periods2) > 0:
                self.statistical_class = 5
                self.physical_class = 'Periodic variable'
                self.period = self.periods2[0][1] * self.periods2[0][6]
                for period in self.periods2[1:]:
                    self.additional_periods.append(period[1] * period[6])
            elif len(self.periods1) == 0 and len(self.periods2) > 0:
                self.statistical_class = 7
                self.physical_class = 'Periodic variable'
                self.period = self.periods2[0][1] * self.periods2[0][6]
                for period in self.periods2[1:]:
                    self.additional_periods.append(period[1] * period[6])

        else:
            if len(self.periods1) == 0 and len(self.periods2) == 0:
                self.statistical_class = 2
                self.physical_class = 'Aperiodic variable with possible long-term periodicity'
            elif len(self.periods1) > 0 and len(self.periods2) == 0:
                self.statistical_class = 4
                self.physical_class = 'Periodic variable with possible long-term periodicity'
                self.period = self.periods1[0][1] * self.periods1[0][6]
                for period in self.periods1[1:]:
                    self.additional_periods.append(period[1] * period[6])
            elif len(self.periods1) > 0 and len(self.periods2) > 0:
                self.statistical_class = 6
                self.physical_class = 'Periodic variable with possible long-term periodicity'
                self.period = self.periods2[0][1] * self.periods2[0][6]
                for period in self.periods2[1:]:
                    self.additional_periods.append(period[1] * period[6])
            elif len(self.periods1) == 0 and len(self.periods2) > 0:
                self.statistical_class = 8
                self.physical_class = 'Periodic variable with possible long-term periodicity'
                self.period = self.periods2[0][1] * self.periods2[0][6]
                for period in self.periods2[1:]:
                    self.additional_periods.append(period[1] * period[6])

        if len(self.additional_periods) == 0:
            self.additional_periods = [None]

	# Half Lightcurve test	
	
        self.comments = ['Statistical class: {0}'.format(self.statistical_class)]
        self.comments.append('RAW LC:')
        if len(self.residuals_test1) > 0:
            self.comments.append('    Repetitions to clean residuals: {0}.'.format(len(self.residuals_test1)))
        self.comments.append('    Trend: {0}'.format(self.long_period))
        if self.long_period:
            self.comments.append('    Possible periodiciy beyond {0} s.'.format(max(self.DT)))
        for period in self.periods1:
            xx = '    Periodogramm peak at {0} s.'.format(period[1])
            if period[6] > 1:
                xx += ' Multiplied by {0} for minimum PDM.'.format(period[6])
            self.comments.append(xx)
        self.comments.append('DT LC:')
        self.comments.append('    Trend: {0}'.format(self.long_period2))
        if self.long_period2:
            self.comments.append('    Possible periodiciy beyond {0} s.'.format(max(self.DT)))
        if len(self.residuals_test2) > 0:
            self.comments.append('    Repetitions to clean residuals: {0}.'.format(len(self.residuals_test2)))
        for period in self.periods2:
            xx = '    Periodogramm peak at {0} s.'.format(period[1])
            if period[6] > 1:
                xx += ' Multiplied by {0} for minimum PDM.'.format(period[6])
            self.comments.append(xx)

        #         export or show

        if export:
            self.export(export)

        if show:
            self.show()
