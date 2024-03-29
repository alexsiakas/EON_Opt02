
__all__ = ['LightCurve']

import numpy as np
import datetime
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from fractions import Fraction
from eon_opt02.tools import *
import json

class LightCurve:

    def __init__(self, file_path, analyse=False, export=None, show=False, acc_key=None):

        self.name = os.path.split(file_path)[1].split('.')[0]
        #         Load file

        tdm = open(file_path).read()

        #         Load meta-data, they can be acessed by directly from LightCurve object
        #         e.g. lightgurve.SENSORid

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
        
        if not hasattr(self, 'SETUPID'):
            self.SETUPID=None

        if not hasattr(self,'SENSORID'):
            self.SENSORID = 'None'

        #         Load sensor data
        if not hasattr(self,'LATITUDE') or not hasattr(self,'LONGITUDE') or not hasattr(self,'ALTITUDE'):
            if acc_key:
                (self.observatory, self.telescope, self.camera, self.filter, self.LATITUDE,
                self.LONGITUDE, self.ALTITUDE) = get_sensor(self.SENSORID,acc_key=acc_key,setup_id=self.SETUPID)  
                self.PARTICIPANT_1 = self.observatory  
                self.MODE = 'Sequential'
            else:
                raise ValueError('Access key is mandatory to get data from SensorBook')

        if not hasattr(self,'telescope'):
            self.telescope = 'No info'

        if not hasattr(self,'filter'):
            self.filter = 'No info'

        if not hasattr(self,'camera'):
            self.camera = 'No info'

        if not hasattr(self,'TRACKLET'):
            self.TRACKLET = self.DATETIME[0].strftime("%Y%m%d%H%M")


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
        for fake in self.fake_peaks1:
            ax2.axvline(fake[1], ls='--', c='m', zorder=2)
        for low_p in self.low_power_periods1:
            ax2.axvline(low_p[1], ls='--', c='g', zorder=2)
        for har_p in self.harmonic_peaks1:
            ax2.axvline(har_p[1], ls='--', c='c', zorder=2)
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


        if self.pdm_dominant_period1 is not None:
            cleaned_mag = self.MAG - self.total_signal1 + self.pdm_dominant_period1[2]*np.sin((2*np.pi/self.pdm_dominant_period1[1])*(self.DT - (min(self.DT) - self.pdm_dominant_period1[3]*self.pdm_dominant_period1[1]))) 
            cleaned_signal = self.pdm_dominant_period1[2]*np.sin((2*np.pi/self.pdm_dominant_period1[1])*(self.DT - (min(self.DT) - self.pdm_dominant_period1[3]*self.pdm_dominant_period1[1])))
            phase = self.DT / self.pdm_dominant_period1[1]
            phase -= np.int_(phase)
            phase *= self.pdm_dominant_period1[1]
        else:
            phase = self.DT
            cleaned_mag = self.MAG - np.median(self.MAG)#self.lt_signal1
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
        for fake in self.fake_peaks2:
            ax6.axvline(fake[1], ls='--', c='m', zorder=2)
        for low_p in self.low_power_periods2:
            ax6.axvline(low_p[1], ls='--', c='g', zorder=2)
        for har_p in self.harmonic_peaks2:
            ax6.axvline(har_p[1], ls='--', c='c', zorder=2)
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
            cleaned_mag = self.detrended_mag - np.median(self.detrended_mag)#- self.lt_signal2
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

        
        if self.pdm_dominant_period2 is not None:
            cleaned_mag = self.detrended_mag - self.total_signal2 + self.pdm_dominant_period2[2]*np.sin((2*np.pi/self.pdm_dominant_period2[1])*(self.detrended_dt - (min(self.detrended_dt) - self.pdm_dominant_period2[3]*self.pdm_dominant_period2[1])))
            cleaned_signal = self.pdm_dominant_period2[2]*np.sin((2*np.pi/self.pdm_dominant_period2[1])*(self.detrended_dt - (min(self.detrended_dt) - self.pdm_dominant_period2[3]*self.pdm_dominant_period2[1]))) 
            phase = self.detrended_dt / self.pdm_dominant_period2[1]
            phase -= np.int_(phase)
            phase *= self.pdm_dominant_period2[1]
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

        fig2 = plt.figure(figsize=(9, 10))
        fig2.suptitle(self.name)

        ax21 = fig2.add_subplot(2,1,1)
        ax21.plot(self.pdm_per1, self.thetas1, 'b-')
        colors = ['r', 'g', 'k']
        for p_idx in range(0,2):
            try :
                pdmp = self.pdm_peaks1[p_idx]
                th = self.pdm_peak_thetas1[p_idx]
            except:
                continue
            for pdidx in range(len(pdmp)):
                ax21.axvline(pdmp[pdidx],ls='--', c=colors[p_idx], zorder=1)
        ax21.set_title('RAW PDM')
        ax21.set_xlabel('Periods')
        ax21.set_ylabel('Thetas')

        ax22 = fig2.add_subplot(2,1,2)
        ax22.plot(self.pdm_per2, self.thetas2, 'b-')
        for p_idx in range(0,2):
            try :
                pdmp = self.pdm_peaks2[p_idx]
                th = self.pdm_peak_thetas2[p_idx]
            except:
                continue
            for pdidx in range(len(pdmp)):
                ax22.axvline(pdmp[pdidx],ls='--', c=colors[p_idx], zorder=1)
        ax22.set_title('DT PDM')
        ax22.set_xlabel('Periods')
        ax22.set_ylabel('Thetas')
        fig2.subplots_adjust(bottom=0.05, top=0.9, left=0.1, right=0.99, hspace=0.5, wspace=0.25)

        fig3 = plt.figure()
        ax31 = fig3.add_subplot(1,1,1)
        ax31.plot(self.DT,self.MAG, 'ko')
        ax31.plot(self.detrended_dt, self.model_plot, 'g-')
        ax31.set_title('Final Model')
        ax31.set_xlabel('Time [s]')
        ax31.set_ylabel('Magnitude')
        fig3.subplots_adjust(bottom=0.05, top=0.9, left=0.1, right=0.99, hspace=0.5, wspace=0.25)


        if directory:
            pdfpag = PdfPages(os.path.join(directory, self.name + '.pdf'))
            fig_nums = plt.get_fignums()
            figs = [plt.figure(n) for n in fig_nums]
            for figu in figs:
                figu.savefig(pdfpag, format='pdf')
            #fig.savefig(os.path.join(directory, self.name + '.pdf'))
            #fig2.savefig(os.path.join(directory, self.name + '.pdf'))
            #plt.close('all')
            pdfpag.close()
            plt.close('all')
            del fig

        else:

            fig.show()
            fig2.show()
            fig3.show()

    def csv(self, directory=None):

        csv = [
            '-----------------------START METADATA-------------------------',
            self.PARTICIPANT_1,
            'Telescope: {0}'.format(self.telescope),
            'Sensor: {0}'.format(self.camera),
            'Latitude [deg]: {0} '.format(self.LATITUDE),
            'Longitude [deg]: {0}'.format(self.LONGITUDE),
            'Elevation [m]: {0}'.format(self.ALTITUDE),
            'Reduction data procedure: {0}'.format(self.MODE),
            'Uncertainty: {0}'.format('From TDM'),
            'Magnitude type: {0}'.format('From TDM'),
            'AVG_PHOTOMETRIC_RMS: {0}'.format(self.AVG_PHOTOMETRIC_RMS),
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
            '\n'.join(ff for ff in self.comments_2),
            'Periodicity classification comments:\n{0}'.format(
                '\n'.join(['            ' + ff for ff in self.comments])
            ),
            '-----------------------END METADATA----------------------------',
        ]

        if self.ANGLE_TYPE.strip() == 'RADEC':
            csv.append('OBSid,TimeSinceFirstEpoch[sec],FOVra[deg],FOVdec[deg],SigPlate[deg],'
                       'Mag,PhaseAngle,Distance,STMag,Model')
        else:
            csv.append('OBSid,TimeSinceFirstEpoch[sec],FOVaz[deg],FOVel[deg],SigPlate[deg],'
                       'Mag,PhaseAngle,Distance,STMag,Model')

        for entry in range(len(self.JD)):
            csv.append(','.join([
                str(float(entry + 1)),
                str(self.DT[entry]),
                str(self.ANGLE_1[entry]),
                str(self.ANGLE_2[entry]),
                str(self.PLATE_SOLUTION_RMS[entry]),
                str(self.RAW_MAG[entry]),
                #str(self.PHOTOMETRIC_RMS[entry]),
                str(self.phase[entry]),
                str(self.distance[entry]),
                str(self.stmag[entry]),
                str(self.model[entry]),
            ]))

        if directory:

            w = open(os.path.join(directory, self.name + '.csv'), 'w')
            w.write('\n'.join(csv))
            w.close()

        else:

            print('\n'.join(self.comments))


    def history_json(self,directory = None):
        file_name = f'history_{int(self.NORADID)}.json'
        t0_UTC = self.UTC[0]
        new_tracklet = {'trackletID':str(int(self.TRACKLET)),
                        'epoch':t0_UTC,
                        'MaxPowerPeriod':self.period,
                        'MinPDMPeriod':self.pdm_period2,
                        'LongPeriod':self.long_period,
                        'PeriodicityClass':self.physical_class,
                        'AdditionalPeriods':self.additional_periods,
                        'Harmonics':f'[{",".join(str(ff[1]) for ff in self.harmonic_peaks2)}]',
                        'LowPower':f'[{",".join(str(ff[1]) for ff in self.low_power_periods2)}]'}
        #njobj = json.dumps(new_tracklet,indent=9)

        file_exists = os.path.exists(directory+'/'+file_name)
        if file_exists:
            with open(directory+'/'+file_name,'r+') as file:
                file_data = json.load(file)
                file_data['tracklets'].append(new_tracklet)
                file.seek(0)
                json.dump(file_data, file, indent = 4)
            #print('file exists')
        else:
            history = {"NORADID": str(self.NORADID),
                       "tracklets": [new_tracklet]}
            new_hist = json.dumps(history,indent=4)
            with open(directory+'/'+file_name, "w") as outfile:
                outfile.write(new_hist)
            #print('create file')

    def write_txt(self,directory = None):#self
        t0_UTC = self.UTC[0] # ISO Date of first point in tracklet
        file_name = f'history_{int(self.NORADID)}.txt' # create filename
    
        file_exists = os.path.exists(directory+'/'+file_name)
        if file_exists: # check if file exists
    
            #check if tracklet is already in file
            with open(directory+'/'+file_name, 'r', encoding='utf8') as f:
                lines = f.readlines()
    
            found = False
            for i, line in enumerate(lines):
                if line.startswith(f'{int(self.TRACKLET) : <8}|{t0_UTC[0:21] : <22}'): # replace tracklet results
                    lines[i] = f'{int(self.TRACKLET) : <8}|{t0_UTC[0:21] : <22}|{str(self.period) : ^18}|{str(self.pdm_period2) : ^18}|{self.long_period : ^11}|{self.physical_class : <54}|{self.additional_periods}|[{",".join(str(ff[1]) for ff in self.harmonic_peaks2)}]|[{",".join(str(ff[1]) for ff in self.low_power_periods2)}]\n'
                    found = True
                    break

            if not found: #Add tracklet results
                lines.append(f'{int(self.TRACKLET) : <8}|{t0_UTC[0:21] : <22}|{str(self.period) : ^18}|{str(self.pdm_period2) : ^18}|{self.long_period : ^11}|{self.physical_class : <54}|{self.additional_periods}|[{",".join(str(ff[1]) for ff in self.harmonic_peaks2)}]|[{",".join(str(ff[1]) for ff in self.low_power_periods2)}]\n')
                
            lines[2:].sort(key=lambda line: line.split("|")[1])

            with open(directory+'/'+file_name, 'w', encoding='utf8') as f:
                f.writelines(lines)
        else:
        #write a new file
            txt = [
                'Satellite NORADID : {0}'.format(int(self.NORADID)),
            ]
            header = ['Tracklet','Date','Max Power Period','Min PDM Period','Long period','Periodicity class','Additional periods','Harmonics','Low_Power']
            txt.append(f'{header[0] : <8}|{header[1] : ^22}|{header[2] : ^18}|{header[3] : ^18}|{header[4] : ^11}|{header[5] : ^54}|{header[6]}|{header[7]}|{header[8]}')
            txt.append(f'{int(self.TRACKLET) : <8}|{t0_UTC[0:21] : <22}|{str(self.period) : ^18}|{str(self.pdm_period2) : ^18}|{self.long_period : ^11}|{self.physical_class : <54}|{self.additional_periods}|[{",".join(str(ff[1]) for ff in self.harmonic_peaks2)}]|[{",".join(str(ff[1]) for ff in self.low_power_periods2)}]\n')

            if directory:
                w = open(directory+'/'+file_name, 'w')
                w.write('\n'.join(txt))
                w.close() 

    def export(self, directory):

        self.plot(directory)
        self.csv(directory)
        self.write_txt(directory)
        self.history_json(directory)

    def show(self):

        self.plot()
        self.csv()

    def analyse(self,
                period_max=2.0, period_min=0.5, period_step=0.01, fap_limit=0.001, long_period_peak_ratio=0.9,
                cleaning_max_power_ratio=0.2, cleaning_alliase_proximity_ratio=0.2, pdm_bins=20,
                half_window=10, poly_deg=1, limit_to_single_winow=5, single_window_poly_deg=3,
                export=None, show=False):

        #         Calculate disctance, phase and standard magnitude
        try:
            self.distance, self.phase = get_range_phase(
                self.FIRSTLINE, self.SECONDLINE, self.JD,
                self.LATITUDE, self.LONGITUDE, self.ALTITUDE)
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
         self.signals1,self.fake_peaks1) = periodogram(self.DT, self.MAG, period_max=period_max, period_min=period_min,
                                      period_step=period_step, fap_limit=fap_limit,
                                      long_period_peak_ratio=long_period_peak_ratio,
                                      cleaning_max_power_ratio=cleaning_max_power_ratio,
                                      cleaning_alliase_proximity_ratio=cleaning_alliase_proximity_ratio,
                                      pdm_bins=pdm_bins)

        #         Check for false positives
        #self.fake_peaks1 = []
        if len(self.periods1) > 0:
            (self.periods1, self.total_signal1, 
            self.signals1, self.harmonic_peaks1,self.harmonic_signals1
            ,self.low_power_periods1, self.low_power_signals1) = Categorize(self.DT, self.MAG, self.periods1, self.signals1,false_limit = 0.2,
                                      period_max=period_max, period_min=period_min,
                                      period_step=period_step, fap_limit=fap_limit,
                                      long_period_peak_ratio=long_period_peak_ratio,
                                      cleaning_max_power_ratio=cleaning_max_power_ratio,
                                      cleaning_alliase_proximity_ratio=cleaning_alliase_proximity_ratio,
                                      pdm_bins=pdm_bins)
        else:
            self.harmonic_peaks1=[]
            self.low_power_periods1=[]

        self.pdm_per1, self.thetas1, self.pdm_dominant_period1 = full_PDM(self.DT, self.MAG,self.periods1,self.harmonic_peaks1, period_min=0.5, period_step=0.001, pdm_bins = 20)

        if self.pdm_dominant_period1 is not None:
            self.pdm_period1 = self.pdm_dominant_period1[1]
        else:
            '''if len(self.periods1) > 0:
                self.pdm_period1 = self.periods1[0][1] * self.periods1[0][6]
            else:'''
            self.pdm_period1 = None

        #         test of repetitions to clean residuals

        self.residuals_test1 = []

        test_mag = np.ones_like(self.MAG) * self.MAG - self.total_signal1
        for i in range(200):
            (periodogramm, periods, long_period, total_signal, lt_signal,
             signals,_) = periodogram(self.DT, test_mag, period_max=period_max, period_min=period_min,
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

        self.trend, self.detrended_dt, self.detrended_mag, self.trend_type = detrend(self.DT, self.MAG, half_window=half_window,
                                                                    poly_deg=poly_deg,
                                                                    limit_to_single_winow=limit_to_single_winow,
                                                                    single_window_poly_deg=single_window_poly_deg)

        self.trend, self.detrended_dt, self.detrended_mag, self.trend_periodigram = test_trend(self.trend, self.detrended_dt, self.detrended_mag,self.periods1, self.fake_peaks1, self.harmonic_peaks1, self.trend_type, limit = 0.2,
                                                                        period_max=period_max, period_min=period_min,
                                                                        period_step=period_step, fap_limit=0.1,
                                                                        long_period_peak_ratio=long_period_peak_ratio,
                                                                        cleaning_max_power_ratio=0.5,
                                                                        cleaning_alliase_proximity_ratio=cleaning_alliase_proximity_ratio,
                                                                        pdm_bins=pdm_bins)
        

        #        compute periodogramm for detrended lightcurve

        (self.periodogramm2, self.periods2, self.long_period2, self.total_signal2, self.lt_signal2,
         self.signals2, self.fake_peaks2) = periodogram(self.detrended_dt, self.detrended_mag, period_max=period_max,
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
             signals,_) = periodogram(self.detrended_dt, test_mag, period_max=period_max, period_min=period_min,
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
        
        #self.fake_peaks2 = []
        if len(self.periods2)>0:
            (self.periods2, self.total_signal2,
             self.signals2, self.harmonic_peaks2,self.harmonic_signals2,
             self.low_power_periods2, self.low_power_signals2) = Categorize(self.detrended_dt, self.detrended_mag, self.periods2,self.signals2, false_limit = 0.2,
                                      period_max=period_max, period_min=period_min,
                                      period_step=period_step, fap_limit=fap_limit,
                                      long_period_peak_ratio=long_period_peak_ratio,
                                      cleaning_max_power_ratio=cleaning_max_power_ratio,
                                      cleaning_alliase_proximity_ratio=cleaning_alliase_proximity_ratio,
                                      pdm_bins=pdm_bins)
        else:
            self.harmonic_peaks2=[]
            self.harmonic_signals2=[]
            self.low_power_periods2=[]
            self.low_power_signals2=[]
        
        
        self.pdm_per2, self.thetas2, self.pdm_dominant_period2 = full_PDM(self.detrended_dt, self.detrended_mag,self.periods2,self.harmonic_peaks2, period_min=0.5, period_step=0.001, pdm_bins = 20)
        
        if self.pdm_dominant_period2 is not None:
            self.pdm_period2 = self.pdm_dominant_period2[1]
        else:
            '''if len(self.periods2) > 0:
                self.pdm_period2 = self.periods2[0][1] * self.periods2[0][6]
            else:'''
            self.pdm_period2 = None
        #         periodicity classification

        self.period = None
        self.additional_periods = []

        if not self.long_period:
            if len(self.periods1) == 0 and len(self.periods2) == 0:
                if len(self.fake_peaks1) > 0 or len(self.fake_peaks2) > 0 or (max(self.trend) - min(self.trend)) > 2* np.std(self.detrended_mag):
                    self.statistical_class = 1.1
                    self.physical_class = 'Aperiodic variable'
                else:
                    self.statistical_class = 1
                    self.physical_class = 'Non Variable'

            elif len(self.periods1) > 0 and len(self.periods2) == 0:
                self.statistical_class = 3
                self.physical_class = 'Periodic variable'
                self.period = self.periods1[0][1] #* self.periods1[0][6]
                for period in self.periods1[1:]:
                    self.additional_periods.append(period[1]) #* period[6])
            elif len(self.periods1) > 0 and len(self.periods2) > 0:
                self.statistical_class = 5
                self.physical_class = 'Periodic variable'
                self.period = self.periods2[0][1] #* self.periods2[0][6]
                for period in self.periods2[1:]:
                    self.additional_periods.append(period[1]) #* period[6])
            elif len(self.periods1) == 0 and len(self.periods2) > 0:
                self.statistical_class = 7
                self.physical_class = 'Periodic variable'
                self.period = self.periods2[0][1] #* self.periods2[0][6]
                for period in self.periods2[1:]:
                    self.additional_periods.append(period[1]) #* period[6])

        else:
            if len(self.periods1) == 0 and len(self.periods2) == 0:
                self.statistical_class = 2
                self.physical_class = 'Aperiodic variable with possible long-term periodicity'
            elif len(self.periods1) > 0 and len(self.periods2) == 0:
                self.statistical_class = 4
                self.physical_class = 'Periodic variable with possible long-term periodicity'
                self.period = self.periods1[0][1] #* self.periods1[0][6]
                for period in self.periods1[1:]:
                    self.additional_periods.append(period[1]) #* period[6])
            elif len(self.periods1) > 0 and len(self.periods2) > 0:
                self.statistical_class = 6
                self.physical_class = 'Periodic variable with possible long-term periodicity'
                self.period = self.periods2[0][1] #* self.periods2[0][6]
                for period in self.periods2[1:]:
                    self.additional_periods.append(period[1]) #* period[6])
            elif len(self.periods1) == 0 and len(self.periods2) > 0:
                if self.periods2[0][1]/(max(self.detrended_dt) -min(self.detrended_dt) ) > 0.4 and len(self.harmonic_peaks2) < 1:
                    self.statistical_class = 8.2
                    self.physical_class = 'Aperiodic variable with possible long-term periodicity'
                    if len(self.fake_peaks2)>0:
                        self.fake_peaks2 = np.append(self.fake_peaks2, np.array([self.periods2[0]]), axis=0)
                    else:
                        self.fake_peaks2=[]
                        self.fake_peaks2.append(self.periods2[0])
                    self.periods2.pop(0)
                    self.total_signal2 = self.total_signal2 - self.signals2[0]
                    self.period = None
                else:
                    self.statistical_class = 8.1
                    self.physical_class = 'Periodic variable with possible long-term periodicity'
                    self.period = self.periods2[0][1] #* self.periods2[0][6]
                for period in self.periods2[1:]:
                    self.additional_periods.append(period[1]) #* period[6])

        if len(self.additional_periods) == 0:
            self.additional_periods = [None]

	
        self.comments = ['Statistical class: {0}'.format(self.statistical_class)]
        self.comments.append('RAW LC:')
        if len(self.residuals_test1) > 0:
            self.comments.append('    Repetitions to clean residuals: {0}.'.format(len(self.residuals_test1)))
        self.comments.append('    Long Period: {0}'.format(self.long_period))
        if self.long_period:
            self.comments.append('    Possible periodiciy beyond {0} s.'.format(max(self.DT)))
        for period in self.periods1:
            xx = '    Periodogramm peak at {0} s.'.format(period[1])
            if period[6] > 1:
                xx += ' Multiplied by {0} for minimum PDM.'.format(period[6])
            self.comments.append(xx)
        self.comments.append('    PDM dominant peak at {0} s.'.format(self.pdm_period1))
        if len(self.harmonic_peaks1) > 0:
            self.comments.append('    Harmonics peaks: [{0}]'.format(','.join(str(ff[1]) for ff in self.harmonic_peaks1)))
        if len(self.low_power_periods1) > 0:
            self.comments.append('    Low Power peaks : [{0}]'.format(','.join(str(ff[1]) for ff in self.low_power_periods1)))
        self.comments.append('DT LC:')
        self.comments.append('    Long Period: {0}'.format(self.long_period2))
        if self.long_period2:
            self.comments.append('    Possible periodiciy beyond {0} s.'.format(max(self.DT)))
        if len(self.residuals_test2) > 0:
            self.comments.append('    Repetitions to clean residuals: {0}.'.format(len(self.residuals_test2)))
        for period in self.periods2:
            xx = '    Periodogramm peak at {0} s.'.format(period[1])
            if period[6] > 1:
                xx += ' Multiplied by {0} for minimum PDM.'.format(period[6])
            self.comments.append(xx)
        self.comments.append('    PDM dominant peak at {0} s.'.format(self.pdm_period2))
        if len(self.harmonic_peaks2) > 0:
            self.comments.append('    Harmonics peaks: [{0}]'.format(','.join(str(ff[1]) for ff in self.harmonic_peaks2)))
        if len(self.low_power_periods2) > 0:
            self.comments.append('    Low Power peaks : [{0}]'.format(','.join(str(ff[1]) for ff in self.low_power_periods2)))
        
        self.all_periods = []
        self.dom_periods = []
        for peri in self.periods2:
            self.all_periods.append(peri)
            self.dom_periods.append(peri)
        for peri in self.harmonic_peaks2:
            self.all_periods.append(peri)
            self.dom_periods.append(peri)
        #for peri in self.low_power_periods2:
        #    self.all_periods.append(peri)
        

        self.harmonic_class = []
        ratios = [2,3,4,5,6,7,8,3/2,5/2,7/2,Fraction(4,3),Fraction(5,3),Fraction(7,3),Fraction(8,3)]
        if len(self.harmonic_peaks2) > 0:
            for peridx in range(len(self.periods2)):
                for harm in self.harmonic_peaks2:
                    for ratio in ratios:
                        if np.abs(max(self.periods2[peridx][1],harm[1])/min(self.periods2[peridx][1],harm[1]) - float(ratio)) < cleaning_alliase_proximity_ratio+(float(ratio)-1)*0.03:
                            self.harmonic_class.append([harm[1],self.periods2[peridx][1],Fraction(ratio),peridx])

        self.comments_2 = ['Present Periods[s]','Periodicity class: {0}'.format(self.physical_class)]
        if len(self.all_periods) > 0:
            self.comments_2.append('[Power,Period[s],Amplitude,Phase[rad]] :\n{0}'.format('\n'.join(f'[{ff[0]}, {ff[1]}, {ff[2]}, {ff[3]}]' for ff in sorted(self.all_periods, key=lambda x :-x[0]))))
            self.comments_2.append('Maximu Power Period[s]: {0}'.format(sorted(self.all_periods, key=lambda x :-x[0])[0][1]))
            self.comments_2.append('PDM minimum  Period[s] : {0}'.format(self.pdm_period2))
            self.comments_2.append('Longest Period[s] :{0}'.format(sorted(self.dom_periods, key=lambda x :-x[1])[0][1]))
            self.comments_2.append('Harmonics to max power :')
            if len(self.harmonic_peaks2) > 0:
                self.comments_2.append('[Period[s],max/min ratio]\n{0}'.format('\n'.join((f'{ff[0]} , {ff[2]}') for ff in self.harmonic_class if int(ff[3])==0)))
            else:
                self.comments_2.append('None'),

            if len(self.additional_periods) > 0:
                self.comments_2.append('Additional periods [s]: {0}'.format(','.join([str(ff) for ff in self.additional_periods])))
                self.comments_2.append('Harmonics to additional')
                self.comments_2.append('[Harmonic[s],Period[s],max/min ratio]\n{0}'.format(','.join((f'{ff[0]} , {ff[1]} -> {ff[2]}') for ff in self.harmonic_class if int(ff[3])!=0)))

            if len(self.low_power_periods2) > 0:
                self.comments_2.append('Low Power Periods [s]: {0}'.format(','.join([str(ff[1]) for ff in self.low_power_periods2])))
        
        else:
            self.comments_2.append('None')
        self.model, self.model_plot = build_model(self.MAG, self.trend, self.signals2, self.harmonic_signals2, self.low_power_signals2)

        #         export or show

        if export:
            self.export(export)

        if show:
            self.show()
