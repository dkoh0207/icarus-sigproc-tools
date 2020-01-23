import ROOT
import uproot
import numpy as np
import math
import sys
import os

from sigproc_tools.sigproc_objects.filterevents import FilterEvents
sigProcPath = "/u/nu/koh0207/icarus-sigproc-tools"
sys.path.insert(0,sigProcPath)
from sigproc_tools.sigproc_objects.fullresponse import FullResponse
from sigproc_tools.sigproc_objects.fieldresponse import FieldResponse
from sigproc_tools.sigproc_objects.electronicsresponse import ElectronicsResponse
from sigproc_tools.sigproc_functions.fakeParticle import \
    genWhiteNoiseWaveform,genSpikeWaveform,createParticleTrajectory

# C++ Noise Removal Modules
from sigproc_tools.sigproc_utils.denoising import Denoiser
from sproc import sproc


def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))


class TPCResponseData:

    def __init__(self, cfg):
        # PATH TO DATA
        self.pathName = cfg['pathName']
        self.recoFileName = cfg['pathName'] + cfg['recoFileName']
        print(self.recoFileName)
        self.recoFolderName = cfg['recoFolderName']
        self.daqName = cfg['daqName']
        print("Opening file: ", self.recoFileName)
        self.data_file = uproot.open(self.recoFileName)
        print("Opening the folder contianing the RawDigits information: ",self.recoFolderName)
        self.events_folder = self.data_file[self.recoFolderName]
        # Go ahead and filter the events
        fEvents = FilterEvents(self.events_folder,self.daqName)

        # What are the basic operating parameters?
        self.nChannelsPerGroup = cfg.get('nChannelsPerGroup', 64)
        self.numChannels       = int(fEvents.rawdigits.numChannels(0))
        self.numTicks = cfg.get('numTicks', 4096)
        self.numGroups         = self.numChannels // self.nChannelsPerGroup
        self.numEvents = fEvents.filterEvents(self.nChannelsPerGroup)
        print("Noise processing complete with",self.numEvents)
        self.waveLessPed = fEvents.waveLessPedAll[0,:,:]

        self.inputFilePath = cfg.get('inputFilePath', "../data/")
        self.tpcNumber = cfg.get('tpcNumber', 1)
        self.tpcResponse2 = FullResponse(self.inputFilePath,"t600_response_vw02_v0.0.root")
        if self.tpcNumber == 0:
            self.tpcResponse = FullResponse(self.inputFilePath,"t600_response_vw00_v0.0.root")
        elif self.tpcNumber == 1:
            self.tpcResponse = FullResponse(self.inputFilePath,"t600_response_vw01_v0.0.root",\
                normalization=self.tpcResponse2.FieldResponse.normFactor)
        elif self.tpcNumber == 2:
            self.tpcResponse = FullResponse(self.inputFilePath,"t600_response_vw02_v0.0.root",\
                normalization=self.tpcResponse2.FieldResponse.normFactor)
        else:
            raise ValueError('Invalid TPC Number: {}'.format(self.tpcNumber))


    def genTrack(self, **kwargs):
        '''
        Generates Toy Track with given angle and number of electrons.

        RETURNS:
            - fullEvent: Pedestal Subtracted Waveform overlayed with fake particle,
            to be used in coherent noise removal. 
            - trueMask: ground truth mask for signal protection.
        '''
        numElectrons = kwargs.get('numElectrons', 6000)
        print("NE = ", numElectrons)
        angle = kwargs.get('angleToWire', 45)
        print("Angle = ", angle)
        slope = math.tan(math.radians(90-angle)) / 0.213
        startTick = kwargs.get('startTick', 2000)
        startWire = kwargs.get('startWire', 128)
        stopWire = kwargs.get('stopWire', 448)
        roi_threshold = kwargs.get('roi_threshold', 0.9)

        print("Angle to wire:",angle,"(deg), tan(theta):",\
            math.tan(math.radians(90-angle)),", slope:",slope)
        wireRange = (startWire,stopWire)
        tickRange = (startTick,min(int(round(\
            slope*(wireRange[1]-wireRange[0])+startTick)), self.numTicks))
        print("Using wireRange:",wireRange,", tickRange:",tickRange)
        spikeResponse, spikeInput = createParticleTrajectory(self.tpcResponse,numElectrons,\
            wireRange,tickRange,(self.numChannels,self.numTicks))
        fullEvent = spikeResponse + self.waveLessPed
        print("Overlay event created, wires from:",wireRange,", ticks from:",tickRange)
        trueMask = abs(spikeResponse) > roi_threshold
        self.fullEvent = fullEvent
        self.denoiser = Denoiser(fullEvent)
        self.trueMask = trueMask.astype(bool)


    def denoise(self, kernel_size=5, window_size=10, filter_name='d', 
                structuringElement=(7, 20), tFactor=6.0):
        '''
        Remove Coherent Noise via Morphological Filters
        '''
        self.denoiser.removeCoherentNoise2D(
            filter_name, structuringElement=structuringElement, 
            window=window_size, threshold=tFactor)
        self.denoisedWaveform = self.denoiser.waveLessCoherent
        self.predMask = self.denoiser.roi.astype(bool)
        self.intrinsicRMS = self.denoiser.intrinsicRMS
        self.correctedMedians = self.denoiser.correctedMedians


    def fit_gaussian_to_rms(self, params=[1., 1., 1.]):

        nBins  = 30.
        lowBin = 0.1
        hiBin  = 0.5
        stepSize = (hiBin-lowBin)/nBins
        xBins = np.arange(lowBin,hiBin,stepSize)

        hist,_ = np.histogram(intrinsicRMS[0],bins=xBins.size,range=(xBins[0],xBins[-1]))
        fitArray = hist.astype(np.float32)
        #max bin?
        maxValue = np.amax(hist)
        maxBin   = np.where(hist == maxValue)
        maxRMS   = xBins[maxBin]

        fitParams = np.array([50.,maxRMS,0.15]).astype(np.float32)
        coeff,varMatrix = curve_fit(gauss,xBins.astype(np.float32),fitArray,p0=fitParams)


    def run_analysis(self):

        # Signal-to-Noise
        true_signal = self.denoisedWaveform * self.trueMask.astype(int)
        trueAmp = abs(true_signal[self.trueMask]).mean()
        trueAmpStdMean = abs(true_signal[self.trueMask]).std() / (np.sum(self.trueMask)-1)
        predicted_signal = self.denoisedWaveform * self.predMask.astype(int)
        predictedAmp = np.abs(predicted_signal[self.predMask]).mean()
        predAmpStdMean = np.abs(predicted_signal[self.predMask]).std() / (np.sum(self.predMask)-1)
        medianRMS = np.median(self.intrinsicRMS)
        print("Median RMS = ", medianRMS)
        stdRMS = 0.
        trueSNR = (trueAmp / medianRMS)**2
        predSNR = (predictedAmp / medianRMS)**2
        trueSNRErr = np.sqrt((2 * trueAmp / medianRMS * trueAmpStdMean)**2 + (2 * trueAmp**2 / medianRMS**3 * stdRMS)**2)
        predSNRErr = np.sqrt((2 * predictedAmp / medianRMS * predAmpStdMean)**2 + (2 * predictedAmp**2 / medianRMS**3 * stdRMS)**2)
        # ROI Purity and Efficiency
        purity = np.logical_and(self.trueMask, self.predMask).sum() / np.sum(self.trueMask)
        efficiency = np.logical_and(self.trueMask, self.predMask).sum() / np.sum(self.predMask)
        # Cross-Correlation
        interGroupCCMeanBefore = np.mean(np.corrcoef(self.correctedMedians)[~np.eye(9, dtype=bool)])
        interGroupCCStdBefore = np.std(np.corrcoef(self.correctedMedians)[~np.eye(9, dtype=bool)])
        interGroupCounts = np.sum(~np.eye(9, dtype=bool))
        interGroupStdMeanBefore = interGroupCCStdBefore / np.sqrt(interGroupCounts)
        # Compute per group medians after coherent noise removal
        noiseRemovedMedians = np.zeros((self.numGroups, self.numTicks))
        for i in range(0,self.numGroups):
            median = np.median(self.denoisedWaveform[i * self.nChannelsPerGroup:(i+1) * self.nChannelsPerGroup, :], axis=0)
            noiseRemovedMedians[i, :] = median
        interGroupCCMeanAfter = np.mean(np.corrcoef(noiseRemovedMedians)[~np.eye(9, dtype=bool)])
        interGroupCCStdAfter = np.std(np.corrcoef(noiseRemovedMedians)[~np.eye(9, dtype=bool)])
        interGroupCounts = np.sum(~np.eye(9, dtype=bool))
        interGroupStdMeanAfter = interGroupCCStdAfter / np.sqrt(interGroupCounts)

        noSignalChannelsBefore = np.corrcoef(np.concatenate([self.fullEvent[0:64],
                                        self.fullEvent[64:128],
                                        self.fullEvent[448:512],
                                        self.fullEvent[512:576]], axis=0))
        noSignalChannelsAfter = np.corrcoef(np.concatenate([self.denoisedWaveform[0:64],
                                        self.denoisedWaveform[64:128],
                                        self.denoisedWaveform[448:512],
                                        self.denoisedWaveform[512:576]], axis=0))
        nonDiagonal = ~np.eye(64*4, dtype=bool)
        intraGroupCCMeanBefore = noSignalChannelsBefore[nonDiagonal].mean()
        intraGroupCCStdBefore = noSignalChannelsBefore[nonDiagonal].std()
        intraGroupCCStdMeanBefore = intraGroupCCStdBefore / np.sqrt(len(noSignalChannelsBefore))
        intraGroupCCMeanAfter = noSignalChannelsAfter[nonDiagonal].mean()
        intraGroupCCStdAfter = noSignalChannelsAfter[nonDiagonal].std()
        intraGroupCCStdMeanAfter = intraGroupCCStdAfter / np.sqrt(len(noSignalChannelsAfter))

        res = {
            "trueSNR": trueSNR,
            "trueSNRErr": trueSNRErr,
            "predSNR": predSNR,
            "predSNRErr": predSNRErr,
            "purity": purity,
            "efficiency": efficiency,
            "interGroupCCMeanBefore": interGroupCCMeanBefore,
            "interGroupCCStdBefore": interGroupCCStdBefore,
            "interGroupStdMeanBefore": interGroupStdMeanBefore,
            "interGroupCCMeanAfter": interGroupCCMeanAfter,
            "interGroupCCStdAfter": interGroupCCStdAfter,
            "interGroupStdMeanAfter": interGroupStdMeanAfter,
            "intraGroupCCMeanBefore": intraGroupCCMeanBefore,
            "intraGroupCCStdBefore": intraGroupCCStdBefore,
            "intraGroupCCStdMeanBefore": intraGroupCCStdMeanBefore,
            "intraGroupCCMeanAfter": intraGroupCCMeanAfter,
            "intraGroupCCStdAfter": intraGroupCCStdAfter,
            "intraGroupCCStdMeanAfter": intraGroupCCStdMeanAfter
        }

        return res

