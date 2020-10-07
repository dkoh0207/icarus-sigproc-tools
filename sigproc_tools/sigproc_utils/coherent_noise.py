import numpy as np
import ROOT
from sproc import sproc

class Denoiser:
    '''
    Python class to interface C++ backend denoising modules conveniently.

    USAGE:

        denoiser = Denoiser(fullEvent)
        # Find ROI and remove coherent noise
        denoiser.removeCoherentNoise2D('d', (7, 20), window=10, threshold=7.5)
    '''
    def __init__(self, fullEvent, **kwargs):

        self.fullEvent = fullEvent
        self.denoiser = ROOT.sigproc_tools.MorphologicalCNC()
        self.numChannels = kwargs.get('numChannels', 576)
        self.numTicks = kwargs.get('numTicks', 4096)
        self.numGroups = kwargs.get('numGroups', 64)

    def denoiseMorph2D(self, filter_name='d',
                            structuringElement=(7,20),
                            window=10,
                            threshold=7.5):
        '''
        Coherent Noise Removal with 2D Morphological Filters
        '''
        selectVals =  ROOT.std.vector('std::vector<bool>')(
            self.numChannels, ROOT.std.vector('bool')(self.numTicks))
        roi = ROOT.std.vector('std::vector<bool>')(
            self.numChannels, ROOT.std.vector('bool')(self.numTicks))
        fullEvent = sproc.pyutil.as_float32_vector_2d(
            self.fullEvent.astype(np.float32))
        waveLessCoherent =  ROOT.std.vector('std::vector<float>')(
            self.numChannels, ROOT.std.vector('float')(self.numTicks))
        morphedWaveforms =  ROOT.std.vector('std::vector<float>')(
            self.numChannels, ROOT.std.vector('float')(self.numTicks))

        # Run 2D ROI finding and denoising.
        self.denoiser.denoiseMorph2D(
            waveLessCoherent, morphedWaveforms, fullEvent,
            selectVals, roi, filter_name, self.numGroups,
            structuringElement[0], structuringElement[1], window, threshold)

        # Coherent Noise Removed 2D Waveform
        self.waveLessCoherent = np.asarray(waveLessCoherent).astype(np.float32)
        self.morphedWaveforms = np.asarray(morphedWaveforms).astype(np.float32)
        # ROI (Region of Interest)
        self.roi = sproc.pyutil.as_ndarray(roi).astype(bool)
        # Region to protect signal from coherent noise removal. Note that
        # in general self.roi != self.selectVals, since it may be desirable to
        # be conservative with the ROIs.
        self.selectVals = sproc.pyutil.as_ndarray(selectVals).astype(bool)

    def denoiseHough2D(self, filter_name='d', 
                       structuringElement=(7, 20),
                       window=10,
                       threshold=5.0,
                       thetaSteps=360,
                       houghThreshold=300,
                       nmsWindowSize=10,
                       angleWindow=50,
                       dilation=(7,20),
                       maxLines=20,
                       eps=0.00001):

        selectVals =  ROOT.std.vector('std::vector<bool>')(
            self.numChannels, ROOT.std.vector('bool')(self.numTicks))
        refinedSelectVals =  ROOT.std.vector('std::vector<bool>')(
            self.numChannels, ROOT.std.vector('bool')(self.numTicks))
        roi = ROOT.std.vector('std::vector<bool>')(
            self.numChannels, ROOT.std.vector('bool')(self.numTicks))
        fullEvent = sproc.pyutil.as_float32_vector_2d(
            self.fullEvent.astype(np.float32))
        waveLessCoherent =  ROOT.std.vector('std::vector<float>')(
            self.numChannels, ROOT.std.vector('float')(self.numTicks))
        morphedWaveforms =  ROOT.std.vector('std::vector<float>')(
            self.numChannels, ROOT.std.vector('float')(self.numTicks))

        # Run 2D ROI finding and denoising.
        self.denoiser.denoiseHough2D(
            waveLessCoherent, morphedWaveforms, fullEvent,
            selectVals, refinedSelectVals, roi, filter_name, self.numGroups,
            structuringElement[0], structuringElement[1], window, threshold,
            thetaSteps, houghThreshold, nmsWindowSize, angleWindow, dilation[0], dilation[1], maxLines, eps)

        # Coherent Noise Removed 2D Waveform
        self.waveLessCoherent = np.asarray(waveLessCoherent).astype(np.float32)
        self.morphedWaveforms = np.asarray(morphedWaveforms).astype(np.float32)
        # ROI (Region of Interest)
        self.roi = sproc.pyutil.as_ndarray(roi).astype(bool)
        # Region to protect signal from coherent noise removal. Note that
        # in general self.roi != self.selectVals, since it may be desirable to
        # be conservative with the ROIs.
        self.selectVals = sproc.pyutil.as_ndarray(selectVals).astype(bool)
        self.refinedSelectVals = sproc.pyutil.as_ndarray(refinedSelectVals).astype(bool)
