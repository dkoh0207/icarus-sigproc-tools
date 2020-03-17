import numpy as np
import ROOT
from sproc import sproc

import scipy.signal as sig


class Deconvolution:
    '''
    Wrapper class for deconvolution methods.
    '''
    def __init__(self, waveLessNoise, response_fn=None, noise_var=None):
        self.waveLessNoise = waveLessNoise
        self.numChannels, self.nTicks = waveLessNoise.shape
        self.response_fn = ROOT.std.vector('float')(self.nTicks)
        for i in range(self.nTicks):
            self.response_fn[i] = response_fn[i]
        self.deconvolver = ROOT.sigproc_tools.Deconvolution()
        if noise_var is not None:
            self.noise_var = noise_var

    def deconvolve1D(self, name='wiener', **kwargs):
        '''
        Deconvolution algorithms using 1D FFT/IFFT on each channels.
        '''
        sx = kwargs.get('sx', 3)
        sy = kwargs.get('sy', 3)
        output = ROOT.std.vector('std::vector<float>')(
            self.numChannels, ROOT.std.vector('float')(self.nTicks))
        input = sproc.pyutil.as_float32_vector_2d(
            self.waveLessNoise)
        if name == 'wiener':
            self.deconvolver.Wiener1D(
                output, input, self.response_fn, self.noise_var)
        elif name == 'inverse':
            self.deconvolver.Inverse1D(
                output, input, self.response_fn)
        elif name == 'adaptive_wiener':
            self.deconvolver.AdaptiveWiener1D(
                output, input, self.response_fn, selectVals)
        else:
            raise ValueError(
                'Deconvolution module <{}> not implemented'.format(name))
        self.deconvWaveform = np.asarray(output)
        return self.deconvWaveform


    def fourier_shrinkage_1d(self, regParam=0.001):
        output = ROOT.std.vector('std::vector<float>')(
            self.numChannels, ROOT.std.vector('float')(self.nTicks))
        input = sproc.pyutil.as_float32_vector_2d(
            self.waveLessNoise)

        self.deconvolver.FourierShrinkage1D(
            output, input, self.response_fn, regParam)

        self.deconvWaveform = np.asarray(output)
        return self.deconvWaveform

    def deconvolve2D(self):
        raise NotImplementedError

    @staticmethod
    def compute_charge(roi, waveform, electronicsGain=67.4):
        return np.sum(waveform[roi]) * electronicsGain

class WaveletTransform:

    def __init__(self, waveLessCoherent, response_fn=None, noise_var=None):
        self.waveLessCoherent = waveLessCoherent
        self.numChannels, self.nTicks = wavelessCoherent.shape
        
    def wt2(self, isign=1, levels=2):
        wf2d = sproc.pyutil.as_float32_vector_2d(
            self.waveLessCoherent)
        self.output = ROOT.std.vector('std::vector<float>')(
            self.numChannels, ROOT.std.vector('float')(self.nTicks))
        wav = ROOT.sigproc_tools.Daubechies4()
        transform = ROOT.sigproc_tools.WaveletTransform()
        transform.wt2(self.wf2d, self.output, 
            self.numChannels, self.nTicks, isign, wav, levels)
        self.output = np.asarray(self.output)