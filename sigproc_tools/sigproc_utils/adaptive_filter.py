import numpy as np
import ROOT
from sproc import sproc

import scipy.signal as sig


class AdaptiveFilter:
    '''
    Class for interfacing deconvolution C++ backend modules.
    '''
    def __init__(self, waveLessCoherent, selectVals=None):
        self.waveLessCoherent = waveLessCoherent
        self.numChannels = waveLessCoherent.shape[0]
        self.nTicks = waveLessCoherent.shape[1]
        if selectVals is None:
            selectVals = ROOT.std.vector('std::vector<bool>')(
                self.numChannels, ROOT.std.vector('bool')(self.numTicks))
        self.deconvolver = ROOT.sigproc_tools.AdaptiveWiener()
        utils = ROOT.sigproc_tools.MiscUtils()
        wLC_float2d = sproc.pyutil.as_float32_vector_2d(
            self.waveLessCoherent)
        self.selectVals = sproc.pyutil.as_bool_vector_2d(selectVals)
        self.noise_var = utils.estimateNoiseVariance(
            wLC_float2d, self.selectVals)

    def denoise_1d(self):
        raise NotImplementedError

    def denoise_2d(self, name='lee', structuringElement=(7,7),
                      noise_var=None, **kwargs):
        '''
        Run 2D Deconvolution algorithms for denoising.

        Note that adaptive local wiener filters are not strictly speaking
        deconvolution algorithms (since they do not involve FFT/IFFT), but
        a local approximation to the Wiener filter which is MMSE optimal.

        INPUTS:
            - name: name of the algorithm to be used. Default is Lee Filter.

            Supported Algorithms:
                - lee: adaptive wiener filter developed by Lee et. al. See:
                    https://ieeexplore.ieee.org/document/4766994
                - lee_median: lee filter except using median instead of mean.
                - kuan: lee filter with adaptive weighting, see:
                    https://ieeexplore.ieee.org/document/4767641

        '''
        if noise_var is None:
            noise_var = self.noise_var
        sx = structuringElement[0]
        sy = structuringElement[1]

        a = kwargs.get('a', 1.0)
        epsilon = kwargs.get('epsilon', 2.5)
        K = kwargs.get('K', 5)
        sigmaFactor = kwargs.get('sigmaFactor', 3.0)

        output = ROOT.std.vector('std::vector<float>')(
            self.numChannels, ROOT.std.vector('float')(self.nTicks))
        input = sproc.pyutil.as_float32_vector_2d(
            self.waveLessCoherent)
        if name == 'lee':
            self.deconvolver.filterLee(
                output, input, noise_var, sx, sy)
        elif name == 'MMWF':
            self.deconvolver.MMWF(
                output, input, noise_var, sx, sy)
        elif name == 'lee_enhanced':
            self.deconvolver.filterLeeEnhanced(
                output, input,
                noise_var, sx, sy, a, epsilon)
        elif name == 'MMWFStar':
            self.deconvolver.MMWFStar(
                output, input, sx, sy)
        elif name == 'lee_enhanced_roi':
            self.deconvolver.adaptiveROIWiener(
                output, input, self.selectVals,
                noise_var, sx, sy, a, epsilon),
        elif name == 'sigmaFilter':
            self.deconvolver.sigmaFilter(
                output, input, noise_var, sx, sy, K, sigmaFactor)
        else:
            raise ValueError('Provided Denoising algorithm name not available.')
        self.result = np.asarray(output)
        return self.result
