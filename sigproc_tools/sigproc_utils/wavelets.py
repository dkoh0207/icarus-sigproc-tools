import numpy as np
import ROOT
from sproc import sproc


class WaveletTransform:

    def __init__(self, wf, wavelet='daub4', grouping=64):
        self.wf = wf
        self.numChannels, self.nTicks = wf.shape
        self.numGroups = self.numChannels // grouping

        if wavelet == 'daub4':
            self.wavelet = ROOT.sigproc_tools.Daubechies4()
        self.tf = ROOT.sigproc_tools.WaveletTransform()


    def wt2(self, levelChannels=4, levelTicks=4):

        self.waveletCoeffs = []
        for i in range(self.numGroups):
            sample = self.wf[i*64:64*(i+1)]
            sample_vec = sproc.pyutil.as_float32_vector_2d(sample)
            a, b = sample.shape
            output =  ROOT.std.vector('std::vector<float>')(
                a, ROOT.std.vector('float')(b))

            self.tf.wt2(sample_vec, output, a, b, 1, 
                self.wavelet, levelChannels, levelTicks)
            # output = np.array(output)
            self.waveletCoeffs.append(output)
        self.a = a
        self.b = b


    def iwt2(self, levelChannels=4, levelTicks=4):

        self.reco = np.zeros(self.wf.shape)
        for i, wc in enumerate(self.waveletCoeffs):
            # coeffs = sproc.pyutil.as_float32_vector_2d(wc)
            reco =  ROOT.std.vector('std::vector<float>')(
                self.a, ROOT.std.vector('float')(self.b))
            self.tf.wt2(wc, reco, self.a, self.b, -1, 
                self.wavelet, levelChannels, levelTicks)
            self.reco[i*64:(i+1)*64] = np.array(reco)

    def visushrink(self):

        threshold = self.noiseEstimates[0][0] * \
            np.log2(self.numChannels * self.nTicks)

        
    
    def estimateNoise(self, levels=4):
        '''
        Estimates noise via median absolute value of wavelet detail coefficients
        '''
        self._noiseEstimates = []
        for i, wc in enumerate(self.waveletCoeffs):
            noiseEstimates = ROOT.std.vector('float')()
            self.tf.estimateLevelNoise(wc, levels, noiseEstimates)
            self._noiseEstimates.append(noiseEstimates)

    @property
    def noiseEstimates(self):
        return np.array(self._noiseEstimates)


    def wwf(self, levels=4):
        '''
        Wavelet Wiener Shrinkage
        '''
        for i, wc in enumerate(self.waveletCoeffs):
            noiseEstimates = self._noiseEstimates[i]
            self.tf.WaveletWienerShrink(wc, levels, noiseEstimates)