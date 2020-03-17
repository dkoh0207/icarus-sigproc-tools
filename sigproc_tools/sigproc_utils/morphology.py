import numpy as np
import ROOT
from sproc import sproc

#######################################################################
#
#   1D and 2D Morphological Filtering Modules with C++ backend.
#
#   2019. 12. 16    Dae Heun Koh
#
#######################################################################

# 1D Filters
INF = 99999

class MorphologyBase:

    def __init__(self, waveform):
        if isinstance(waveform, np.ndarray):
            self.waveform = waveform.astype(np.float32)
            self.waveform_stdvec = sproc.pyutil.as_float32_vector_2d(self.waveform)
        elif isinstance(waveform, ROOT.std.vector):
            self.waveform = np.asarray(waveform).astype(np.float32)
            self.waveform_stdvec = waveform
        else:
            raise ValueError('Input waveform must be an instance of numpy \
            array or C++ std::vector')


class Morphology1D(MorphologyBase):
    '''
    Wrapper class for 1D Morphological Operations.

    For faster implementation, see Morphology1DFast.
    '''
    def __init__(self, waveform):
        super(Morphology1D, self).__init__(waveform)
        self.morph1d = ROOT.sigproc_tools.Morph1D()

    def dilation(self, se=10):
        '''
        INPUTS:
            - input (np.float32 array): input 2D array to compute dilation.
            If None, uses the stored 2D array (instance attribute).
            - se (int tuple): structuring element; determines the size of
            moving window.

        RETURNS:
            - dilation1D (np.array): 1D dilation filter applied output.
        '''
        numChannels, numTicks = self.waveform.shape
        out = ROOT.std.vector('std::vector<float>')(
            numChannels, ROOT.std.vector('float')(numTicks))
        self.morph1d.getDilation(self.waveform_stdvec, se, out)
        out = np.asarray(out).astype(np.float32)
        return out

    def erosion(self, se=10):
        '''
        INPUTS:
            - input (np.float32 array): input 2D array to compute dilation.
            - se (int tuple): structuring element; determines the size of
            moving window.

        RETURNS:
            - dilation2D (np.array): 1D dilation filter applied output.
        '''
        numChannels, numTicks = self.waveform.shape
        out = ROOT.std.vector('std::vector<float>')(
            numChannels, ROOT.std.vector('float')(numTicks))
        self.morph1d.getErosion(self.waveform_stdvec, se, out)
        out = np.asarray(out).astype(np.float32)
        return out

    def gradient(self, se=10):
        '''
        INPUTS:
            - input (np.float32 array): input 2D array to compute dilation.
            - se (int tuple): structuring element; determines the size of
            moving window.

        RETURNS:
            - dilation2D (np.array): 1D dilation filter applied output.
        '''
        numChannels, numTicks = self.waveform.shape
        out = ROOT.std.vector('std::vector<float>')(
            numChannels, ROOT.std.vector('float')(numTicks))
        self.morph1d.getGradient(self.waveform_stdvec, se, out)
        out = np.asarray(out).astype(np.float32)
        return out

    def median(self, se=10):
        numChannels, numTicks = self.waveform.shape
        out = ROOT.std.vector('std::vector<float>')(
            numChannels, ROOT.std.vector('float')(numTicks))
        self.morph1d.getMedian(self.waveform_stdvec, se, out)
        out = np.asarray(out).astype(np.float32)
        return out


class Morphology2D(MorphologyBase):

    def __init__(self, waveform):
        super(Morphology2D, self).__init__(waveform)
        self.morph2d = ROOT.sigproc_tools.Morph2D()

    def dilation(self, se=(7, 20)):
        '''
        INPUTS:
            - input (np.float32 array): input 2D array to compute dilation.
            If None, uses the stored 2D array (instance attribute).
            - se (int tuple): structuring element; determines the size of
            moving window.

        RETURNS:
            - dilation1D (np.array): 1D dilation filter applied output.
        '''
        numChannels, numTicks = self.waveform.shape
        out = ROOT.std.vector('std::vector<float>')(
            numChannels, ROOT.std.vector('float')(numTicks))
        self.morph2d.getDilation(self.waveform_stdvec, se[0], se[1], out)
        out = np.asarray(out).astype(np.float32)
        return out

    def erosion(self, se=(7, 20)):
        '''
        INPUTS:
            - input (np.float32 array): input 2D array to compute dilation.
            - se (int tuple): structuring element; determines the size of
            moving window.

        RETURNS:
            - dilation2D (np.array): 1D dilation filter applied output.
        '''
        numChannels, numTicks = self.waveform.shape
        out = ROOT.std.vector('std::vector<float>')(
            numChannels, ROOT.std.vector('float')(numTicks))
        self.morph2d.getErosion(self.waveform_stdvec, se[0], se[1], out)
        out = np.asarray(out).astype(np.float32)
        return out

    def gradient(self, se=(7, 20)):
        '''
        INPUTS:
            - input (np.float32 array): input 2D array to compute dilation.
            - se (int tuple): structuring element; determines the size of
            moving window.

        RETURNS:
            - dilation2D (np.array): 1D dilation filter applied output.
        '''
        numChannels, numTicks = self.waveform.shape
        out = ROOT.std.vector('std::vector<float>')(
            numChannels, ROOT.std.vector('float')(numTicks))
        self.morph2d.getGradient(self.waveform_stdvec, se[0], se[1], out)
        out = np.asarray(out).astype(np.float32)
        return out

    def median(self, se=(7, 20)):
        numChannels, numTicks = self.waveform.shape
        out = ROOT.std.vector('std::vector<float>')(
            numChannels, ROOT.std.vector('float')(numTicks))
        self.morph2d.getMedian(self.waveform_stdvec, se[0], se[1], out)
        out = np.asarray(out).astype(np.float32)
        return out

class Morphology2DFast(MorphologyBase):

    def __init__(self, waveform):
        super(Morphology2DFast, self).__init__(waveform)
        self.morph2d = ROOT.sigproc_tools.Morph2DFast()

    def dilation(self, se=(7, 20)):
        '''
        INPUTS:
            - input (np.float32 array): input 2D array to compute dilation.
            If None, uses the stored 2D array (instance attribute).
            - se (int tuple): structuring element; determines the size of
            moving window.

        RETURNS:
            - dilation1D (np.array): 1D dilation filter applied output.
        '''
        numChannels, numTicks = self.waveform.shape
        pad_x = se[0] - (numChannels % se[0])
        pad_y = se[1] - (numTicks % se[1])
        paddedWaveform = np.pad(
            self.waveform, [(0, pad_x), (0, pad_y)], constant_values=-INF)
        numChannelsPadded, numTicksPadded = paddedWaveform.shape
        paddedWaveform = sproc.pyutil.as_float32_vector_2d(paddedWaveform)
        out = ROOT.std.vector('std::vector<float>')(
            numChannelsPadded, ROOT.std.vector('float')(numTicksPadded))
        self.morph2d.getDilation(paddedWaveform, se[0], se[1], out)
        out = np.asarray(out).astype(np.float32)
        out = out[:numChannelsPadded-pad_x, :numTicksPadded-pad_y]
        return out

    def erosion(self, se=(7, 20)):
        '''
        INPUTS:
            - input (np.float32 array): input 2D array to compute dilation.
            - se (int tuple): structuring element; determines the size of
            moving window.

        RETURNS:
            - dilation2D (np.array): 1D dilation filter applied output.
        '''
        numChannels, numTicks = self.waveform.shape
        pad_x = se[0] - (numChannels % se[0])
        pad_y = se[1] - (numTicks % se[1])
        paddedWaveform = np.pad(
            self.waveform, [(0, pad_x), (0, pad_y)], constant_values=INF)
        numChannelsPadded, numTicksPadded = paddedWaveform.shape
        paddedWaveform = sproc.pyutil.as_float32_vector_2d(paddedWaveform)
        out = ROOT.std.vector('std::vector<float>')(
            numChannelsPadded, ROOT.std.vector('float')(numTicksPadded))
        self.morph2d.getErosion(paddedWaveform, se[0], se[1], out)
        out = np.asarray(out).astype(np.float32)
        out = out[:numChannelsPadded-pad_x, :numTicksPadded-pad_y]
        return out

    def gradient(self, se=(7, 20)):
        '''
        INPUTS:
            - input (np.float32 array): input 2D array to compute dilation.
            - se (int tuple): structuring element; determines the size of
            moving window.

        RETURNS:
            - dilation2D (np.array): 1D dilation filter applied output.
        '''
        dilation = self.dilation(se)
        erosion = self.erosion(se)
        out = dilation - erosion
        return out

    def opening(self, se=(7, 20)):
        erosion = self.erosion(se)
        out = self.dilation(erosion)
        return out

    def closing(self, se=(7, 20)):
        dilation = self.dilation(se)
        out = self.erosion(dilation)
        return out
