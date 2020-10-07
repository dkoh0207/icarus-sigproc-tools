import numpy as np

SPROC_PATH='/sdf/group/neutrino/koh0207/DSP/signal_processing'
import sys
sys.path.insert(0,'%s/build/lib' % SPROC_PATH)
sys.path.insert(0,'%s/python'    % SPROC_PATH)
import ROOT
ROOT.gSystem.Load('%s/build/lib/libLiteFMWK_PyUtil.so' % SPROC_PATH)
ROOT.gSystem.Load('%s/build/lib/libLiteFMWK_sigproc_tools.so' % SPROC_PATH)

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
        numChannels, numTicks = self.waveform.shape
        waveform = sproc.pyutil.as_float32_vector_2d(self.waveform)
        out = ROOT.std.vector('std::vector<float>')(
            numChannels, ROOT.std.vector('float')(numTicks))
        self.morph2d.getDilation(waveform, se[0], se[1], out)
        out = np.asarray(out).astype(np.float32)
        return out

    def erosion(self, se=(7, 20)):
        numChannels, numTicks = self.waveform.shape
        waveform = sproc.pyutil.as_float32_vector_2d(self.waveform)
        out = ROOT.std.vector('std::vector<float>')(
            numChannels, ROOT.std.vector('float')(numTicks))
        self.morph2d.getErosion(waveform, se[0], se[1], out)
        out = np.asarray(out).astype(np.float32)
        return out

    def gradient(self, se=(7, 20)):
        numChannels, numTicks = self.waveform.shape
        waveform = sproc.pyutil.as_float32_vector_2d(self.waveform)
        out = ROOT.std.vector('std::vector<float>')(
            numChannels, ROOT.std.vector('float')(numTicks))
        self.morph2d.getGradient(waveform, se[0], se[1], out)
        out = np.asarray(out).astype(np.float32)
        return out

    def closing(self, se=(7, 20)):
        numChannels, numTicks = self.waveform.shape
        waveform = sproc.pyutil.as_float32_vector_2d(self.waveform)
        out = ROOT.std.vector('std::vector<float>')(
            numChannels, ROOT.std.vector('float')(numTicks))
        self.morph2d.getClosing(waveform, se[0], se[1], out)
        out = np.asarray(out).astype(np.float32)
        return out

    def opening(self, se=(7, 20)):
        numChannels, numTicks = self.waveform.shape
        waveform = sproc.pyutil.as_float32_vector_2d(self.waveform)
        out = ROOT.std.vector('std::vector<float>')(
            numChannels, ROOT.std.vector('float')(numTicks))
        self.morph2d.getOpening(waveform, se[0], se[1], out)
        out = np.asarray(out).astype(np.float32)
        return out