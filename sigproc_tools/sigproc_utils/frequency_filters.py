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
#   1D and 2D Frequency Space Filtering Modules with C++ backend.
#
#   2019. 12. 16    Dae Heun Koh
#
#######################################################################

class FrequencyFilterBase:

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


class FrequencyFilter1D(FrequencyFilterBase):

    def __init__(self, waveform):
        super(FrequencyFilter1D, self).__init__(waveform)
        self.frequency1d = ROOT.sigproc_tools.FrequencyFilters1D()
        self.mode_dict = {
            'hb': 0,
            'hh': 1,
            'hg': 2,
            'lb': 3,
            'lg': 4
        }

    def filter(self, waveform, threshold=10, order=4, mode='hb'):
        numChannels, numTicks = self.waveform.shape
        waveform = sproc.pyutil.as_float32_vector_2d(self.waveform)
        out = ROOT.std.vector('std::vector<float>')(
            numChannels, ROOT.std.vector('float')(numTicks))
        m = self.mode_dict[mode]
        self.frequency1d.filterImage(waveform, threshold, out, order, m)
        out = np.asarray(out).astype(np.float32)
        return out