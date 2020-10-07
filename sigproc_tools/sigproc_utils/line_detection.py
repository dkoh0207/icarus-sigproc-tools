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
#   Hough Transform Related Modules with C++ backend.
#
#   2019. 12. 16    Dae Heun Koh
#
#######################################################################


def hough_transform(binary2D, rhoSteps=360, thetaSteps=360):

    module = ROOT.sigproc_tools.LineDetection()

    wf = sproc.pyutil.as_bool_vector_2d(binary2D)
    out = ROOT.std.vector('std::vector<int>')(
        rhoSteps, ROOT.std.vector('int')(thetaSteps))

    module.HoughTransform(wf, out, thetaSteps)
    out_np = np.asarray(out)
    return out, out_np


def nms(accumulator, max_lines=20, window_size=2, angle_window=10, threshold=300, return_np=True):

    module = ROOT.sigproc_tools.LineDetection()

    rhoIndex = ROOT.std.vector('int')()
    thetaIndex = ROOT.std.vector('int')()

    module.FindPeaksNMS(accumulator, rhoIndex, thetaIndex, threshold, angle_window, max_lines, window_size)

    if return_np:
        rhoIndex = np.asarray(rhoIndex)
        thetaIndex = np.asarray(thetaIndex)

    return rhoIndex, thetaIndex


def refine_selectVals(selectVals, thetaSteps=360, threshold=300,
                      angleWindow=30, maxLines=20, 
                      windowSize=20, dilation=(7,20), eps=0.0001):
    module = ROOT.sigproc_tools.LineDetection()
    wf = sproc.pyutil.as_bool_vector_2d(selectVals)
    numChannels, numTicks = selectVals.shape
    out = ROOT.std.vector('std::vector<bool>')(
        numChannels, ROOT.std.vector('bool')(numTicks))
    module.refineSelectVals(wf, out, thetaSteps, threshold, angleWindow, 
        maxLines, windowSize, dilation[0], dilation[1], eps)

    return sproc.pyutil.as_ndarray(out).astype(bool)