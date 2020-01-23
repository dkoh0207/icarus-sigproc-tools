import uproot
import numpy as np
import pandas as pd
import yaml
import math
import sys
import os
import argparse

sigProcPath = "/u/nu/koh0207/icarus-sigproc-tools"
sys.path.insert(0,sigProcPath)

from benchmark import *

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-test_cfg', '--test_config', help='config_path', type=str)
    #parser.add_argument('-ckpt', '--checkpoint_number', type=int)
    args = parser.parse_args()
    args = vars(args)
    cfg = yaml.load(open(args['test_config'], 'r'), Loader=yaml.Loader)
    print(cfg)
    nEMin = cfg['nEMin']
    nEMax = cfg['nEMax']
    angleMin = cfg['angleMin']
    angleMax = cfg['angleMax']
    nSteps = cfg.get('nSteps', 10)
    kernel_size = cfg.get('kernel_size', 5)
    structuringElementx = cfg.get('structuringElementx', 5)
    structuringElementy = cfg.get('structuringElementy', 20)
    structuringElement = (structuringElementx, structuringElementy)
    window_size = cfg.get('window_size', 0)
    filter_name = cfg.get('filter_name', 'd')
    output_filename = cfg.get('output_filename', 'output.csv')
    threshold = cfg.get('threshold', 6.0)

    nElectrons = np.linspace(nEMin, nEMax, nSteps)
    angles = np.linspace(angleMin, angleMax, nSteps)
    output = []
    data = TPCResponseData(cfg)
    for ne in nElectrons:
        for angle in angles:
            # x.append(ne)
            # y.append(angle)
            data.genTrack(numElectrons=int(ne), angleToWire=angle)
            data.denoise(kernel_size=kernel_size, 
                        window_size=window_size, 
                        filter_name=filter_name,
                        structuringElement=structuringElement,
                        tFactor=threshold)
            res = data.run_analysis()
            print("trueSNR = ", res['trueSNR'])
            print("trueSNRErr = ", res['trueSNRErr'])
            print("Purity = ", res['purity'])
            print("Efficiency = ", res['efficiency'])
            row = [ne, angle, threshold] + list(res.values())
            row = tuple(row)
            output.append(row)
    columns = ['num_electrons', 'angle', 'threshold'] + list(res.keys())
    output = pd.DataFrame(output, columns=columns)
    output.to_csv(output_filename, index=False)