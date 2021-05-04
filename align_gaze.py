import csv
import numpy as np
import os
import glob
import cv2
import json
from time import time
import datetime
# import pickle as pkl
import math
import csv
import pandas as pd
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# import match_intersection as objmatch_roi
import make_gaze as mg
from util_calc import *

# 최대 줄 수 설정
pd.set_option('display.max_rows', 2500)
# 최대 열 수 설정
pd.set_option('display.max_columns', 200)
# 표시할 가로의 길이
pd.set_option('display.width', 160)
# 출력값 소숫점4자리로 설정
pd.options.display.float_format = '{:.4f}'.format

def change_mra2_roi_index_to_target_name(extData, ret_ExtROI):
    extData['MRA2_TARGET_NAME'] = ""
    for tindex in extData.index.values:
        print(tindex,"번째 index, frameID = ", extData.loc[tindex, 'f_frame_counter_left_camera'],extData.loc[tindex, 'MRA2_ORG_CAN_S_Gaze_ROI'],'\n')
        dataroi_idx = extData.loc[tindex, 'MRA2_ORG_CAN_S_Gaze_ROI']
        for tidx in ret_ExtROI.index.values:
            troi_id = ret_ExtROI["tID"][tidx]
            troi_name = ret_ExtROI["tTargetName"][tidx]
            if(troi_id == dataroi_idx):
                print(' ', tidx, ret_ExtROI['tID'][tidx], ret_ExtROI['tTargetName'][tidx])
                extData.loc[tindex, 'MRA2_TARGET_NAME'] = troi_name
                break
            # print('\n')
        # print(1/0)
    return extData

def load_jsonfile_ROI(fname):
    print("//////////", funcname(), "//////////")

    fp = open(fname)
    fjs = json.load(fp)
    fp.close()
    # print(fjs)
    return fjs

def extract_availData_from_3D_target_ROI(pROI):
    print("//////////", funcname(), "//////////")
    tValid = []
    target_roi = pROI['ROI']

    for i, data in enumerate(target_roi):
        # print(i, data)
        bValid = True
        tID = []
        tTargetName = ["None"]
        ttop_left = []
        ttop_right = []
        tbottom_left = []
        tbottom_right = []

        for name in data:
            # print('name',name)
            if (name == 'id'):
                print('id', data['id'])
                tID.append(data['id'])
            elif(name == 'obj_params'):
                # print('obj_params', data['obj_params'])
                print('obj_params', data['obj_params']['top_left'], data['obj_params']['top_right'], data['obj_params']['bottom_left'])
                if(data['obj_params']['top_left'][0]==0.0 and data['obj_params']['top_right'][0]==0.0 and data['obj_params']['bottom_left'][0]==0.0):
                    bValid = False
                    break
                ttop_left.append(data['obj_params']['top_left'])
                ttop_right.append(data['obj_params']['top_right'])
                tbottom_left.append(data['obj_params']['bottom_left'])
                temp = np.round((np.array(tbottom_left) + np.array(ttop_right) - np.array(ttop_left)),5).tolist()[0]
                tbottom_right.append(temp)
            elif(name == '_comment'):
                print('_comment', data['_comment'])
                tTargetName[-1] = data['_comment']
        if(bValid == True):
            tValid.append([tID, tTargetName, ttop_left, ttop_right, tbottom_left, tbottom_right])

    print('tValid', tValid)
    tValid = np.array(tValid.copy())
    available_dict = {"tID":tValid.T[0][0],"tTargetName":tValid.T[0][1],
                      "ttop_left":tValid.T[0][2],
                      "ttop_right": tValid.T[0][3],
                      "tbottom_left": tValid.T[0][4],
                      "tbottom_right": tValid.T[0][5]
                      }

    available_df = pd.DataFrame(available_dict)  # index 지정
    print(available_df)
    return available_df

if __name__ == '__main__':
    if (0):
        sys.stdout = open('DebugLog.txt', 'w')

    #final target output
    inputPath_CVT_CALC = "./roi_output000.csv"

    inputPath_MRA2_CALC = "./basegaze_output000.csv"

    inputPath_MRA2_ORIGIN = "./refer/GT_3531_96_670222_0001_all.csv"
    # inputPath_GT = "./refer/GT_3531_96_670222_0001_small.csv"
    # inputPath_GT = "./refer/GT_3531_96_670222_0001_mix.csv"
    # inputPath_GT = "./refer/GT/3810_10_811709_0001_all.csv"
    # inputPath_GT = "./refer/GT/3810_20_811728_0001_all.csv"
    # inputPath_GT = "./refer/GT/3810_30_811746_0001_all.csv"
    # inputPath_GT = "./refer/GT/3810_40_811766_0001_all.csv"
    # inputPath_GT = "./refer/GT/3810_50_811786_0001_all.csv"
    # inputPath_GT = "./refer/GT/3810_70_811824_0001_all.csv"
    # inputPath_GT = "./refer/GT/3810_60_811805_0001_all.csv"
    # inputPath_GT = "./refer/GT/3810_80_811843_0001_all.csv"
    # inputPath_GT = "./refer/GT/3810_80_811843_0001_all.csv"
    # inputPath_GT = "./refer/GT/3810_80_811843_0001_all.csv"
    # inputPath_GT = "./refer/GT/3810_90_811862_0001_all.csv"
    # inputPath_GT = "./refer/GT/3810_100_811882_0001_all.csv"

    # roi_config.json
    inputPath_ROI = "./refer/roi_config.json"

    objgaze = mg.make_gaze_and_roi()
    ret_resultMRA2_ORG = objgaze.extract_resultRoi_from_GT(inputPath_MRA2_ORIGIN, "MRA2_ORG_")
    ret_resultMRA2_CALC = objgaze.extract_resultRoi_from_output(inputPath_MRA2_CALC, "MRA2_CALC_")
    ret_resultCVT_CALC = objgaze.extract_resultRoi_from_output(inputPath_CVT_CALC, "CVT_CALC_")

    ret_roi = load_jsonfile_ROI(inputPath_ROI)
    ret_ExtROI = extract_availData_from_3D_target_ROI(ret_roi)
    ret_resultMRA2_ORG = change_mra2_roi_index_to_target_name(ret_resultMRA2_ORG, ret_ExtROI)
    print(ret_resultMRA2_ORG)

    ret_roi_result = pd.merge(ret_resultMRA2_ORG, ret_resultMRA2_CALC, how='left', left_on="f_frame_counter_left_camera", right_on="f_frame_counter_left_camera")
    # test = pd.concat([ret_match, ret_resultGT], axis=1)
    print(ret_roi_result)
    ret_roi_result = pd.merge(ret_roi_result, ret_resultCVT_CALC, how='left', left_on="f_frame_counter_left_camera",
                              right_on="f_frame_counter_left_camera")
    print(ret_roi_result)


    objgaze.save_csvfile(ret_roi_result, "./align_roi.csv")
    # print(1/0)








