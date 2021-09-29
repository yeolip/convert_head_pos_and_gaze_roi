import csv
import numpy as np
import os
import glob
import cv2
import json
from time import time
import datetime as dt
# import pickle as pkl
import math
import csv
import pandas as pd
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl

import match_intersection as objmatch_roi
from util_calc import *

from tkinter import filedialog
from tkinter import *

# 최대 줄 수 설정
pd.set_option('display.max_rows', 2500)
# 최대 열 수 설정
pd.set_option('display.max_columns', 200)
# 표시할 가로의 길이
pd.set_option('display.width', 160)
# 출력값 소숫점4자리로 설정
pd.options.display.float_format = '{:.4f}'.format

deg2Rad = math.pi/180
rad2Deg = 180/math.pi

bcheck_match = objmatch_roi.match_intersection_roi()

def funcname():
    return sys._getframe(1).f_code.co_name + "()"

def callername():
    return sys._getframe(2).f_code.co_name + "()"

def print_current_time(text=''):
    tnow = dt.datetime.now()
    print('%s-%2s-%2s %2s:%2s:%2s \t%s' % (tnow.year, tnow.month, tnow.day, tnow.hour, tnow.minute, tnow.second, text))


def policy_gaze_roi_accuracy(tdata, text_column):
    tdata['hit_green_roi'] = "None"
    tdata['hit_amber_roi'] = "None"
    roi_group_green = {1: [1], 3: [3,4], 4: [3,4], 5: [5,9], 6:[6], 7:[7], 8:[8], 9:[9], 10:[10], 11:[11,19],
                       12:[12, 13], 14:[14, 12], 15:[15], 16:[16], 17:[17], 19:[19,11]}

    roi_group_amber = {1: [1,4,6,13], 3: [1,3,4,5,11,13,19], 4: [1,3,4,5,11,13,19], 5: [4,5,9,11,13,17,19], 6:[1,6,10,12,13,18,20], 7:[7,13,14,15],
                       8: [5,8,10,13,16,17], 9:[4,5,9,11,19], 10:[6,8,10,13,16,17,18], 11:[4,5,9,11,15,17,19],
                       12:[1,6,12,13,14,20], 14:[7,12,13,14,15], 15:[7,14,15], 16:[8,10,16,17,18,20], 17:[5,8,16,17], 19:[4,5,9,11,15,17,19]}


    for idx in tdata.index.values:
        # print(idx, 'index')
        # tdata.loc()
        tvalue = tdata.loc[idx, text_column]
        t_gt = tdata.loc[idx, 'gt_s_gaze_roi_das']
        # print(t_gt)
        try:
            fcheck = False
            for i, tkey in enumerate(roi_group_green[t_gt]):
                # print(i, tkey)
                if(tkey == tvalue):
                    fcheck = True
                    break
            if(fcheck == True):
                tdata.loc[idx, 'hit_green_roi'] = "TRUE"
            else:
                tdata.loc[idx, 'hit_green_roi'] = "FALSE"
        except KeyError:
            print("Skip...", KeyError)


        try:
            fcheck = False
            for i, tkey in enumerate(roi_group_amber[t_gt]):
                # print(i, tkey)
                if(tkey == tvalue):
                    fcheck = True
                    break
            if(fcheck == True):
                tdata.loc[idx, 'hit_amber_roi'] = "TRUE"
            else:
                tdata.loc[idx, 'hit_amber_roi'] = "FALSE"
        except KeyError:
            print("Skip...", KeyError)
    # print(1/0)

    print(tdata)
    return tdata


def counting_gaze_roi(tdata):
    # tdata['hit_green_roi'] = "None"
    # tdata['hit_amber_roi'] = "None"
    # tdata['hit_amber_roi'] = "None"
    # 'gt_s_gaze_roi_das', 'Load_file'
    print('unique', tdata['gt_s_gaze_roi_das'].unique())
    print('value_counts', tdata['gt_s_gaze_roi_das'].value_counts())
    # print(tdata[(tdata['hit_green_roi'] == 'TRUE') & (tdata['gt_s_gaze_roi_das'] == 1)])
    print(tdata[(tdata['hit_amber_roi'] == 'TRUE') & (tdata['gt_s_gaze_roi_das'] == 1)])

    for i in tdata['gt_s_gaze_roi_das'].unique():
        print()
        tcount_green = tdata[(tdata['gt_s_gaze_roi_das'] == i)]['hit_green_roi'].count()
        tpass_green = tdata[(tdata['hit_green_roi'] == 'TRUE') & (tdata['gt_s_gaze_roi_das'] == i) ]['hit_green_roi'].count()
        print('green roi',i,' - ',tpass_green, '/' , tcount_green, '={:.2f}'.format(tpass_green/tcount_green*100), '%')

        tcount_amber = tdata[(tdata['gt_s_gaze_roi_das'] == i)]['hit_amber_roi'].count()
        tpass_amber = tdata[(tdata['hit_amber_roi'] == 'TRUE') & (tdata['gt_s_gaze_roi_das'] == i) ]['hit_amber_roi'].count()
        print('amber roi',i,' - ',tpass_amber, '/' , tcount_amber, '={:.2f}'.format(tpass_amber/tcount_amber*100), '%')
        # print(tcount_amber, "  ", tpass_amber)
        # print(1 / 0)

        # tpass_amber = tdata[(tdata['hit_amber_roi'] == 'TRUE') & (tdata['gt_s_gaze_roi_das'] == i) &  (tdata['hit_amber_roi'] == 'TRUE')].count()
        # print(tpass_amber, '/' , tcount_amber, '=', tpass_amber/tcount_amber*100, '%')
    # df_sort[(df_sort['title'] == tTitle) & (df_sort['compare_group'] == tGcomp)]
    # tdata[
    pass

def load_jsonfile_ROI(fname):
    print("//////////", funcname(), "//////////")

    fp = open(fname)
    fjs = json.load(fp)
    fp.close()
    # print(fjs)
    return fjs

def save_csvfile(tdatas, filename):
    print("//////////", funcname(), "//////////")
    if (tdatas.empty is True):
        print("저장할 데이터가 없습니다.")
        return
    tdata = tdatas.copy()
    tfile = os.path.splitext(filename)

    for num in range(100):
        tfilename = '%s' % (tfile[len(tfile) - 2]) + '%03d' % (num) + tfile[len(tfile) - 1]
        if not os.path.exists(tfilename):
             break
    print("파일을 저장합니다.", tfilename)
    tdata.to_csv(tfilename, mode='w', index=False, header=True, sep=',', quotechar=" ", float_format='%.4f')
    pass


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

def extract_availData_from_GT_and_result(inputPath_GT):
    print("//////////", funcname(), "//////////")
    extGT = pd.read_csv(inputPath_GT)
    df_extGT = extGT[[
                         'CAN_S_Gaze_ROI', 'gt_s_gaze_roi_das', 'gt_name_gaze_roi', 'roi_idx_h','roi_name_h', 'Load_file'
                    ]]
    df_extGT = df_extGT.dropna()
    df_extGT = df_extGT.sort_values(['gt_s_gaze_roi_das', 'Load_file'], ascending=(True, True))
    # print('df_extGT\n\n', df_extGT)
    return df_extGT

def rendering_roi_with_head_gaze(pROI, extData):

    fig = plt.figure(figsize=(10,8))
    ax3 = fig.add_subplot(111, projection='3d')

    plt.title('3D Target ROI')
    # for i in pROI:
    #     print(i)

    for i in pROI.index[0:-1]:
        # print(pROI.tID[i])
        # print(pROI.tTargetName[i])
        # print(pROI.ttop_left[i][0],pROI.ttop_left[i][1],pROI.ttop_left[i][2])
        # print(pROI.ttop_right[i])
        # print(pROI.tbottom_left[i])
        # print(pROI.tbottom_right[i])
        x0 = pROI.ttop_left[i][0] * 1000
        y0 = pROI.ttop_left[i][1] * 1000
        z0 = pROI.ttop_left[i][2] * 1000
        # ax3.scatter(xs=x, ys=y, zs=z, label=i[1])
        x1 = pROI.ttop_right[i][0] * 1000
        y1 = pROI.ttop_right[i][1] * 1000
        z1 = pROI.ttop_right[i][2] * 1000
        # ax3.scatter(xs=x, ys=y, zs=z, label=i[1])
        x2 = pROI.tbottom_left[i][0] * 1000
        y2 = pROI.tbottom_left[i][1] * 1000
        z2 = pROI.tbottom_left[i][2] * 1000
        # ax3.scatter(xs=x, ys=y, zs=z, label=i[1])
        x3 = pROI.tbottom_right[i][0] * 1000
        y3 = pROI.tbottom_right[i][1] * 1000
        z3 = pROI.tbottom_right[i][2] * 1000
        # print([x0,x1,x3,x2])
        ax3.scatter(xs=np.array([x0,x1,x3,x2]), ys=np.array([y0,y1,y3,y2]), zs=np.array([z0,z1,z3,z2]) )
        ax3.plot([x0,x1,x3,x2,x0], [y0,y1,y3,y2,y0], [z0,z1,z3,z2,z0], label=str("%.2d_"%pROI.tID[i])+pROI.tTargetName[i])

    for tindex in extData.index.values:
        # print(tindex, "번째 index, frameID = ",extData.loc[tindex, 'f_frame_counter_left_camera'], extData.loc[tindex, 'intersect_x_h'],extData.loc[tindex, 'intersect_y_h'],extData.loc[tindex, 'intersect_z_h'], '\n')
        h_x = extData.loc[tindex, 'HSVL_MS_S_Head_Pos_Veh_X']
        h_y = extData.loc[tindex, 'HSVL_MS_S_Head_Pos_Veh_Y']
        h_z = extData.loc[tindex, 'HSVL_MS_S_Head_Pos_Veh_Z']
        in_x = extData.loc[tindex, 'intersect_x_h']
        in_y = extData.loc[tindex, 'intersect_y_h']
        in_z = extData.loc[tindex, 'intersect_z_h']
        # ax3.scatter(xs=np.array([h_x, in_x]), ys=np.array([h_y, in_y]), zs=np.array([h_z, in_z]), s=50, c='r')
        ax3.scatter(xs=np.array([h_x]), ys=np.array([h_y]), zs=np.array([h_z]), s=50, c='r')
        ax3.scatter(xs=np.array([in_x]), ys=np.array([in_y]), zs=np.array([in_z]), s=10, c='b')
        ax3.plot([h_x, in_x], [h_y, in_y], [h_z, in_z] ) #label=str(extData.loc[tindex, 'f_frame_counter_left_camera'])
        # ax3.scatter(xs=np.array([h_x0, x1, x3, x2]), ys=np.array([y0, y1, y3, y2]), zs=np.array([z0, z1, z3, z2]))

    # print(1/0)
    # for ii in extData.index:
    #     print(ii)
    #     tview_point2[0]

    ax3.set_zlim(-1000, 1000)
    ax3.set_title("3D Target ROI")
    ax3.set_xlabel('veh X', fontsize=16)
    ax3.set_ylabel('veh Y', fontsize=16)
    ax3.set_zlabel('veh Z', fontsize=16)
    ax3.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    ax3.view_init(5, -10)

    plt.show()
    pass


if __name__ == '__main__':
    print("\n\n\n test/////////////////////")
    if(0):
        sys.stdout = open('DebugLog.txt', 'w')

    inputfile = "./accuracy_output005.csv"
    df_data = extract_availData_from_GT_and_result(inputfile)
    df_data2 = policy_gaze_roi_accuracy(df_data, 'roi_idx_h')
    counting_gaze_roi(df_data2)
    save_csvfile(df_data2, 'save_output.csv')
    print(1/0)
    # fold_names = filedialog.askdirectory()
    # files_to_replace_base = []
    #
    # if (1):
    #     for dirpath, dirnames, filenames in os.walk(fold_names):
    #         for filename in [f for f in filenames if f.endswith(".csv")]:
    #             files_to_replace_base.append(os.path.join(dirpath, filename))
    #             print(os.path.join(dirpath, filename))
    #
    #             # if (filename.__contains__("GT_") == True):
    #             #     files_to_replace_base.append(os.path.join(dirpath, filename))
    #             #     print(os.path.join(dirpath, filename))
    #             # elif (filename.__contains__("DisplayCenter") == True):
    #             #     files_to_replace.append(os.path.join(dirpath, filename))
    #             #     print(os.path.join(dirpath, filename))
    #
    # print(len(files_to_replace_base))
    # if(len(files_to_replace_base)== 0):
    #     print("No select file..!!!")
    #     print(1/0)
    # # files_to_replace_target = files_to_replace_base.copy()
    # print("*" * 50)
    # print(sorted(files_to_replace_base,key=lambda x: str(x).split() ))
    # print(files_to_replace_base)



    inputPath_ROI = "./refer/roi_config_eva.json"
    # roi_config.json
    ret_roi = load_jsonfile_ROI(inputPath_ROI)
    ret_ExtROI = extract_availData_from_3D_target_ROI(ret_roi)

    test = ['D:/Project/CVT/demo/ROI_GT/3810(EVA2DAS_GazeROI)\\3810_285_812238_0001_all.csv', 'D:/Project/CVT/demo/ROI_GT/3810(EVA2DAS_GazeROI)\\3810_2860_817361_0001_all.csv']
    test1 = ['D:\Source\convert_head_pos_and_gaze_roi/refer\GT/3810_8600_827980_0001_all_19.csv']
    test2 = ['./refer/GT_3531_96_670222_0001_all.csv']
    df_merge = pd.DataFrame()
    if(1):
        for i, tname in enumerate(files_to_replace_base):
            ttext = str(i) + '/' + str(len(files_to_replace_base)) +' - ' + os.path.basename(tname)
            print_current_time(ttext)
            # print('\n',i,'/',len(files_to_replace_base), tname)
            # print(1/0)
            inputPath_GT = tname

            ret_ExtGT = extract_availData_from_GT_short(inputPath_GT)
            ret_ExtGT_with_direction = retcalcuate_head_eye_direction_short(ret_ExtGT)
            # print('\n\n', ret_ExtGT_with_direction)
            ret_match = check_match_roi(ret_ExtGT_with_direction, ret_ExtROI)  # 150
            ret_match['Load_file'] =  os.path.basename(tname)
            # print('ret_match', ret_match)
            df_merge = pd.concat([df_merge, ret_match]).reset_index(drop=True)

        print("df_merge",df_merge)
        df_merge = df_merge.astype({'roi_idx_h':"int64", "gt_s_gaze_roi_das":"int64"})
        df_merge['hit'] = (df_merge['roi_idx_h'] == df_merge['gt_s_gaze_roi_das'])

        print("***Final hit accuracy\nTrue={}개, Total={}개, {}%".format(df_merge['hit'].value_counts()[1], df_merge['hit'].size, df_merge['hit'].value_counts()[1]/df_merge['hit'].size*100))
        save_csvfile(df_merge, "./accuracy_output.csv")

        # rendering_roi_with_head_gaze(ret_ExtROI, df_merge)
        print(1/0)
        # inputPath_GT = "./refer/GT_3531_96_670222_0001_all.csv"
        # inputPath_GT = "./refer/GT_3531_96_670222_0001_small.csv"
        # inputPath_GT = "./refer/GT_3531_96_670222_0001_mix.csv"
        inputPath_GT = "./refer/GT/3810_10_811709_0001_all.csv"
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


        ret_ExtGT = extract_availData_from_GT_short(inputPath_GT)
        # print('ret_ExtGT\n\n', ret_ExtGT)

        ret_ExtGT_with_direction = retcalcuate_head_eye_direction_short(ret_ExtGT)

        print('\n\n',ret_ExtGT_with_direction)
        ret_match = check_match_roi(ret_ExtGT_with_direction, ret_ExtROI, 0)    #150
        # save_csvfile(ret_match, "./accuracy_output.csv")
        # ret_match.to_csv("filename.csv", mode='w', index=False, header=False, sep=',', quotechar=" ",
        #                  float_format='%.4f')

        # rendering_roi_with_head_gaze(ret_ExtROI, ret_match)
        # ret_match.to_csv("filename.csv", mode='w', index=False, header=False, sep=',', quotechar=" ", float_format='%.4f')


    # print(1/0)
