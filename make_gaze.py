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

import match_intersection as mi
from util_calc import *

# 최대 줄 수 설정
pd.set_option('display.max_rows', 2500)
# 최대 열 수 설정
pd.set_option('display.max_columns', 200)
# 표시할 가로의 길이
pd.set_option('display.width', 160)
# 출력값 소숫점4자리로 설정
pd.options.display.float_format = '{:.4f}'.format

# deg2Rad = math.pi/180
# rad2Deg = 180/math.pi

C_PRINT_ENABLE = 1



class make_gaze_and_roi(object):
    def __del__(self):
        print("*************delete make_gaze_and_roi class***********\n")

    def __init__(self):
        print("*************initial make_gaze_and_roi class***********\n")
        self.debugflag = C_PRINT_ENABLE
        self.obj_mi = mi.match_intersection_roi()
        pass

    def load_jsonfile_ROI(self, fname):
        print("//////////", funcname(), "//////////")

        fp = open(fname)
        fjs = json.load(fp)
        fp.close()
        # print(fjs)
        return fjs

    def save_csvfile(self, tdatas, filename):
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

    def extract_availData_from_3D_target_ROI(self, pROI):
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

    def extract_resultRoi_from_GT(self, inputPath_GT, text=""):
        print("//////////", funcname(), "//////////")
        extGT = pd.read_csv(inputPath_GT)
        df_extGT = extGT[['f_frame_counter_left_camera', 'CAN_S_Gaze_ROI', 'MS_S_Gaze_ROI_X_Raw', 'MS_S_Gaze_ROI_Y_Raw']]
        df_extGT = df_extGT.dropna()
        df_extGT.columns = ['f_frame_counter_left_camera', text+'CAN_S_Gaze_ROI', text+'MS_S_Gaze_ROI_X_Raw', text+'MS_S_Gaze_ROI_Y_Raw']
        # df_extGT = df_extGT.rename(columns={"f_frame_counter_left_camera": "f_frame_counter_left_camera","CAN_S_Gaze_ROI": "MRA2_CAN_S_Gaze_ROI","MS_S_Gaze_ROI_X_Raw": "MRA2_MS_S_Gaze_ROI_X_Raw", "MS_S_Gaze_ROI_Y_Raw": "MRA2_MS_S_Gaze_ROI_Y_Raw"}, inplace = True)
        # print('ret_ExtGT\n\n',ret_ExtGT)
        print('df_extGT\n\n', df_extGT)
        return df_extGT

    def extract_resultRoi_from_output(self, inputPath_out, text=""):
        print("//////////", funcname(), "//////////")
        extOut = pd.read_csv(inputPath_out)
        df_extOUT = extOut[['f_frame_counter_left_camera', 'roi_idx_h', 'roi_name_h', 'roi_X', 'roi_Y']]
        df_extOUT = df_extOUT.dropna()
        df_extOUT.columns = ['f_frame_counter_left_camera', text+'roi_idx_h', text+'roi_name_h', text+'roi_X', text+'roi_Y']
        print('df_extOUT\n\n', df_extOUT)
        return df_extOUT

    def extract_availData_from_GT(self, inputPath_GT):
        print("//////////", funcname(), "//////////")
        extGT = pd.read_csv(inputPath_GT)
        df_extGT = extGT[['f_frame_counter_left_camera', 'CAN_S_Gaze_ROI', 'MS_S_Gaze_ROI_X_Raw', 'MS_S_Gaze_ROI_Y_Raw',
                              'MS_S_Head_rot_X', 'MS_S_Head_rot_Y', 'MS_S_Head_rot_Z',
                              'HSVL_MS_S_Head_Pos_Veh_X', 'HSVL_MS_S_Head_Pos_Veh_Y', 'HSVL_MS_S_Head_Pos_Veh_Z',
                              'MS_S_Gaze_LE_Center_X', 'MS_S_Gaze_LE_Center_Y', 'MS_S_Gaze_LE_Center_Z',
                              'MS_S_Gaze_LE_VA_rot_X', 'MS_S_Gaze_LE_VA_rot_Y', 'MS_S_Gaze_LE_VA_rot_Z',
                              'MS_S_Gaze_RE_Center_X', 'MS_S_Gaze_RE_Center_Y', 'MS_S_Gaze_RE_Center_Z',
                              'MS_S_Gaze_RE_VA_rot_X', 'MS_S_Gaze_RE_VA_rot_Y', 'MS_S_Gaze_RE_VA_rot_Z',
                              'MS_S_Gaze_rot_X',       'MS_S_Gaze_rot_Y',     'MS_S_Gaze_rot_Z']]
        df_extGT = df_extGT.dropna()
        # print('ret_ExtGT\n\n',ret_ExtGT)
        print('df_extGT\n\n', df_extGT)
        return df_extGT

    def extract_availData_from_pandas(self, extPd):
        print("//////////", funcname(), "//////////")
        # extGT = pd.read_csv(inputPath_GT)
        df_extPd = extPd[['f_frame_counter_left_camera', 'CAN_S_Gaze_ROI', 'MS_S_Gaze_ROI_X_Raw', 'MS_S_Gaze_ROI_Y_Raw',
                              'HSVL_MS_CAN_S_Head_tracking_status',
                              'MS_S_Head_rot_X', 'MS_S_Head_rot_Y', 'MS_S_Head_rot_Z',
                              'HSVL_MS_S_Head_Pos_Veh_X', 'HSVL_MS_S_Head_Pos_Veh_Y', 'HSVL_MS_S_Head_Pos_Veh_Z',
                              'f_gaze_le_result_valid',
                              'MS_S_Gaze_LE_Center_X', 'MS_S_Gaze_LE_Center_Y', 'MS_S_Gaze_LE_Center_Z',
                              'MS_S_Gaze_LE_VA_rot_X', 'MS_S_Gaze_LE_VA_rot_Y', 'MS_S_Gaze_LE_VA_rot_Z',
                              'f_gaze_re_result_valid',
                              'MS_S_Gaze_RE_Center_X', 'MS_S_Gaze_RE_Center_Y', 'MS_S_Gaze_RE_Center_Z',
                              'MS_S_Gaze_RE_VA_rot_X', 'MS_S_Gaze_RE_VA_rot_Y', 'MS_S_Gaze_RE_VA_rot_Z',
                              'MS_S_Gaze_rot_X',       'MS_S_Gaze_rot_Y',     'MS_S_Gaze_rot_Z']]
        # df_extPd = df_extPd.dropna()
        # print('ret_ExtGT\n\n',ret_ExtGT)
        print('df_extPd\n\n', df_extPd)
        return df_extPd

    def retcalcuate_head_eye_direction(self, extData):
        print("//////////", funcname(), "//////////")
        # print(extData)
        extData['headDir_X_L'] =  0
        extData['headDir_Y_L'] =  0
        extData['headDir_Z_L'] =  0
        extData['headDir_X_R'] = 0
        extData['headDir_Y_R'] = 0
        extData['headDir_Z_R'] = 0
        extData['headDir_X_mid'] = 0
        extData['headDir_Y_mid'] = 0
        extData['headDir_Z_mid'] = 0
        # extData['Rot_headDir_X_L'] =  0
        # extData['Rot_headDir_Y_L'] =  0
        # extData['Rot_headDir_Z_L'] =  0
        # extData['Rot_headDir_X_R'] = 0
        # extData['Rot_headDir_Y_R'] = 0
        # extData['Rot_headDir_Z_R'] = 0
        extData['Rot_headDir_X_mid'] = 0
        extData['Rot_headDir_Y_mid'] = 0
        extData['Rot_headDir_Z_mid'] = 0

        for tindex in extData.index.values:
            # print(extData.irisHeight.loc[tindex])
            print(tindex)
            f_detect_face = extData.loc[tindex,'HSVL_MS_CAN_S_Head_tracking_status']
            f_detect_left_eye = extData.loc[tindex,'f_gaze_le_result_valid']
            f_detect_right_eye = extData.loc[tindex,'f_gaze_re_result_valid']
            if(f_detect_face == 0):
                continue
            if(f_detect_left_eye == 0 or f_detect_right_eye==0):
                continue
            tframecnt = extData.loc[tindex, 'f_frame_counter_left_camera']
            troi_result = extData.loc[tindex, 'CAN_S_Gaze_ROI']
            troi_x = extData.loc[tindex, 'MS_S_Gaze_ROI_X_Raw']
            troi_y = extData.loc[tindex, 'MS_S_Gaze_ROI_Y_Raw']
            thead_pos = np.array((extData.loc[tindex, 'HSVL_MS_S_Head_Pos_Veh_X'], extData.loc[tindex, 'HSVL_MS_S_Head_Pos_Veh_Y'], extData.loc[tindex, 'HSVL_MS_S_Head_Pos_Veh_Z']))
            thead_rot = np.array((extData.loc[tindex, 'MS_S_Head_rot_X'], extData.loc[tindex, 'MS_S_Head_rot_Y'], extData.loc[tindex, 'MS_S_Head_rot_Z']))

            tgaze_pos_l = np.array((extData.loc[tindex, 'MS_S_Gaze_LE_Center_X'], extData.loc[tindex, 'MS_S_Gaze_LE_Center_Y'], extData.loc[tindex, 'MS_S_Gaze_LE_Center_Z']))
            tgaze_vec_l = np.array((extData.loc[tindex, 'MS_S_Gaze_LE_VA_rot_X'], extData.loc[tindex, 'MS_S_Gaze_LE_VA_rot_Y'], extData.loc[tindex, 'MS_S_Gaze_LE_VA_rot_Z']))

            tgaze_pos_r = np.array((extData.loc[tindex, 'MS_S_Gaze_RE_Center_X'], extData.loc[tindex, 'MS_S_Gaze_RE_Center_Y'], extData.loc[tindex, 'MS_S_Gaze_RE_Center_Z']))
            tgaze_vec_r = np.array((extData.loc[tindex, 'MS_S_Gaze_RE_VA_rot_X'], extData.loc[tindex, 'MS_S_Gaze_RE_VA_rot_Y'], extData.loc[tindex, 'MS_S_Gaze_RE_VA_rot_Z']))
            tgaze_vec_mid = np.array((extData.loc[tindex, 'MS_S_Gaze_rot_X'], extData.loc[tindex, 'MS_S_Gaze_rot_Y'], extData.loc[tindex, 'MS_S_Gaze_rot_Z']))

            print(tframecnt, troi_result, troi_x, troi_y, thead_pos, thead_rot, tgaze_pos_l, tgaze_vec_l, tgaze_pos_r, tgaze_vec_r )
            # print(tframecnt, troi_result, troi_x, troi_y)

            headPos3D_mm = thead_pos
            # headOri_radian = thead_rot * deg2Rad
            headOri_radian = np.array([0, 0, 0]) * deg2Rad
            print("headPos3D_mm", headPos3D_mm)
            print("headOri_radian", headOri_radian)

            lpupil_roll_pitch_yaw_rad = tgaze_vec_l * deg2Rad #np.array([0, -9.8, 0.2])
            rpupil_roll_pitch_yaw_rad = tgaze_vec_r * deg2Rad #np.array([0, -9.8, 0.2])
            mideye_roll_pitch_yaw_rad = tgaze_vec_mid * deg2Rad

            rt_2 = np.dot(eulerAnglesToRotationMatrix(np.array([0, 0, math.pi])), np.array([1, 1, 1])).round(5)
            rt = eulerAnglesToRotationMatrix(headOri_radian * rt_2)

            rot2_l = eulerAnglesToRotationMatrix(lpupil_roll_pitch_yaw_rad)
            rot2_r = eulerAnglesToRotationMatrix(rpupil_roll_pitch_yaw_rad)
            rot2_mid = eulerAnglesToRotationMatrix(mideye_roll_pitch_yaw_rad)

            print('rot2_l',rot2_l)
            # headDir_r = np.dot(rot2_mid,[1,0,0]) * np.dot(rt, [1, 0, 0])
            headDir_l = np.dot(rot2_l, np.dot(rt, [1, 0, 0]))
            # headDir_l = np.dot(rt, np.dot(rot2_l, [1, 0, 0]))
            headDir_r = np.dot(rot2_r, np.dot(rt, [1, 0, 0]))
            # headDir_r = np.dot(rt, np.dot(rot2_r, [1, 0, 0]))
            headDir_mid = np.dot(rot2_mid, np.dot(rt, [1, 0, 0]))
            # headDir_mid = np.dot(rt, np.dot(rot2_mid, [1, 0, 0]))

            headDir_l = np.dot(np.dot(rot2_l , rt), [1,0,0])
            headDir_r = np.dot(np.dot(rot2_r , rt), [1,0,0])
            headDir_mid = np.dot(np.dot(rot2_mid , rt), [1,0,0])
            # print(headDir_mid)
            # print(rotationMatrixToEulerAngles(headDir_mid))
            # print(rotationMatrixToEulerAngles(headDir_mid)*rad2Deg)
            # np.dot(rot2_mid, np.dot(rt, [1, 0, 0]))
            print('headDir_l',headDir_l)
            # print(1/0)
            extData.loc[tindex, 'headDir_X_L'] = headDir_l[0]
            extData.loc[tindex, 'headDir_Y_L'] = headDir_l[1]
            extData.loc[tindex, 'headDir_Z_L'] = headDir_l[2]
            extData.loc[tindex, 'headDir_X_R'] = headDir_r[0]
            extData.loc[tindex, 'headDir_Y_R'] = headDir_r[1]
            extData.loc[tindex, 'headDir_Z_R'] = headDir_r[2]
            extData.loc[tindex, 'headDir_X_mid'] = headDir_mid[0]
            extData.loc[tindex, 'headDir_Y_mid'] = headDir_mid[1]
            extData.loc[tindex, 'headDir_Z_mid'] = headDir_mid[2]
            # aaa = changeRotation_unitvec2radian_check2('RPY', headDir_l, 'RPY') * rad2Deg
            # bbb = changeRotation_unitvec2radian_check2('RPY', headDir_r, 'RPY') * rad2Deg
            ccc = changeRotation_unitvec2radian('RPY', headDir_mid, 'RPY') * rad2Deg
            # print('aaa',aaa)
            # print('bbb',bbb)
            print('ccc',ccc)
            # extData.loc[tindex,'Rot_headDir_X_L'] = aaa[0]
            # extData.loc[tindex,'Rot_headDir_Y_L'] = aaa[1]
            # extData.loc[tindex,'Rot_headDir_Z_L'] = aaa[2]
            # extData.loc[tindex,'Rot_headDir_X_R'] = bbb[0]
            # extData.loc[tindex,'Rot_headDir_Y_R'] = bbb[1]
            # extData.loc[tindex,'Rot_headDir_Z_R'] = bbb[2]
            extData.loc[tindex,'Rot_headDir_X_mid'] = ccc[0]
            extData.loc[tindex,'Rot_headDir_Y_mid'] = ccc[1]
            extData.loc[tindex,'Rot_headDir_Z_mid'] = ccc[2]

        return extData

    def calc_match_roi(self, p0, p1, p3, p2, camPlaneOrthVector, pointOnPlan, headDir, headPos):

        # tview_point = intersectionWithPlan(headPos, headDir, camPlaneOrthVector, pointOnPlan)
        # print(' ', 'tview_point', tview_point)
        tview_point2 = self.obj_mi.line_plane_collision(camPlaneOrthVector, pointOnPlan, headDir, headPos)
        print("tview_point2", tview_point2)

        ret_match = self.obj_mi.check_available_point_on_plane(p0, p1, p3, p2, tview_point2)
        head_vector = np.dot(eulerAnglesToRotationMatrix(np.array([0, 0, math.pi])), headDir).round(5)
        ret_sameDirect = self.obj_mi.is_same_direction(tview_point2, head_vector, headPos)
        print("ret_match", ret_match, 'ret_sameDirect', ret_sameDirect)
        # print('각도', changeRotation_unitvec2radian_check2('RPY', headDir, 'RPY') * rad2Deg)
        if (ret_match == True and ret_sameDirect == True):
            return True, tview_point2
        return False, np.array([0,0,0])

    def calc_score(self, roi_center, roi_one_of_points, target_point):
        # calc 1 - (dist(target_point - roi_center) / dist(roi_one_of_points - roi_center))
        dist1 = distance_xyz(roi_center, target_point)
        dist2 = distance_xyz(roi_center, roi_one_of_points)
        result = np.round(1 - (dist1 / dist2), 2)
        return result

    def check_match_roi(self, extData, ret_ExtROI, errDist = 0):
        extData['roi_idx_h'] =  ""
        extData['roi_name_h'] = ""
        extData['roi_X'] = 0
        extData['roi_Y'] = 0
        extData['intersect_x_h'] =  0
        extData['intersect_y_h'] =  0
        extData['intersect_z_h'] =  0
        # extData['roi_idx_le'] =  ""
        # extData['roi_name_le'] = ""
        # extData['intersect_x_le'] =  0
        # extData['intersect_y_le'] =  0
        # extData['intersect_z_le'] =  0
        # extData['roi_idx_re'] =  ""
        # extData['roi_name_re'] = ""
        # extData['intersect_x_re'] =  0
        # extData['intersect_y_re'] =  0
        # extData['intersect_z_re'] =  0

        for tindex in extData.index.values:
            print(tindex,"번째 index, frameID = ", extData.loc[tindex, 'f_frame_counter_left_camera'],'\n')
            headDir_l = np.array((extData.loc[tindex, 'headDir_X_L'], extData.loc[tindex, 'headDir_Y_L'], extData.loc[tindex, 'headDir_Z_L']))
            headDir_r = np.array((extData.loc[tindex, 'headDir_X_R'], extData.loc[tindex, 'headDir_Y_R'], extData.loc[tindex, 'headDir_Z_R']))
            headDir_mid = np.array((extData.loc[tindex, 'headDir_X_mid'], extData.loc[tindex, 'headDir_Y_mid'], extData.loc[tindex, 'headDir_Z_mid']))

            # print(tindex, 'headDir_l',headDir_l)
            headPos = np.array((extData.loc[tindex, 'HSVL_MS_S_Head_Pos_Veh_X'], extData.loc[tindex, 'HSVL_MS_S_Head_Pos_Veh_Y'], extData.loc[tindex, 'HSVL_MS_S_Head_Pos_Veh_Z']))
            headPos_LE = np.array((extData.loc[tindex, 'MS_S_Gaze_LE_Center_X'],
                                extData.loc[tindex, 'MS_S_Gaze_LE_Center_Y'],
                                extData.loc[tindex, 'MS_S_Gaze_LE_Center_Z']))
            headPos_RE = np.array((extData.loc[tindex, 'MS_S_Gaze_RE_Center_X'],
                                extData.loc[tindex, 'MS_S_Gaze_RE_Center_Y'],
                                extData.loc[tindex, 'MS_S_Gaze_RE_Center_Z']))
            # headPos = headPos + np.array([ 0, -140 , -140]) #lip
            # headPos_LE = headPos_LE + np.array([ 0, 0, 0])
            # headPos_RE = headPos_RE + np.array([ 0, 0, 0])

            # np.array([ 9.78, -45.84, -47.19])
            # headRot = np.array((extData.loc[tindex, 'MS_S_Head_rot_X'], extData.loc[tindex, 'MS_S_Head_rot_Y'], extData.loc[tindex, 'MS_S_Head_rot_Z']))
            # rt_2 = np.dot(eulerAnglesToRotationMatrix(np.array([0, 0, math.pi])), np.array([1, 1, 1])).round(5)
            # rt = eulerAnglesToRotationMatrix(headRot * deg2Rad * rt_2)
            # headrot_unitvec = np.dot(rt, [1, 0, 0])

            for tidx in ret_ExtROI.index.values:
                # offset = 0
                # print(tidx)
                # print(ret_ExtROI['tID'], ret_ExtROI['tTargetName'], ret_ExtROI['ttop_left'], ret_ExtROI['ttop_right'], ret_ExtROI['tbottom_left'], ret_ExtROI['tbottom_right'])
                print(' ',tidx, ret_ExtROI['tID'][tidx], ret_ExtROI['tTargetName'][tidx])
                troi_id = ret_ExtROI["tID"][tidx]
                troi_name = ret_ExtROI["tTargetName"][tidx]

                if (troi_id == 7):
                    offset = 135 + 40 + 5 + 10 + 10 + 10 + 10 + 5
                elif (troi_id == 8):
                    offset = 170 + 40
                elif (troi_id == 9):
                    offset = 60 + 40 + 20 + 25 + 20 + 20 + 10 + 15
                elif (troi_id == 1):
                    offset = 80 + 0 + 5 + 10 + 20 + 10 + 20 + 20 + 15
                elif (troi_id == 3 or troi_id == 4):  # or troi_id == 5
                    offset = 80
                elif (troi_id == 5):
                    offset = 60 + 10
                elif (troi_id == 6):
                    offset = 25 + 15 + 15 + 20 + 20 + 10 + 20 + 20
                elif (troi_id == 14):
                    offset = 25 + 15 + 0 + 5
                elif (troi_id == 19):
                    offset = 25 + 15 + 0 + 5 + 0 + 0 + 5
                elif (troi_id == 16):
                    offset = 25 + 15 + 0 + 5 + 10 + 10
                elif (troi_id == 11):
                    offset = 25 + 15 + 0 + -5
                elif (troi_id == 10):
                    offset = 25 + 0 + 0 + 10 + 10 + 10 + 5
                else:
                    offset = errDist
                p0 = np.array(ret_ExtROI["ttop_left"][tidx]) * 1000  + np.array([0,-offset,offset])
                p1 = np.array(ret_ExtROI["ttop_right"][tidx]) * 1000 + np.array([0,offset,offset])
                p2 = np.array(ret_ExtROI["tbottom_left"][tidx]) * 1000 + np.array([0,-offset,-offset])
                p3 = np.array(ret_ExtROI["tbottom_right"][tidx]) * 1000 + np.array([0,offset,-offset])

                camPlaneOrthVector = np.cross((p3 - p1), (p2 - p3))/np.linalg.norm(np.cross((p3 - p1), (p2 - p3)))
                pointOnPlan = (p0+p1+p2+p3)/4

                # #최단거리
                # d = np.dot(camPlaneOrthVector, p0)
                # t = -(np.dot(origin,camPlaneOrthVector)+d)/ np.dot(gazeVector,camPlaneOrthVector)
                # pResult = origin + np.dot(t,gazeVector)
                # print('d',d)
                # print('t',t)
                # print('pResult',pResult)
                print(' ',"p0", p0, "\n  p1",p1, "\n  p2", p2, "\n  p3", p3)
                print(' ','headPos',headPos)
                print(' ','headDir_l', headDir_l)
                print(' ','camPlaneOrthVector',camPlaneOrthVector)
                print(' ','pointOnPlan',pointOnPlan)
                ret_check, point_mapping = self.calc_match_roi(p0, p1, p3, p2, camPlaneOrthVector, pointOnPlan, headDir_mid, headPos)
                if(ret_check == True):
                    extData.loc[tindex, 'roi_idx_h'] = str(troi_id) #extData.loc[tindex, 'roi_idx_h'] +'/'+ str(troi_id)
                    extData.loc[tindex, 'roi_name_h'] = extData.loc[tindex, 'roi_name_h'] +'/'+ troi_name
                    extData.loc[tindex, 'intersect_x_h'] = point_mapping[0]
                    extData.loc[tindex, 'intersect_y_h'] = point_mapping[1]
                    extData.loc[tindex, 'intersect_z_h'] = point_mapping[2]
                    extData.loc[tindex, 'roi_X'] = int(self.obj_mi.line_point_min_dist(point_mapping, p0, p2) / distance_xyz(p0, p1) * 100)
                    extData.loc[tindex, 'roi_Y'] = int(self.obj_mi.line_point_min_dist(point_mapping, p0, p1) / distance_xyz(p0,p2) * 100)
                    # ret_check_l, point_mapping_l = self.calc_match_roi(p0, p1, p3, p2, camPlaneOrthVector, pointOnPlan, headDir_l, headPos_LE)
                    # if(ret_check_l == True):
                    #     extData.loc[tindex, 'roi_idx_le'] = extData.loc[tindex, 'roi_idx_le'] +'#'+ str(troi_id)
                    #     extData.loc[tindex, 'roi_name_le'] = extData.loc[tindex, 'roi_name_le'] +'#'+ troi_name
                    #     extData.loc[tindex, 'intersect_x_le'] = point_mapping_l[0]
                    #     extData.loc[tindex, 'intersect_y_le'] = point_mapping_l[1]
                    #     extData.loc[tindex, 'intersect_z_le'] = point_mapping_l[2]
                    # ret_check_r, point_mapping_r = self.calc_match_roi(p0, p1, p3, p2, camPlaneOrthVector, pointOnPlan, headDir_r, headPos_RE)
                    # if(ret_check_r == True):
                    #     extData.loc[tindex, 'roi_idx_re'] = extData.loc[tindex, 'roi_idx_re'] +'*'+ str(troi_id)
                    #     extData.loc[tindex, 'roi_name_re'] = extData.loc[tindex, 'roi_name_re'] +'*'+ troi_name
                    #     extData.loc[tindex, 'intersect_x_re'] = point_mapping_r[0]
                    #     extData.loc[tindex, 'intersect_y_re'] = point_mapping_r[1]
                    #     extData.loc[tindex, 'intersect_z_re'] = point_mapping_r[2]
                    break
                print('\n')
        return extData

    def check_match_roi2(self, extData, ret_ExtROI, errDist=0):
        extData['roi_idx_h'] = ""
        extData['roi_name_h'] = ""
        extData['roi_X'] = 0
        extData['roi_Y'] = 0
        extData['intersect_x_h'] = 0
        extData['intersect_y_h'] = 0
        extData['intersect_z_h'] = 0
        # extData['roi_idx_le'] =  ""
        # extData['roi_name_le'] = ""
        # extData['intersect_x_le'] =  0
        # extData['intersect_y_le'] =  0
        # extData['intersect_z_le'] =  0
        # extData['roi_idx_re'] =  ""
        # extData['roi_name_re'] = ""
        # extData['intersect_x_re'] =  0
        # extData['intersect_y_re'] =  0
        # extData['intersect_z_re'] =  0

        for tindex in extData.index.values:
            print(tindex, "번째 index, frameID = ", extData.loc[tindex, 'f_frame_counter_left_camera'], '\n')
            headDir_l = np.array((extData.loc[tindex, 'headDir_X_L'], extData.loc[tindex, 'headDir_Y_L'],
                                  extData.loc[tindex, 'headDir_Z_L']))
            headDir_r = np.array((extData.loc[tindex, 'headDir_X_R'], extData.loc[tindex, 'headDir_Y_R'],
                                  extData.loc[tindex, 'headDir_Z_R']))
            headDir_mid = np.array((extData.loc[tindex, 'headDir_X_mid'], extData.loc[tindex, 'headDir_Y_mid'],
                                    extData.loc[tindex, 'headDir_Z_mid']))

            # print(tindex, 'headDir_l',headDir_l)
            headPos = np.array((extData.loc[tindex, 'HSVL_MS_S_Head_Pos_Veh_X'],
                                extData.loc[tindex, 'HSVL_MS_S_Head_Pos_Veh_Y'],
                                extData.loc[tindex, 'HSVL_MS_S_Head_Pos_Veh_Z']))
            headPos_LE = np.array((extData.loc[tindex, 'MS_S_Gaze_LE_Center_X'],
                                   extData.loc[tindex, 'MS_S_Gaze_LE_Center_Y'],
                                   extData.loc[tindex, 'MS_S_Gaze_LE_Center_Z']))
            headPos_RE = np.array((extData.loc[tindex, 'MS_S_Gaze_RE_Center_X'],
                                   extData.loc[tindex, 'MS_S_Gaze_RE_Center_Y'],
                                   extData.loc[tindex, 'MS_S_Gaze_RE_Center_Z']))
            # headPos = headPos + np.array([ 0, -140 , -140]) #lip
            # headPos_LE = headPos_LE + np.array([ 0, 0, 0])
            # headPos_RE = headPos_RE + np.array([ 0, 0, 0])

            # np.array([ 9.78, -45.84, -47.19])
            # headRot = np.array((extData.loc[tindex, 'MS_S_Head_rot_X'], extData.loc[tindex, 'MS_S_Head_rot_Y'], extData.loc[tindex, 'MS_S_Head_rot_Z']))
            # rt_2 = np.dot(eulerAnglesToRotationMatrix(np.array([0, 0, math.pi])), np.array([1, 1, 1])).round(5)
            # rt = eulerAnglesToRotationMatrix(headRot * deg2Rad * rt_2)
            # headrot_unitvec = np.dot(rt, [1, 0, 0])

            max_target_id = "0"
            max_target_name = "UNKOWN"
            max_target_point = np.array([0, 0, 0])
            max_target_roi_x = 0
            max_target_roi_y = 0
            max_target_roi_score = 0

            for tidx in ret_ExtROI.index.values:
                # offset = 0
                # print(tidx)
                # print(ret_ExtROI['tID'], ret_ExtROI['tTargetName'], ret_ExtROI['ttop_left'], ret_ExtROI['ttop_right'], ret_ExtROI['tbottom_left'], ret_ExtROI['tbottom_right'])
                print(' ', tidx, ret_ExtROI['tID'][tidx], ret_ExtROI['tTargetName'][tidx])
                troi_id = ret_ExtROI["tID"][tidx]
                troi_name = ret_ExtROI["tTargetName"][tidx]

                if (troi_id == 7):
                    offset = 135 + 40 + 5 + 10 + 10 + 10 + 10 + 5
                elif (troi_id == 8):
                    offset = 170 + 40
                elif (troi_id == 9):
                    offset = 60 + 40 + 20 + 25 + 20 + 20 + 10 + 15
                elif (troi_id == 1):
                    offset = 80 + 0 + 5 + 10 + 20 + 10 + 20 + 20 + 15
                elif (troi_id == 3 or troi_id == 4):  # or troi_id == 5
                    offset = 80
                elif (troi_id == 5):
                    offset = 60 + 10
                elif (troi_id == 6):
                    offset = 25 + 15 + 15 + 20 + 20 + 10 + 20 + 20
                elif (troi_id == 14):
                    offset = 25 + 15 + 0 + 5
                elif (troi_id == 19):
                    offset = 25 + 15 + 0 + 5 + 0 + 0 + 5
                elif (troi_id == 16):
                    offset = 25 + 15 + 0 + 5 + 10 + 10
                elif (troi_id == 11):
                    offset = 25 + 15 + 0 + -5
                elif (troi_id == 10):
                    offset = 25 + 0 + 0 + 10 + 10 + 10 + 5
                else:
                    offset = errDist
                p0 = np.array(ret_ExtROI["ttop_left"][tidx]) * 1000 + np.array([0, -offset, offset])
                p1 = np.array(ret_ExtROI["ttop_right"][tidx]) * 1000 + np.array([0, offset, offset])
                p2 = np.array(ret_ExtROI["tbottom_left"][tidx]) * 1000 + np.array([0, -offset, -offset])
                p3 = np.array(ret_ExtROI["tbottom_right"][tidx]) * 1000 + np.array([0, offset, -offset])

                camPlaneOrthVector = np.cross((p3 - p1), (p2 - p3)) / np.linalg.norm(np.cross((p3 - p1), (p2 - p3)))
                pointOnPlan = (p0 + p1 + p2 + p3) / 4

                # #최단거리
                # d = np.dot(camPlaneOrthVector, p0)
                # t = -(np.dot(origin,camPlaneOrthVector)+d)/ np.dot(gazeVector,camPlaneOrthVector)
                # pResult = origin + np.dot(t,gazeVector)
                # print('d',d)
                # print('t',t)
                # print('pResult',pResult)
                print(' ', "p0", p0, "\n  p1", p1, "\n  p2", p2, "\n  p3", p3)
                print(' ', 'headPos', headPos)
                print(' ', 'headDir_l', headDir_l)
                print(' ', 'camPlaneOrthVector', camPlaneOrthVector)
                print(' ', 'pointOnPlan', pointOnPlan)
                ret_check, point_mapping = self.calc_match_roi(p0, p1, p3, p2, camPlaneOrthVector, pointOnPlan,
                                                               headDir_mid, headPos)
                if (ret_check == True):
                    if (point_mapping[0] != 0 or point_mapping[1] != 0 or point_mapping[2] != 0):
                        target_roi_score = self.calc_score(pointOnPlan, p0,
                                                      [point_mapping[0], point_mapping[1], point_mapping[2]])
                        if (max_target_roi_score < target_roi_score):
                            max_target_roi_score = target_roi_score
                            max_target_id = str(troi_id)
                            max_target_name = troi_name
                            max_target_point = point_mapping
                            max_target_roi_x = int(
                                self.obj_mi.line_point_min_dist(point_mapping, p0, p2) / distance_xyz(p0, p1) * 100)
                            max_target_roi_y = int(
                                self.obj_mi.line_point_min_dist(point_mapping, p0, p1) / distance_xyz(p0, p2) * 100)

                # if (ret_check == True):
                #     extData.loc[tindex, 'roi_idx_h'] = str(
                #         troi_id)  # extData.loc[tindex, 'roi_idx_h'] +'/'+ str(troi_id)
                #     extData.loc[tindex, 'roi_name_h'] = extData.loc[tindex, 'roi_name_h'] + '/' + troi_name
                #     extData.loc[tindex, 'intersect_x_h'] = point_mapping[0]
                #     extData.loc[tindex, 'intersect_y_h'] = point_mapping[1]
                #     extData.loc[tindex, 'intersect_z_h'] = point_mapping[2]
                #     extData.loc[tindex, 'roi_X'] = int(
                #         self.obj_mi.line_point_min_dist(point_mapping, p0, p2) / distance_xyz(p0, p1) * 100)
                #     extData.loc[tindex, 'roi_Y'] = int(
                #         self.obj_mi.line_point_min_dist(point_mapping, p0, p1) / distance_xyz(p0, p2) * 100)
                    # ret_check_l, point_mapping_l = self.calc_match_roi(p0, p1, p3, p2, camPlaneOrthVector, pointOnPlan, headDir_l, headPos_LE)
                    # if(ret_check_l == True):
                    #     extData.loc[tindex, 'roi_idx_le'] = extData.loc[tindex, 'roi_idx_le'] +'#'+ str(troi_id)
                    #     extData.loc[tindex, 'roi_name_le'] = extData.loc[tindex, 'roi_name_le'] +'#'+ troi_name
                    #     extData.loc[tindex, 'intersect_x_le'] = point_mapping_l[0]
                    #     extData.loc[tindex, 'intersect_y_le'] = point_mapping_l[1]
                    #     extData.loc[tindex, 'intersect_z_le'] = point_mapping_l[2]
                    # ret_check_r, point_mapping_r = self.calc_match_roi(p0, p1, p3, p2, camPlaneOrthVector, pointOnPlan, headDir_r, headPos_RE)
                    # if(ret_check_r == True):
                    #     extData.loc[tindex, 'roi_idx_re'] = extData.loc[tindex, 'roi_idx_re'] +'*'+ str(troi_id)
                    #     extData.loc[tindex, 'roi_name_re'] = extData.loc[tindex, 'roi_name_re'] +'*'+ troi_name
                    #     extData.loc[tindex, 'intersect_x_re'] = point_mapping_r[0]
                    #     extData.loc[tindex, 'intersect_y_re'] = point_mapping_r[1]
                    #     extData.loc[tindex, 'intersect_z_re'] = point_mapping_r[2]
                    break
                print('\n')

            extData.loc[tindex, 'roi_idx_h'] = max_target_id
            extData.loc[tindex, 'roi_name_h'] = max_target_name
            extData.loc[tindex, 'intersect_x_h'] = max_target_point[0]
            extData.loc[tindex, 'intersect_y_h'] = max_target_point[1]
            extData.loc[tindex, 'intersect_z_h'] = max_target_point[2]
            extData.loc[tindex, 'roi_X'] = max_target_roi_x
            extData.loc[tindex, 'roi_Y'] = max_target_roi_y
            extData.loc[tindex, 'roi_score'] = max_target_roi_score

        return extData

    def check_match_roi_cto(self, extData, ret_ExtROI, errDist=0):
        extData['roi_idx_h'] = ""
        extData['roi_name_h'] = ""
        extData['roi_X'] = 0
        extData['roi_Y'] = 0
        extData['intersect_x_h'] = 0
        extData['intersect_y_h'] = 0
        extData['intersect_z_h'] = 0
        # extData['roi_idx_le'] =  ""
        # extData['roi_name_le'] = ""
        # extData['intersect_x_le'] =  0
        # extData['intersect_y_le'] =  0
        # extData['intersect_z_le'] =  0
        # extData['roi_idx_re'] =  ""
        # extData['roi_name_re'] = ""
        # extData['intersect_x_re'] =  0
        # extData['intersect_y_re'] =  0
        # extData['intersect_z_re'] =  0
        extData['roi_score'] = ""
        extData['max_roi_idx_h'] = ""
        extData['max_roi_score'] = ""

        for tindex in extData.index.values:
            print(tindex, "번째 index, frameID = ", extData.loc[tindex, 'f_frame_counter_left_camera'], '\n')
            headDir_l = np.array((extData.loc[tindex, 'headDir_X_L'], extData.loc[tindex, 'headDir_Y_L'],
                                  extData.loc[tindex, 'headDir_Z_L']))
            headDir_r = np.array((extData.loc[tindex, 'headDir_X_R'], extData.loc[tindex, 'headDir_Y_R'],
                                  extData.loc[tindex, 'headDir_Z_R']))
            headDir_mid = np.array((extData.loc[tindex, 'headDir_X_mid'], extData.loc[tindex, 'headDir_Y_mid'],
                                    extData.loc[tindex, 'headDir_Z_mid']))

            # print(tindex, 'headDir_l',headDir_l)
            headPos = np.array((extData.loc[tindex, 'HSVL_MS_S_Head_Pos_Veh_X'],
                                extData.loc[tindex, 'HSVL_MS_S_Head_Pos_Veh_Y'],
                                extData.loc[tindex, 'HSVL_MS_S_Head_Pos_Veh_Z']))
            headPos_LE = np.array((extData.loc[tindex, 'MS_S_Gaze_LE_Center_X'],
                                   extData.loc[tindex, 'MS_S_Gaze_LE_Center_Y'],
                                   extData.loc[tindex, 'MS_S_Gaze_LE_Center_Z']))
            headPos_RE = np.array((extData.loc[tindex, 'MS_S_Gaze_RE_Center_X'],
                                   extData.loc[tindex, 'MS_S_Gaze_RE_Center_Y'],
                                   extData.loc[tindex, 'MS_S_Gaze_RE_Center_Z']))
            # headPos = headPos + np.array([ 0, -140 , -140]) #lip
            # headPos_LE = headPos_LE + np.array([ 0, 0, 0])
            # headPos_RE = headPos_RE + np.array([ 0, 0, 0])

            # np.array([ 9.78, -45.84, -47.19])
            # headRot = np.array((extData.loc[tindex, 'MS_S_Head_rot_X'], extData.loc[tindex, 'MS_S_Head_rot_Y'], extData.loc[tindex, 'MS_S_Head_rot_Z']))
            # rt_2 = np.dot(eulerAnglesToRotationMatrix(np.array([0, 0, math.pi])), np.array([1, 1, 1])).round(5)
            # rt = eulerAnglesToRotationMatrix(headRot * deg2Rad * rt_2)
            # headrot_unitvec = np.dot(rt, [1, 0, 0])

            max_target_id = "0"
            max_target_name = "UNKOWN"
            max_target_point = np.array([0, 0, 0])
            max_target_roi_x = 0
            max_target_roi_y = 0
            max_target_roi_score = 0

            for tidx in ret_ExtROI.index.values:
                # offset = 0
                # print(tidx)
                # print(ret_ExtROI['tID'], ret_ExtROI['tTargetName'], ret_ExtROI['ttop_left'], ret_ExtROI['ttop_right'], ret_ExtROI['tbottom_left'], ret_ExtROI['tbottom_right'])
                print(' ', tidx, ret_ExtROI['tID'][tidx], ret_ExtROI['tTargetName'][tidx])
                troi_id = ret_ExtROI["tID"][tidx]
                troi_name = ret_ExtROI["tTargetName"][tidx]

                if (troi_id == 7):
                    offset = errDist + 30 + 20 + 10 #225 + 30 + 30 + 30
                elif (troi_id == 8):
                    offset = errDist + 30 + 30 + 20 + 20 + 30 + 30 #210 + 30 + 30 + 30
                elif (troi_id == 9):
                    offset = errDist - 60 #210
                elif (troi_id == 1):
                    offset = errDist - 100 - 20#180 - 30 - 50 - 30
                elif (troi_id == 3 or troi_id == 4):  # or troi_id == 5
                    offset = errDist + 20 + 20 + 10 + 20 + 30 + 70 + 30#80 + 30 + 50
                elif (troi_id == 5):
                    offset = errDist -100  #70 - 20 - 10
                elif (troi_id == 6):
                    offset = errDist #145
                elif (troi_id == 14):
                    offset = errDist #45
                elif (troi_id == 19):
                    offset = errDist #50
                elif (troi_id == 16):
                    offset = errDist #65
                elif (troi_id == 11):
                    offset = errDist #35s
                elif (troi_id == 10):
                    offset = errDist #50
                elif (troi_id == 15):
                    offset = errDist - 30 #0
                elif (troi_id == 13):
                    offset = errDist - 100 -20# 0
                else:
                    offset = errDist
                p0 = np.array(ret_ExtROI["ttop_left"][tidx]) * 1000 + np.array([0, -offset, offset])
                p1 = np.array(ret_ExtROI["ttop_right"][tidx]) * 1000 + np.array([0, offset, offset])
                p2 = np.array(ret_ExtROI["tbottom_left"][tidx]) * 1000 + np.array([0, -offset, -offset])
                p3 = np.array(ret_ExtROI["tbottom_right"][tidx]) * 1000 + np.array([0, offset, -offset])

                camPlaneOrthVector = np.cross((p3 - p1), (p2 - p3)) / np.linalg.norm(np.cross((p3 - p1), (p2 - p3)))
                pointOnPlan = (p0 + p1 + p2 + p3) / 4

                # #최단거리
                # d = np.dot(camPlaneOrthVector, p0)
                # t = -(np.dot(origin,camPlaneOrthVector)+d)/ np.dot(gazeVector,camPlaneOrthVector)
                # pResult = origin + np.dot(t,gazeVector)
                # print('d',d)
                # print('t',t)
                # print('pResult',pResult)
                print(' ', "p0", p0, "\n  p1", p1, "\n  p2", p2, "\n  p3", p3)
                print(' ', 'headPos', headPos)
                print(' ', 'headDir_l', headDir_l)
                print(' ', 'camPlaneOrthVector', camPlaneOrthVector)
                print(' ', 'pointOnPlan', pointOnPlan)
                ret_check, point_mapping = self.calc_match_roi(p0, p1, p3, p2, camPlaneOrthVector, pointOnPlan,
                                                               headDir_mid, headPos)
                if (ret_check == True):
                    if (point_mapping[0] != 0 or point_mapping[1] != 0 or point_mapping[2] != 0):
                        target_roi_score = self.calc_score(pointOnPlan, p0, [point_mapping[0], point_mapping[1], point_mapping[2]])

                        extData.loc[tindex, 'roi_idx_h'] = str(extData.loc[tindex, 'roi_idx_h']) + '/' + str(troi_id)
                        extData.loc[tindex, 'roi_name_h'] = extData.loc[tindex, 'roi_name_h'] + '/' + troi_name
                        extData.loc[tindex, 'intersect_x_h'] = str(extData.loc[tindex, 'intersect_x_h']) + '/' + str(np.round(point_mapping[0], 1))
                        extData.loc[tindex, 'intersect_y_h'] = str(extData.loc[tindex, 'intersect_y_h']) + '/' + str(np.round(point_mapping[1], 1))
                        extData.loc[tindex, 'intersect_z_h'] = str(extData.loc[tindex, 'intersect_z_h']) + '/' + str(np.round(point_mapping[2], 1))
                        extData.loc[tindex, 'roi_X'] = str(extData.loc[tindex, 'roi_X']) + '/' + str(int(self.obj_mi.line_point_min_dist(point_mapping, p0, p2) / distance_xyz(p0, p1) * 100))
                        extData.loc[tindex, 'roi_Y'] = str(extData.loc[tindex, 'roi_Y']) + '/' + str(int(self.obj_mi.line_point_min_dist(point_mapping, p0, p1) / distance_xyz(p0, p2) * 100))
                        extData.loc[tindex, 'roi_score'] = str(extData.loc[tindex, 'roi_score']) + '/' + str(target_roi_score)

                        if (max_target_roi_score < target_roi_score):
                            max_target_roi_score = target_roi_score
                            max_target_id = str(troi_id)

                        # target_roi_score = self.calc_score(pointOnPlan, p0,
                        #                                    [point_mapping[0], point_mapping[1], point_mapping[2]])
                        # if (max_target_roi_score < target_roi_score):
                        #     max_target_roi_score = target_roi_score
                        #     max_target_id = str(troi_id)
                        #     max_target_name = troi_name
                        #     max_target_point = point_mapping
                        #     max_target_roi_x = int(
                        #         self.obj_mi.line_point_min_dist(point_mapping, p0, p2) / distance_xyz(p0, p1) * 100)
                        #     max_target_roi_y = int(
                        #         self.obj_mi.line_point_min_dist(point_mapping, p0, p1) / distance_xyz(p0, p2) * 100)


                    # break
                print('\n')

            # extData.loc[tindex, 'roi_idx_h'] = max_target_id
            # extData.loc[tindex, 'roi_name_h'] = max_target_name
            # extData.loc[tindex, 'intersect_x_h'] = max_target_point[0]
            # extData.loc[tindex, 'intersect_y_h'] = max_target_point[1]
            # extData.loc[tindex, 'intersect_z_h'] = max_target_point[2]
            # extData.loc[tindex, 'roi_X'] = max_target_roi_x
            # extData.loc[tindex, 'roi_Y'] = max_target_roi_y
            # extData.loc[tindex, 'roi_score'] = max_target_roi_score
            extData.loc[tindex, 'max_roi_idx_h'] = max_target_id
            extData.loc[tindex, 'max_roi_score'] = max_target_roi_score

        return extData

    def rendering_roi_with_head_gaze(self, pROI, extData, nMax = -1):

        fig = plt.figure(figsize=(10,8))
        ax3 = fig.add_subplot(111, projection='3d')

        plt.title('3D Target ROI')
        # for i in pROI:
        #     print(i)

        for i in pROI.index[0:nMax]:
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
    print("\n\n\n make_gaze test/////////////////////")
    if(0):
        sys.stdout = open('DebugLog.txt', 'w')
    if(1):
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

        inputPath_ROI = "./refer/roi_config.json"
        # roi_config.json
        obj = make_gaze_and_roi()
        ret_roi = obj.load_jsonfile_ROI(inputPath_ROI)

        ret_ExtROI = obj.extract_availData_from_3D_target_ROI(ret_roi)

        ret_ExtGT = obj.extract_availData_from_GT(inputPath_GT)
        # print('ret_ExtGT\n\n', ret_ExtGT)

        ret_ExtGT_with_direction = obj.retcalcuate_head_eye_direction(ret_ExtGT)

        print('\n\n',ret_ExtGT_with_direction)
        ret_match = obj.check_match_roi(ret_ExtGT_with_direction, ret_ExtROI, 150)
        # obj.save_csvfile(ret_match, "./filename.csv")
        # ret_match.to_csv("filename.csv", mode='w', index=False, header=False, sep=',', quotechar=" ",
        #                  float_format='%.4f')

        obj.rendering_roi_with_head_gaze(ret_ExtROI, ret_match)
        # ret_match.to_csv("filename.csv", mode='w', index=False, header=False, sep=',', quotechar=" ", float_format='%.4f')



