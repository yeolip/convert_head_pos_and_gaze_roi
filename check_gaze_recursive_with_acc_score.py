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

def eulerAnglesToRotationMatrix(theta):
    R_x = np.array([[1, 0, 0],
        [0, math.cos(theta[0]), -math.sin(theta[0])],
        [0, math.sin(theta[0]), math.cos(theta[0])]])
    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
        [0, 1, 0],
        [-math.sin(theta[1]), 0, math.cos(theta[1])]])
    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
        [math.sin(theta[2]), math.cos(theta[2]), 0],
        [0, 0, 1]])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R

def rotationMatrixToEulerAngles(R):
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])

def intersectionWithPlan(linePoint, lineDir, planOrth, planPoint):
    d = np.dot(np.subtract(linePoint, planPoint), planOrth) / (np.dot(lineDir, planOrth))
    intersectionPoint = np.subtract(np.multiply(d, lineDir), linePoint)
    return intersectionPoint

def changeAxis_opencv2daimler(nR, nT):
    #opencv     x    y    z axis-> daimler   -y    -z   x
    #opencv  pitch, yaw, roll   -> daimler -pitch -yaw roll
    #daimler    x    y    z axis-> opencv   z   -x     -y
    #daimler roll, pitch, yaw   -> opencv roll -pitch  -yaw
    ret_nT = np.zeros((3, 1))
    ret_nR = np.zeros((3, 1))

    ret_nT[0] = nT[2]
    ret_nT[1] = -nT[0]
    ret_nT[2] = -nT[1]

    ret_nR[0] = nR[2]
    ret_nR[1] = -nR[0]
    ret_nR[2] = -nR[1]

    return ret_nR, ret_nT

def changeAxis_daimler2opencv(nR, nT):
    #opencv     x    y    z axis-> daimler   -y    -z   x
    #opencv  pitch, yaw, roll   -> daimler -pitch -yaw roll
    #daimler    x    y    z axis-> opencv   z   -x     -y
    #daimler roll, pitch, yaw   -> opencv roll -pitch  -yaw
    ret_nT = np.zeros((3, 1))
    ret_nR = np.zeros((3, 1))

    ret_nT[0] = -nT[1]
    ret_nT[1] = -nT[2]
    ret_nT[2] = nT[0]

    ret_nR[0] = -nR[1]
    ret_nR[1] = -nR[2]
    ret_nR[2] = nR[0]

    return ret_nR, ret_nT

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

    # available_df = pd.DataFrame({tframeId,theadPos3D, theadPos3D, theadOri,tisFaceDetected }, index=tframeId, columns=available_columns)  # index 지정

    # available_df = pd.DataFrame({'frameId':tframeId,"theadPos3D":theadPos3D}, columns=["frameId","theadPos3D"])  # index 지정


    #
    # fig = plt.figure(figsize=(10,8))
    # ax3 = fig.add_subplot(111, projection='3d')
    # plt.title('3D Target ROI')
    #
    # for i in tValid[:][0:-1]:
    #     print(i[2][0][0], i[2][0][1], i[2][0][2], '-----')
    #     x0 = i[2][0][0] * 1000
    #     y0 = i[2][0][1] * 1000
    #     z0 = i[2][0][2] * 1000
    #     # ax3.scatter(xs=x, ys=y, zs=z, label=i[1])
    #     x1 = i[3][0][0] * 1000
    #     y1 = i[3][0][1] * 1000
    #     z1 = i[3][0][2] * 1000
    #     # ax3.scatter(xs=x, ys=y, zs=z, label=i[1])
    #     x2 = i[4][0][0] * 1000
    #     y2 = i[4][0][1] * 1000
    #     z2 = i[4][0][2] * 1000
    #     # ax3.scatter(xs=x, ys=y, zs=z, label=i[1])
    #     x3 = i[5][0][0] * 1000
    #     y3 = i[5][0][1] * 1000
    #     z3 = i[5][0][2] * 1000
    #     ax3.scatter(xs=[x0,x1,x3,x2], ys=[y0,y1,y3,y2], zs=[z0,z1,z3,z2] )
    #     ax3.plot([x0,x1,x3,x2,x0], [y0,y1,y3,y2,y0], [z0,z1,z3,z2,z0], label=i[1])
    #
    # ax3.set_zlim(-1000, 1000)
    # ax3.set_title("3D Target ROI")
    # ax3.set_xlabel('veh X', fontsize=16)
    # ax3.set_ylabel('veh Y', fontsize=16)
    # ax3.set_zlabel('veh Z', fontsize=16)
    # ax3.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    # ax3.view_init(5, -10)
    #
    # plt.grid(True)
    # plt.show()

    available_df = pd.DataFrame(available_dict)  # index 지정
    print(available_df)
    return available_df

def extract_availData_from_GT_short(inputPath_GT):
    print("//////////", funcname(), "//////////")
    extGT = pd.read_csv(inputPath_GT)
    df_extGT = extGT[[
                          'HSVL_MS_S_Head_Pos_Veh_X', 'HSVL_MS_S_Head_Pos_Veh_Y', 'HSVL_MS_S_Head_Pos_Veh_Z',
                          'MS_S_Head_rot_X', 'MS_S_Head_rot_Y', 'MS_S_Head_rot_Z',
                          'MS_S_Gaze_rot_X',       'MS_S_Gaze_rot_Y',     'MS_S_Gaze_rot_Z',
                          'f_frame_counter_left_camera', 'HSVL_MS_CAN_S_Head_tracking_status',
                          'MS_S_Gaze_ROI_X_Raw', 'MS_S_Gaze_ROI_Y_Raw', 'CAN_S_Gaze_ROI', 'gt_s_gaze_roi_das'
                    ]]
    df_extGT = df_extGT.dropna()
    # print('ret_ExtGT\n\n',ret_ExtGT)
    print('df_extGT\n\n', df_extGT)
    return df_extGT

def retcalcuate_head_eye_direction_short(extData):
    print("//////////", funcname(), "//////////")
    # print(extData)

    extData['gt_name_gaze_roi'] = ""
    extData['headDir_X_mid'] = 0
    extData['headDir_Y_mid'] = 0
    extData['headDir_Z_mid'] = 0
    # extData['headDir_X_mid_rad'] = 0
    # extData['headDir_Y_mid_rad'] = 0
    # extData['headDir_Z_mid_rad'] = 0

    for tindex in extData.index.values:
        print(tindex)
        tframecnt = extData.loc[tindex, 'f_frame_counter_left_camera']
        # troi_result = extData.loc[tindex, 'gt_s_gaze_roi_das']
        # troi_x = extData.loc[tindex, 'MS_S_Gaze_ROI_X_Raw']
        # troi_y = extData.loc[tindex, 'MS_S_Gaze_ROI_Y_Raw']
        thead_pos = np.array((extData.loc[tindex, 'HSVL_MS_S_Head_Pos_Veh_X'], extData.loc[tindex, 'HSVL_MS_S_Head_Pos_Veh_Y'], extData.loc[tindex, 'HSVL_MS_S_Head_Pos_Veh_Z']))
        thead_rot = np.array((extData.loc[tindex, 'MS_S_Head_rot_X'], extData.loc[tindex, 'MS_S_Head_rot_Y'], extData.loc[tindex, 'MS_S_Head_rot_Z']))
        tgaze_vec_mid = np.array((extData.loc[tindex, 'MS_S_Gaze_rot_X'], extData.loc[tindex, 'MS_S_Gaze_rot_Y'], extData.loc[tindex, 'MS_S_Gaze_rot_Z']))

        # print(tframecnt, troi_result, troi_x, troi_y, thead_pos, thead_rot,  tgaze_vec_mid)
        print(tframecnt, thead_pos, thead_rot,  tgaze_vec_mid)
        # print(tframecnt, troi_result, troi_x, troi_y)

        headPos3D_mm = thead_pos
        headOri_radian = thead_rot * deg2Rad
        print("headPos3D_mm", headPos3D_mm)
        print("headOri_radian", headOri_radian)

        mideye_roll_pitch_yaw_rad = tgaze_vec_mid * deg2Rad
        # print("eulerAngles\n", eulerAnglesToRotationMatrix(lpupil_roll_pitch_yaw_rad))

        rt_2 = np.dot(eulerAnglesToRotationMatrix(np.array([0, 0, math.pi])), np.array([1, 1, 1])).round(5)
        rt = eulerAnglesToRotationMatrix(headOri_radian * rt_2)

        rot2_mid = eulerAnglesToRotationMatrix(mideye_roll_pitch_yaw_rad)

        print('rot2_mid',rot2_mid)
        headDir_mid = np.dot(rot2_mid, np.dot(rt, [1, 0, 0]))
        # headDir_mid = np.dot(np.dot(rot2_mid , rt), [1,0,0])
        print('headDir_mid',headDir_mid)
        extData.loc[tindex, 'headDir_X_mid'] = headDir_mid[0]
        extData.loc[tindex, 'headDir_Y_mid'] = headDir_mid[1]
        extData.loc[tindex, 'headDir_Z_mid'] = headDir_mid[2]

        # ccc = changeRotation_unitvec2radian('RPY', headDir_mid, 'RPY') * rad2Deg
        # print('ccc',ccc)
        # extData.loc[tindex,'headDir_X_mid_rad'] = ccc[0]
        # extData.loc[tindex,'headDir_Y_mid_rad'] = ccc[1]
        # extData.loc[tindex,'headDir_Z_mid_rad'] = ccc[2]

    return extData

def calc_match_roi(p0, p1, p3, p2, camPlaneOrthVector, pointOnPlan, headDir, headPos):

    # tview_point = intersectionWithPlan(headPos, headDir, camPlaneOrthVector, pointOnPlan)
    # print(' ', 'tview_point', tview_point)
    tview_point2 = bcheck_match.line_plane_collision(camPlaneOrthVector, pointOnPlan, headDir, headPos)
    print("tview_point2", tview_point2)

    ret_match = bcheck_match.check_available_point_on_plane(p0, p1, p3, p2, tview_point2)
    head_vector = np.dot(eulerAnglesToRotationMatrix(np.array([0, 0, math.pi])), headDir).round(5)
    ret_sameDirect = bcheck_match.is_same_direction(tview_point2, head_vector, headPos)
    print("ret_match", ret_match, 'ret_sameDirect', ret_sameDirect)
    # print('각도', changeRotation_unitvec2radian_check2('RPY', headDir, 'RPY') * rad2Deg)
    if (ret_match == True and ret_sameDirect == True):
        return True, tview_point2
    elif (ret_match == False and ret_sameDirect == True):
        return False, tview_point2
    return False, np.array([0,0,0])

def calc_score(roi_center, roi_one_of_points, target_point):
    # calc 1 - (dist(target_point - roi_center) / dist(roi_one_of_points - roi_center))
    dist1 = distance_xyz(roi_center, target_point)
    dist2 = distance_xyz(roi_center, roi_one_of_points)
    result = np.round(1 - (dist1/dist2),2)
    return result

def check_match_roi(extData, ret_ExtROI, errDist = 0):
    extData['roi_idx_h'] =  ""
    extData['roi_name_h'] = ""
    extData['roi_X'] = ""
    extData['roi_Y'] = ""
    extData['intersect_x_h'] =  ""
    extData['intersect_y_h'] =  ""
    extData['intersect_z_h'] =  ""
    extData['roi_score'] =  ""
    extData['max_target_id'] =  ""
    extData['max_roi_score'] =  ""


    for tindex in extData.index.values:
        print(tindex,"번째 index, frameID = ", extData.loc[tindex, 'f_frame_counter_left_camera'],'\n')
        headDir_mid = np.array((extData.loc[tindex, 'headDir_X_mid'], extData.loc[tindex, 'headDir_Y_mid'], extData.loc[tindex, 'headDir_Z_mid']))
        headPos = np.array((extData.loc[tindex, 'HSVL_MS_S_Head_Pos_Veh_X'], extData.loc[tindex, 'HSVL_MS_S_Head_Pos_Veh_Y'], extData.loc[tindex, 'HSVL_MS_S_Head_Pos_Veh_Z']))
        roi_idx_origin = extData.loc[tindex, 'gt_s_gaze_roi_das']

        for tidx2 in ret_ExtROI.index.values:
            troi_id = ret_ExtROI["tID"][tidx2]
            troi_name = ret_ExtROI["tTargetName"][tidx2]
            if(roi_idx_origin == troi_id):
                extData.loc[tindex, 'gt_name_gaze_roi'] = troi_name
                break
        max_target_id = "0"
        max_target_name = "UNKOWN"
        max_target_point = np.array([0,0,0])
        max_target_roi_x = 0
        max_target_roi_y = 0
        max_target_roi_score = 0

        for tidx in ret_ExtROI.index.values:
            offset = 0
            # print(tidx)
            # print(ret_ExtROI['tID'], ret_ExtROI['tTargetName'], ret_ExtROI['ttop_left'], ret_ExtROI['ttop_right'], ret_ExtROI['tbottom_left'], ret_ExtROI['tbottom_right'])
            print(' ',tidx, ret_ExtROI['tID'][tidx], ret_ExtROI['tTargetName'][tidx])
            troi_id = ret_ExtROI["tID"][tidx]
            troi_name = ret_ExtROI["tTargetName"][tidx]
            # if (troi_id == 7):
            #     offset = 300
            # elif (troi_id == 8):
            #     offset = 400
            # elif (troi_id == 4 or troi_id == 5 or troi_id == 6):
            #     offset = 50
            # elif(troi_id == 9 or troi_id == 1):
            #     offset = 100
            # elif(troi_id == 10):
            #     offset = 50
            # elif(troi_id == 11 or troi_id == 12 or troi_id == 19):
            #     offset = 400
            # elif (troi_id == 14 or troi_id == 16):
            #     offset = 50
            # elif (troi_id == 15 or troi_id == 17):
            #     offset = 50
            # else:
            #     offset = 0
            p0 = np.array(ret_ExtROI["ttop_left"][tidx]) * 1000  + np.array([0,-offset,offset])
            p1 = np.array(ret_ExtROI["ttop_right"][tidx]) * 1000 + np.array([0,offset,offset])
            p2 = np.array(ret_ExtROI["tbottom_left"][tidx]) * 1000 + np.array([0,-offset,-offset])
            p3 = np.array(ret_ExtROI["tbottom_right"][tidx]) * 1000 + np.array([0,offset,-offset])

            camPlaneOrthVector = np.cross((p3 - p1), (p2 - p3))/np.linalg.norm(np.cross((p3 - p1), (p2 - p3)))
            pointOnPlan = (p0+p1+p2+p3)/4

            print(' ',"p0", p0, "\n  p1",p1, "\n  p2", p2, "\n  p3", p3)
            print(' ','headPos',headPos)
            print(' ','headDir_mid', headDir_mid)
            print(' ','camPlaneOrthVector',camPlaneOrthVector)
            print(' ','pointOnPlan',pointOnPlan)
            ret_check, point_mapping = calc_match_roi(p0, p1, p3, p2, camPlaneOrthVector, pointOnPlan, headDir_mid, headPos)
            # if(point_mapping[0] != 0 or point_mapping[1] != 0 or point_mapping[2] != 0):
            #     target_roi_score = calc_score(pointOnPlan, p0, [point_mapping[0], point_mapping[1], point_mapping[2]])
            #     if(max_target_roi_score  < target_roi_score):
            #         max_target_roi_score = target_roi_score
            #         max_target_id = str(troi_id)
            #         max_target_name = troi_name
            #         max_target_point = point_mapping
            #         max_target_roi_x = int(bcheck_match.line_point_min_dist(point_mapping, p0, p2) / distance_xyz(p0, p1) * 100)
            #         max_target_roi_y = int(bcheck_match.line_point_min_dist(point_mapping, p0, p1) / distance_xyz(p0,p2) * 100)
            # if(ret_check == True):
            #     extData.loc[tindex, 'roi_idx_h'] = str(extData.loc[tindex, 'roi_idx_h']) + '/' + str(troi_id)
            #     extData.loc[tindex, 'roi_name_h'] = extData.loc[tindex, 'roi_name_h'] + '/' + troi_name
            #     # extData.loc[tindex, 'roi_name_h'] = troi_name
            #     extData.loc[tindex, 'intersect_x_h'] = point_mapping[0]
            #     extData.loc[tindex, 'intersect_y_h'] = point_mapping[1]
            #     extData.loc[tindex, 'intersect_z_h'] = point_mapping[2]
            #     extData.loc[tindex, 'roi_X'] = int(bcheck_match.line_point_min_dist(point_mapping, p0, p2) / distance_xyz(p0, p1) * 100)
            #     extData.loc[tindex, 'roi_Y'] = int(bcheck_match.line_point_min_dist(point_mapping, p0, p1) / distance_xyz(p0,p2) * 100)
            #     extData.loc[tindex, 'roi_score'] = str(extData.loc[tindex, 'roi_score']) + '/' +  str(calc_score(pointOnPlan, p0, [point_mapping[0], point_mapping[1], point_mapping[2]]))
            #     # break
            if(point_mapping[0] != 0 or point_mapping[1] != 0 or point_mapping[2] != 0):
                extData.loc[tindex, 'roi_idx_h'] = str(extData.loc[tindex, 'roi_idx_h']) + '/' + str(troi_id)
                extData.loc[tindex, 'roi_name_h'] = extData.loc[tindex, 'roi_name_h'] + '/' + troi_name
                extData.loc[tindex, 'intersect_x_h'] = str(extData.loc[tindex, 'intersect_x_h']) + '/' + str(np.round(point_mapping[0],1))
                extData.loc[tindex, 'intersect_y_h'] = str(extData.loc[tindex, 'intersect_y_h']) + '/' + str(np.round(point_mapping[1],1))
                extData.loc[tindex, 'intersect_z_h'] = str(extData.loc[tindex, 'intersect_z_h']) + '/' + str(np.round(point_mapping[2],1))
                extData.loc[tindex, 'roi_X'] = str(extData.loc[tindex, 'roi_X']) + '/' + str(int(bcheck_match.line_point_min_dist(point_mapping, p0, p2) / distance_xyz(p0, p1) * 100))
                extData.loc[tindex, 'roi_Y'] = str(extData.loc[tindex, 'roi_Y']) + '/' + str(int(bcheck_match.line_point_min_dist(point_mapping, p0, p1) / distance_xyz(p0,p2) * 100))
                extData.loc[tindex, 'roi_score'] = str(extData.loc[tindex, 'roi_score']) + '/' +  str(calc_score(pointOnPlan, p0, [point_mapping[0], point_mapping[1], point_mapping[2]]))

                target_roi_score = calc_score(pointOnPlan, p0, [point_mapping[0], point_mapping[1], point_mapping[2]])
                if(max_target_roi_score  < target_roi_score):
                    max_target_roi_score = target_roi_score
                    max_target_id = str(troi_id)

            print('\n')

        # extData.loc[tindex, 'roi_idx_h'] = max_target_id
        # extData.loc[tindex, 'roi_name_h'] = max_target_name
        # extData.loc[tindex, 'intersect_x_h'] = max_target_point[0]
        # extData.loc[tindex, 'intersect_y_h'] = max_target_point[1]
        # extData.loc[tindex, 'intersect_z_h'] = max_target_point[2]
        # extData.loc[tindex, 'roi_X'] = max_target_roi_x
        # extData.loc[tindex, 'roi_Y'] = max_target_roi_y
        # extData.loc[tindex, 'roi_score'] = max_target_roi_score
        extData.loc[tindex, 'max_target_id'] = max_target_roi_score
        extData.loc[tindex, 'max_roi_score'] = max_target_id
    return extData

def distance_xyz(a,b):
    temp = a - b
    dist = np.sqrt(temp[0] * temp[0] + temp[1] * temp[1] + temp[2] * temp[2])
    return dist

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

def policy_gaze_roi_accuracy(tdata, text_column):
    tdata['hit_green_roi'] = "None"
    tdata['hit_amber_roi'] = "None"
    roi_group_green = {1: [1], 3: [3,4], 4: [3,4], 5: [5,9], 6:[6], 7:[7], 8:[8], 9:[9], 10:[10], 11:[11,19],
                       12:[12, 13], 14:[14, 12], 15:[15], 16:[16], 17:[17], 19:[11]}

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

if __name__ == '__main__':
    print("\n\n\n test/////////////////////")
    if(0):
        sys.stdout = open('DebugLog.txt', 'w')

    fold_names = filedialog.askdirectory()
    files_to_replace_base = []

    if (1):
        for dirpath, dirnames, filenames in os.walk(fold_names):
            for filename in [f for f in filenames if f.endswith(".csv")]:
                files_to_replace_base.append(os.path.join(dirpath, filename))
                print(os.path.join(dirpath, filename))

                # if (filename.__contains__("GT_") == True):
                #     files_to_replace_base.append(os.path.join(dirpath, filename))
                #     print(os.path.join(dirpath, filename))
                # elif (filename.__contains__("DisplayCenter") == True):
                #     files_to_replace.append(os.path.join(dirpath, filename))
                #     print(os.path.join(dirpath, filename))

    print(len(files_to_replace_base))
    if(len(files_to_replace_base)== 0):
        print("No select file..!!!")
        print(1/0)
    # files_to_replace_target = files_to_replace_base.copy()
    print("*" * 50)
    print(sorted(files_to_replace_base,key=lambda x: str(x).split() ))
    print(files_to_replace_base)

    # print(1/0)

    inputPath_ROI = "./refer/roi_config_eva5.json"
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
            ret_match = check_match_roi(ret_ExtGT_with_direction, ret_ExtROI, 250)  # 150
            ret_match['Load_file'] =  os.path.basename(tname)
            # print('ret_match', ret_match)
            df_merge = pd.concat([df_merge, ret_match]).reset_index(drop=True)

        save_csvfile(df_merge, 'save_output.csv')
        print(1/0)
        print("df_merge",df_merge)
        df_merge = df_merge.astype({'roi_idx_h':"int64", "gt_s_gaze_roi_das":"int64"})
        df_merge['hit'] = (df_merge['roi_idx_h'] == df_merge['gt_s_gaze_roi_das'])

        print("***Final hit accuracy\nTrue={}개, Total={}개, {}%".format(df_merge['hit'].value_counts()[1], df_merge['hit'].size, df_merge['hit'].value_counts()[1]/df_merge['hit'].size*100))

        df_data2 = policy_gaze_roi_accuracy(df_merge, 'roi_idx_h')
        counting_gaze_roi(df_data2)
        save_csvfile(df_data2, 'save_output.csv')

        # save_csvfile(df_merge, "./accuracy_output.csv")

        # rendering_roi_with_head_gaze(ret_ExtROI, df_merge)

    # print(1/0)
