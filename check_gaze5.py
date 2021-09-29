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

import match_intersection as objmatch_roi

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

# converting from Cartesian Coordinates to Spherical Coordinates.
def changeRotation_pitchyaw2unitvec(typeIn, nR_eulerangle, typeOut ):
    print("//////////", funcname(), "//////////")
    up = np.array([0,0,1])
    print('')
    t_pitch_ang = 0
    t_yaw_ang = 0
    t_roll_ang = 0

    print(" Enter", typeIn, "return", typeOut)
    if (typeIn == "PYR"):  # Pitch / Yaw / Roll
        t_pitch_ang = nR_eulerangle[0]
        t_yaw_ang = nR_eulerangle[1]
        t_roll_ang = nR_eulerangle[2]
    elif (typeIn == "RPY"):  # Roll / Pitch / Yaw
        t_pitch_ang = nR_eulerangle[1]
        t_yaw_ang = nR_eulerangle[2]
        t_roll_ang = nR_eulerangle[0]
    else:
        print("Not support!!",1/0)

    beta_tilt = -t_pitch_ang
    alpha_yaw = t_yaw_ang
    gazeVector = [math.sin(alpha_yaw) * math.cos(beta_tilt), math.sin(beta_tilt),
                  math.cos(alpha_yaw) * math.cos(beta_tilt)]
    # gazeVector = lpupil_roll_pitch_yaw * deg2Rad

    if (typeOut == "PYR"):  # Pitch / Yaw / Roll
        t_x =  gazeVector[0]
        t_y = gazeVector[1]
        t_z = gazeVector[2]
    elif (typeOut == "RPY"):  # Roll / Pitch / Yaw
        t_x = gazeVector[2]
        t_y = gazeVector[0]
        t_z = gazeVector[1]
    else:
        print("Not support!!",1/0)

    print('gazeVector=',typeOut, np.array([t_x, t_y, t_z]))
    return np.array([t_x, t_y, t_z])


'''
This is related to the problem of converting from Cartesian Coordinates to Spherical Coordinates. Note that the reverse operation is not unique: there are many possible angle triplets that produce the same rotation transformation, so any function we choose will necessarily have to standardize on one option. This means an angle triplet might not necessarily round-trip convert to vectors and back as you expect, even though the result will be equivalent in effect when rotating vectors.

The details will vary a little based on the conventions you choose (eg. in what order are the angles in an angle triplet applied?) In your case it looks like your conventions are:

In a neutral rotation (0 degrees on all axes)...

forward = (1, 0 0)
right = (0, -1, 0)
up = (0, 0, 1)

(ie. this is a right-handed coordinate system, with x+ forward, y+ left, z+ up)

From this neutral rotation...

increasing YAW rotates the forward vector to the left
increasing PITCH rotates the forward vector downward
increasing ROLL rotates the up vector to the right
Rotations are applied in the order (from most local to most global)

Roll
Pitch
Yaw
'''
# https://gamedev.stackexchange.com/questions/172147/convert-3d-direction-vectors-to-yaw-pitch-roll-angles
def changeRotation_unitvec2radian_check2(typeIn, nR_unitvec, typeOut ):
    print("//////////", funcname(), "//////////")
    up = np.array([0,0,1])
    print('\n')
    t_pitch_vec = 0
    t_yaw_vec = 0
    t_roll_vec = 0

    print(" Enter", typeIn, "return", typeOut)
    if (typeIn == "PYR"):  # Pitch / Yaw / Roll
        t_pitch_vec = nR_unitvec[0]
        t_yaw_vec = nR_unitvec[1]
        t_roll_vec = nR_unitvec[2]
    elif (typeIn == "RPY"):  # Roll / Pitch / Yaw
        t_pitch_vec = nR_unitvec[1]
        t_yaw_vec = nR_unitvec[2]
        t_roll_vec = nR_unitvec[0]
    else:
        print("Not support!!",1/0)

    # Yaw is the bearing of the forward vector's shadow in the xy plane.
    # yaw = math.atan2(t_pitch_vec[1], t_roll_vec[0])
    print(t_pitch_vec, t_roll_vec)
    yaw = math.atan(t_pitch_vec/t_roll_vec)

    # Pitch is the altitude of the forward vector off the xy plane, toward the down direction.
    pitch = -math.asin(t_yaw_vec)
    # pitch2 = math.acos(t_yaw_vec/1)
    # print("--------------pitch", pitch*rad2Deg, 'pitch2',pitch2*rad2Deg)
    # Find the vector in the xy plane 90 degrees to the right of our bearing.

    planeRightX = math.sin(yaw)
    planeRightY = -math.cos(yaw)

    # Roll is the rightward lean of our up vector, computed here using a dot product.
    roll = math.asin(up[0]*planeRightX + up[1]*planeRightY)
    # If we're twisted upside-down, return a roll in the range +-(pi/2, pi)
    print('!!!!!!!roll////', roll * rad2Deg)
    if(up[2] < 0):
        roll = np.sign(roll) * math.pi - roll
    # Convert radians to degrees.
    # angles[YAW]   =   yaw * 180 / math.pi
    # angles[PITCH] = pitch * 180 / math.pi
    # angles[ROLL]  =  roll * 180 / math.pi

    if (typeOut == "PYR"):  # Pitch / Yaw / Roll
        t_x =  pitch
        t_y = yaw
        t_z = roll
    elif (typeOut == "RPY"):  # Roll / Pitch / Yaw
        t_x = roll
        t_y = pitch
        t_z = yaw
    else:
        print("Not support!!",1/0)

    return np.array([t_x, t_y, t_z])

def changeRotation_unitvec2radian_check(typeIn, nR_unitvec, typeOut ):
    print("//////////", funcname(), "//////////")

    # nR_unitvec[0] = 0.048387713730335236
    # nR_unitvec[1] = -0.30887135863304138
    # nR_unitvec[2] = -0.94987112283706665

    # alpha_yaw = math.atan(nR_unitvec[0] / nR_unitvec[2])
    # beta_tilt = math.acos(nR_unitvec[1])
    # print('yaw',math.atan(nR_unitvec[0] / nR_unitvec[2]), alpha_yaw*rad2Deg)
    # print('tilt',math.acos(nR_unitvec[1]), beta_tilt*rad2Deg)
    # print('r3', [math.sin(alpha_yaw)*math.sin(beta_tilt), math.cos(beta_tilt), math.cos(alpha_yaw)*math.sin(beta_tilt) ])
    # print("degree", ret2 * rad2Deg)
    print('\n')
    t_pitch_vec = 0
    t_yaw_vec = 0
    t_roll_vec = 0

    print(" Enter", typeIn, "return", typeOut)
    if (typeIn == "PYR"):  # Pitch / Yaw / Roll
        t_pitch_vec = nR_unitvec[0]
        t_yaw_vec = nR_unitvec[1]
        t_roll_vec = nR_unitvec[2]
    elif (typeIn == "RPY"):  # Roll / Pitch / Yaw
        t_pitch_vec = nR_unitvec[1]
        t_yaw_vec = nR_unitvec[2]
        t_roll_vec = nR_unitvec[0]
    else:
        print("Not support!!",1/0)

    alpha_yaw = math.atan(t_pitch_vec / t_roll_vec)
    beta_tilt = -math.asin(t_yaw_vec)
    print('yaw',math.atan(t_pitch_vec / t_roll_vec), alpha_yaw*rad2Deg)
    print('tilt',math.asin(t_yaw_vec), beta_tilt*rad2Deg)
    print('r3', [math.sin(alpha_yaw)*math.cos(beta_tilt), math.sin(beta_tilt), math.cos(alpha_yaw)*math.cos(beta_tilt) ])

    if (typeOut == "PYR"):  # Pitch / Yaw / Roll
        t_x =  beta_tilt
        t_y = alpha_yaw
        t_z = 0
    elif (typeOut == "RPY"):  # Roll / Pitch / Yaw
        t_x = 0
        t_y = beta_tilt
        t_z = alpha_yaw
    else:
        print("Not support!!",1/0)

    return np.array([t_x, t_y, t_z])
    # return np.array([alpha_yaw, beta_tilt, 0])

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

def extract_availData_from_GT(inputPath_GT):
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
    # df_extGT = df_extGT.dropna()
    # print('ret_ExtGT\n\n',ret_ExtGT)
    print('df_extGT\n\n', df_extGT)
    return df_extGT

def retcalcuate_head_eye_direction(extData):
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

    # 'CAN_S_Gaze_ROI', 'CAN_S_Gaze_ROI_X', 'CAN_S_Gaze_ROI_Y',
    # 'MS_S_Head_rot_X', 'MS_S_Head_rot_Y', 'MS_S_Head_rot_Z',
    # 'HSVL_MS_S_Head_Pos_Veh_X', 'HSVL_MS_S_Head_Pos_Veh_Y', 'HSVL_MS_S_Head_Pos_Veh_Z',
    # 'MS_S_Gaze_LE_Center_X', 'MS_S_Gaze_LE_Center_Y', 'MS_S_Gaze_LE_Center_Z',
    # 'MS_S_Gaze_LE_VA_rot_X', 'MS_S_Gaze_LE_VA_rot_Y', 'MS_S_Gaze_LE_VA_rot_Z',
    # 'MS_S_Gaze_RE_Center_X', 'MS_S_Gaze_RE_Center_Y', 'MS_S_Gaze_RE_Center_Z',
    # 'MS_S_Gaze_RE_VA_rot_X', 'MS_S_Gaze_RE_VA_rot_Y', 'MS_S_Gaze_RE_VA_rot_Z'

    for tindex in extData.index.values:
        # print(extData.irisHeight.loc[tindex])
        print(tindex)
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
        headOri_radian = np.array([0,0,0]) * deg2Rad
        print("headPos3D_mm", headPos3D_mm)
        print("headOri_radian", headOri_radian)

        lpupil_roll_pitch_yaw_rad = tgaze_vec_l * deg2Rad #np.array([0, -9.8, 0.2])
        rpupil_roll_pitch_yaw_rad = tgaze_vec_r * deg2Rad #np.array([0, -9.8, 0.2])
        mideye_roll_pitch_yaw_rad = tgaze_vec_mid * deg2Rad
        # print("eulerAngles\n", eulerAnglesToRotationMatrix(lpupil_roll_pitch_yaw_rad))

        # print('lpupil_roll_pitch_yaw_deg', lpupil_roll_pitch_yaw_rad * rad2Deg)
        # aaa = changeRotation_pitchyaw2unitvec('RPY',lpupil_roll_pitch_yaw_rad ,'RPY')
        # print('---gaze_vector', aaa)
        # bbb = changeRotation_unitvec2radian_check('RPY', aaa, 'RPY')
        # print('---radian ', bbb*rad2Deg)
        # bbb2 = changeRotation_unitvec2radian_check2('RPY', aaa, 'RPY')
        # print('---radian2 ', bbb2*rad2Deg)

        # beta_tilt = lpupil_roll_pitch_yaw[1] * deg2Rad
        # alpha_yaw = lpupil_roll_pitch_yaw[2] * deg2Rad
        # gazeVector = [math.sin(alpha_yaw) * math.cos(beta_tilt), math.sin(beta_tilt),
        #                    math.cos(alpha_yaw) * math.cos(beta_tilt)]
        # # gazeVector = lpupil_roll_pitch_yaw * deg2Rad
        # print('gazeVector_pitch_yaw_roll', gazeVector)
        #
        # # origin = tgaze_pos_l
        # origin = (headPos3D_mm)
        #
        # pitch_yaw_roll = changeRotation_unitvec2radian_check2('PYR',gazeVector,'PYR')
        # _, roll_pitch_yaw = changeAxis_opencv2daimler(np.array([0, 0, 0]), pitch_yaw_roll)
        # print('roll_pitch_yaw', roll_pitch_yaw.T)
        #
        # print('pitch_yaw_roll', pitch_yaw_roll * rad2Deg)
        # print('roll_pitch_yaw', roll_pitch_yaw * rad2Deg,'\n')
        # roll_pitch_yaw = lpupil_roll_pitch_yaw * deg2Rad


        # rt1 = eulerAnglesToRotationMatrix(np.array([0, 0, 0]))
        # rt = rt1
        # rot2 = np.eye(3)
        rt_2 = np.dot(eulerAnglesToRotationMatrix(np.array([0, 0, math.pi])), np.array([1, 1, 1])).round(5)
        rt = eulerAnglesToRotationMatrix(headOri_radian * rt_2 * np.array([1, 1, 1]))

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

        # headDir_l = np.dot(np.dot(rot2_l , rt), [1,0,0])
        # headDir_r = np.dot(np.dot(rot2_r , rt), [1,0,0])
        # headDir_mid = np.dot(np.dot(rot2_mid , rt), [1,0,0])
        headDir_l = np.dot(np.dot(rt,rot2_l ), [1,0,0])
        headDir_r = np.dot(np.dot(rt,rot2_r ), [1,0,0])
        headDir_mid = np.dot(np.dot(rt,rot2_mid), [1,0,0])
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
        ccc = changeRotation_unitvec2radian_check2('RPY', headDir_mid, 'RPY') * rad2Deg
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

        max_target_id = "0"
        max_target_roi_score = 0

        for tidx in ret_ExtROI.index.values:
            offset = 0
            # print(tidx)
            # print(ret_ExtROI['tID'], ret_ExtROI['tTargetName'], ret_ExtROI['ttop_left'], ret_ExtROI['ttop_right'], ret_ExtROI['tbottom_left'], ret_ExtROI['tbottom_right'])
            print(' ',tidx, ret_ExtROI['tID'][tidx], ret_ExtROI['tTargetName'][tidx])
            troi_id = ret_ExtROI["tID"][tidx]
            troi_name = ret_ExtROI["tTargetName"][tidx]

            if(troi_id == 7  ):
                offset = 135+ 40 + 5 + 10 + 10 + 10 + 10 + 5
            elif (troi_id == 8):
                offset = 170+40
            elif(troi_id == 9):
                offset = 60+ 40 + 20 + 25 + 20 + 20 + 10 + 15
            elif(troi_id == 1):
                offset = 80 + 0 + 5 + 10 + 20 + 10 +20 + 20 + 15
            elif (troi_id == 3 or troi_id == 4):
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
            ret_check, point_mapping = calc_match_roi(p0, p1, p3, p2, camPlaneOrthVector, pointOnPlan, headDir_mid, headPos)
            if (point_mapping[0] != 0 or point_mapping[1] != 0 or point_mapping[2] != 0):
                extData.loc[tindex, 'roi_idx_h'] = str(extData.loc[tindex, 'roi_idx_h']) + '/' + str(troi_id)
                extData.loc[tindex, 'roi_name_h'] = extData.loc[tindex, 'roi_name_h'] + '/' + troi_name
                extData.loc[tindex, 'intersect_x_h'] = str(extData.loc[tindex, 'intersect_x_h']) + '/' + str(
                    np.round(point_mapping[0], 1))
                extData.loc[tindex, 'intersect_y_h'] = str(extData.loc[tindex, 'intersect_y_h']) + '/' + str(
                    np.round(point_mapping[1], 1))
                extData.loc[tindex, 'intersect_z_h'] = str(extData.loc[tindex, 'intersect_z_h']) + '/' + str(
                    np.round(point_mapping[2], 1))
                extData.loc[tindex, 'roi_X'] = str(extData.loc[tindex, 'roi_X']) + '/' + str(
                    int(bcheck_match.line_point_min_dist(point_mapping, p0, p2) / distance_xyz(p0, p1) * 100))
                extData.loc[tindex, 'roi_Y'] = str(extData.loc[tindex, 'roi_Y']) + '/' + str(
                    int(bcheck_match.line_point_min_dist(point_mapping, p0, p1) / distance_xyz(p0, p2) * 100))
                extData.loc[tindex, 'roi_score'] = str(extData.loc[tindex, 'roi_score']) + '/' + str(
                    calc_score(pointOnPlan, p0, [point_mapping[0], point_mapping[1], point_mapping[2]]))

                target_roi_score = calc_score(pointOnPlan, p0, [point_mapping[0], point_mapping[1], point_mapping[2]])
                if (max_target_roi_score < target_roi_score):
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
        extData.loc[tindex, 'max_target_id'] = max_target_id
        extData.loc[tindex, 'max_roi_score'] = max_target_roi_score

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

    pROI = pROI.sort_values(['tID'], ascending=True)
    # tdata_second_type = tdata_second_type.sort_values(['title', 'type_idx', 'point_name'], ascending=True)

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

if __name__ == '__main__':
    print("\n\n\n test/////////////////////")
    if(0):
        sys.stdout = open('DebugLog.txt', 'w')
    if(1):
        inputPath_GT = "./refer/GT_3531_96_670222_0001_all.csv"
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
        # inputPath_GT = "./refer/GT_ALL/3810_91_811864_0001_all.csv"
        # inputPath_GT = "./refer/GT_ALL/3810_158_811994_0001_all.csv"
        # inputPath_GT = "./refer/GT_ALL/3810_2606_816850_0001_all.csv"
        # inputPath_GT = "./refer/GT_ALL/3810_2088_815814_0001_all.csv"
        # inputPath_GT = "./refer/GT_ALL/3810_1565_814768_0001_all.csv"
        # inputPath_GT = "./refer/GT_ALL/3810_3124_817892_0001_all.csv"

        inputPath_ROI = "./refer/roi_config.json"

        # roi_config.json
        ret_roi = load_jsonfile_ROI(inputPath_ROI)

        ret_ExtROI = extract_availData_from_3D_target_ROI(ret_roi)

        ret_ExtGT = extract_availData_from_GT(inputPath_GT)
        # print('ret_ExtGT\n\n', ret_ExtGT)

        ret_ExtGT_with_direction = retcalcuate_head_eye_direction(ret_ExtGT)

        print('\n\n',ret_ExtGT_with_direction)
        ret_match = check_match_roi(ret_ExtGT_with_direction, ret_ExtROI, 25)    #150
        save_csvfile(ret_match, "./basegaze_output.csv")
        # ret_match.to_csv("filename.csv", mode='w', index=False, header=False, sep=',', quotechar=" ",
        #                  float_format='%.4f')

        # rendering_roi_with_head_gaze(ret_ExtROI, ret_match)
        # ret_match.to_csv("filename.csv", mode='w', index=False, header=False, sep=',', quotechar=" ", float_format='%.4f')


    # print(1/0)
