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
    print('\n')
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

    available_df = pd.DataFrame(available_dict)  # index 지정
    print(available_df)
    return available_df

def extract_availData_from_GT(inputPath_GT):
    print("//////////", funcname(), "//////////")
    extGT = pd.read_csv(inputPath_GT)
    df_extGT = extGT[['f_frame_counter_left_camera', 'CAN_S_Gaze_ROI', 'CAN_S_Gaze_ROI_X', 'CAN_S_Gaze_ROI_Y',
                          'MS_S_Head_rot_X', 'MS_S_Head_rot_Y', 'MS_S_Head_rot_Z',
                          'HSVL_MS_S_Head_Pos_Veh_X', 'HSVL_MS_S_Head_Pos_Veh_Y', 'HSVL_MS_S_Head_Pos_Veh_Z',
                          'MS_S_Gaze_LE_Center_X', 'MS_S_Gaze_LE_Center_Y', 'MS_S_Gaze_LE_Center_Z',
                          'MS_S_Gaze_LE_VA_rot_X', 'MS_S_Gaze_LE_VA_rot_Y', 'MS_S_Gaze_LE_VA_rot_Z',
                          'MS_S_Gaze_RE_Center_X', 'MS_S_Gaze_RE_Center_Y', 'MS_S_Gaze_RE_Center_Z',
                          'MS_S_Gaze_RE_VA_rot_X', 'MS_S_Gaze_RE_VA_rot_Y', 'MS_S_Gaze_RE_VA_rot_Z']]
    df_extGT = df_extGT.dropna()
    # print('ret_ExtGT\n\n',ret_ExtGT)
    print('df_extGT\n\n', df_extGT)
    return df_extGT

def retcalcuate_head_eye_direction(extData):
    print("//////////", funcname(), "//////////")
    # print(extData)
    tframecnt = 0
    troi_result = 0
    troi_x = 0
    troi_y = 0
    thead_pos = 0
    thead_rot = 0
    tgaze_pos_l = 0
    tgaze_vec_l = 0
    tgaze_pos_r = 0
    tgaze_vec_r = 0

    headPos3D_meter = np.array([0, 0, 0])
    headOri_radian = np.array([0, 0, 0])

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
        troi_x = extData.loc[tindex, 'CAN_S_Gaze_ROI_X']
        troi_y = extData.loc[tindex, 'CAN_S_Gaze_ROI_Y']
        thead_pos = np.array((extData.loc[tindex, 'HSVL_MS_S_Head_Pos_Veh_X'], extData.loc[tindex, 'HSVL_MS_S_Head_Pos_Veh_Y'], extData.loc[tindex, 'HSVL_MS_S_Head_Pos_Veh_Z']))
        thead_rot = np.array((extData.loc[tindex, 'MS_S_Head_rot_X'], extData.loc[tindex, 'MS_S_Head_rot_Y'], extData.loc[tindex, 'MS_S_Head_rot_Z']))
        tgaze_pos_l = np.array((extData.loc[tindex, 'MS_S_Gaze_LE_Center_X'], extData.loc[tindex, 'MS_S_Gaze_LE_Center_Y'], extData.loc[tindex, 'MS_S_Gaze_LE_Center_Z']))
        tgaze_vec_l = np.array((extData.loc[tindex, 'MS_S_Gaze_LE_VA_rot_X'], extData.loc[tindex, 'MS_S_Gaze_LE_VA_rot_Y'], extData.loc[tindex, 'MS_S_Gaze_LE_VA_rot_Z']))
        tgaze_pos_r = np.array((extData.loc[tindex, 'MS_S_Gaze_RE_Center_X'], extData.loc[tindex, 'MS_S_Gaze_RE_Center_Y'], extData.loc[tindex, 'MS_S_Gaze_RE_Center_Z']))
        tgaze_vec_r = np.array((extData.loc[tindex, 'MS_S_Gaze_RE_VA_rot_X'], extData.loc[tindex, 'MS_S_Gaze_RE_VA_rot_Y'], extData.loc[tindex, 'MS_S_Gaze_RE_VA_rot_Z']))
        print(tframecnt, troi_result, troi_x, troi_y, thead_pos, thead_rot, tgaze_pos_l, tgaze_vec_l, tgaze_pos_r, tgaze_vec_r )
        # print(tframecnt, troi_result, troi_x, troi_y)

        headPos3D_mm = thead_pos
        headOri_radian = thead_rot * deg2Rad
        print("headPos3D_mm", headPos3D_mm)
        print("headOri_radian", headOri_radian)

        lpupil_roll_pitch_yaw_rad = tgaze_vec_l * deg2Rad #np.array([0, -9.8, 0.2])
        rpupil_roll_pitch_yaw_rad = tgaze_vec_r * deg2Rad #np.array([0, -9.8, 0.2])
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

        rt = eulerAnglesToRotationMatrix(headOri_radian)
        # rt1 = eulerAnglesToRotationMatrix(np.array([0, 0, 0]))
        # rt = rt1
        # rot2 = np.eye(3)
        rot2_l = eulerAnglesToRotationMatrix(lpupil_roll_pitch_yaw_rad)
        rot2_r = eulerAnglesToRotationMatrix(rpupil_roll_pitch_yaw_rad)
        print('rot2_l',rot2_l)
        headDir_l = np.dot(rot2_l, np.dot(rt, [1, 0, 0]))
        headDir_r = np.dot(rot2_r, np.dot(rt, [1, 0, 0]))
        print('headDir_l',headDir_l)


        print(1/0)

    return ret_ExtGT

def check_match_roi(ret_ExtGT_with_direction, ret_ExtROI):
    pass

if __name__ == '__main__':
    print("\n\n\n test/////////////////////")
    inputPath_GT = "./refer/GT_3531_96_670222_0001_all.csv"

    inputPath_ROI = "./refer/roi_config.json"
    # roi_config.json
    ret_roi = load_jsonfile_ROI(inputPath_ROI)

    ret_ExtROI = extract_availData_from_3D_target_ROI(ret_roi)

    ret_ExtGT = extract_availData_from_GT(inputPath_GT)
    # print('ret_ExtGT\n\n', ret_ExtGT)

    ret_ExtGT_with_direction = retcalcuate_head_eye_direction(ret_ExtGT)
    # check_match_roi(ret_ExtGT_with_direction, ret_ExtROI)

    print(1/0)

    # 1번 샘플 - count frame 45282
    # roi ID, roi x, roi y
    # 3 10 80
    # "_comment": "모름",
    # "id": 3,
    # "obj_params":
    # {
    # "top_left": [1.316, -0.127, 0.985],     1224.3,    -800,    975.076
    # "top_right": [1.316, 0.127, 0.985],    1224.3,    100,    975.076
    # "bottom_left": [1.316, -0.127, 0.905]    650,    -800,    630

    #Headrot_veh x, y, z (1.5 -9.3 1.5)  /
    #Headpos_veh 1501 -422 857         /

    #lpupil 1501 -453 859 / 0 -9.8 0.2
    #rpupil 1498 -392 861 / 0 -9.8 0.2

    p0 = top_left = np.array([1224.3,    -800,    975.076])
    p1 = top_right = np.array([1224.3,    100,    975.076])
    p2 = bottom_left = np.array([650,    -800,    630])
    p3 = bottom_right = bottom_left + top_right - top_left

    print("top_left",top_left)
    print("top_right",top_right)
    print("bottom_left",bottom_left)
    print("bottom_right",bottom_right,'\n')
    lpupil_3d = np.array([1501, -453, 859])
    rpupil_3d = np.array([1498, -392, 861])

    # print('test', (top_left - lpupil_3d)/zAxis)
    # translationVect = zAxis*251 + (top_left - lpupil_3d)
    # print('translationVect', translationVect)
    # translationVect2 = zAxis*-251 + lpupil_3d
    # print('translationVect2', translationVect2)

    print("1", top_left - lpupil_3d)
    print("2", top_right - lpupil_3d)
    print("3", bottom_left - lpupil_3d)
    print("4", bottom_right - lpupil_3d)

    headPos3D_meter = np.array([1501, -422, 857])
    headOri_radian = np.array([1.5, -9.3, 1.5]) * deg2Rad + np.array([0,0, 0])
    print("headOri", headOri_radian)
    # headPos3D_meter = np.array([1499.255194,	-427.7108402,	858.7503376])
    # headOri_radian = np.array([2.844070379,	-9.817217123,	3.68428392]) * deg2Rad

    lpupil_roll_pitch_yaw = np.array([0, -9.8, 0.2])
    # lpupil_roll_pitch_yaw = np.array([0, -9.8, 0.2])
    rpupil_roll_pitch_yaw = np.array([0, -9.8, 0.2])

    beta_tilt = lpupil_roll_pitch_yaw[1] * deg2Rad
    alpha_yaw = lpupil_roll_pitch_yaw[2] * deg2Rad
    gazeVector = [math.sin(alpha_yaw) * math.cos(beta_tilt), math.sin(beta_tilt),
                       math.cos(alpha_yaw) * math.cos(beta_tilt)]
    # gazeVector = lpupil_roll_pitch_yaw * deg2Rad
    print('gazeVector_pitch_yaw_roll', gazeVector)


    origin = rpupil_3d
    # origin = (headPos3D_meter)

    pitch_yaw_roll = changeRotation_unitvec2radian_check('PYR',gazeVector,'PYR')
    _, roll_pitch_yaw = changeAxis_opencv2daimler(np.array([0, 0, 0]), pitch_yaw_roll)
    print('roll_pitch_yaw', roll_pitch_yaw.T)

    print('pitch_yaw_roll', pitch_yaw_roll * rad2Deg)
    print('roll_pitch_yaw', roll_pitch_yaw * rad2Deg,'\n')
    # roll_pitch_yaw = lpupil_roll_pitch_yaw * deg2Rad

    rt = eulerAnglesToRotationMatrix(headOri_radian)
    # rt1 = eulerAnglesToRotationMatrix(np.array([0, 0, 0]))
    # rt = rt1
    # rot2 = np.eye(3)
    rot2 = eulerAnglesToRotationMatrix(roll_pitch_yaw)
    print('rot2',rot2)
    headDir = np.dot(rot2, np.dot(rt, [1, 0, 0]))
    print('headDir',headDir)
    # headDir = np.dot(rt, [0, 0, 1])
    camPlaneOrthVector = np.cross((p3 - p1), (p2 - p3))/np.linalg.norm(np.cross((p3 - p1), (p2 - p3)))
    pointOnPlan = (p0+p1+p2+p3)/4

    #최단거리
    d = np.dot(camPlaneOrthVector, p0)
    t = -(np.dot(origin,camPlaneOrthVector)+d)/ np.dot(gazeVector,camPlaneOrthVector)
    pResult = origin + np.dot(t,gazeVector)
    print('d',d)
    print('t',t)
    print('pResult',pResult)
    print('camPlaneOrthVector',camPlaneOrthVector)

    print('pointOnPlan',pointOnPlan)

    tview_point = intersectionWithPlan(origin, headDir, camPlaneOrthVector, pointOnPlan)
    print('tview_point', tview_point)

    zAxis = np.array([gazeVector[0], gazeVector[1], gazeVector[2]])
    print('zAxis', zAxis)



    # print("gaze_scale", zAxis * 1000)

    n = np.cross((p2 - p0) , (p1 - p0))
    n2 = np.cross((p2 - p1) , (p1 - p0))
    plane_n = np.cross((p2 - p1), (p3 - p0))

    n2 = (p2 - p0)
    n3 = (p1 - p2)
    test = n2 * n3 / np.linalg.norm(n2 * n3)

    # np.linalg.norm
    print('n', n)
    print('n2', n2, n3)
    print('plain',plane_n)
    print('test',test)
    print(np.cross((1,0,1), (-2,2,1)), np.dot((1,0,1), (-2,2,1)))

    pOrigin = lpupil_3d
    vDir = zAxis

    pp = pOrigin -  headDir * 195.8
    print('pp',pp)
    # np.dot(plain_n, pp-


    tempVec = np.array([0.9 , -2.0 , -0.5])
    tempVecNorm = tempVec / np.linalg.norm(tempVec)
    print("\n\ntempVecNorm",tempVecNorm)
    pp2 = pOrigin -  tempVecNorm * 460.8
    print("pp2",pp2)

    # v_result = plane_p0 - line_origin
    # f_result = np.dot(v_result, plane_n)
    # f_t = f_result / np.dot(line_vDir, plane_n)
    # f_s = line_origin + line_vDir_scale(f_t)

    # origin = lpupil_3d
    # # origin = (headPos3D_meter)
    #
    # print('\n\n')
    # alpha_yaw = math.atan(nR_unitvec[0] / nR_unitvec[2])
    # beta_tilt = math.asin(nR_unitvec[1])
    # print('yaw',math.atan(nR_unitvec[0] / nR_unitvec[2]), alpha_yaw*rad2Deg)
    # print('tilt',math.asin(nR_unitvec[1]), beta_tilt*rad2Deg)
    # print('r3', [math.sin(alpha_yaw)*math.cos(beta_tilt), math.sin(beta_tilt), math.cos(alpha_yaw)*math.cos(beta_tilt) ])



    _,pitch_yaw_roll_vector = changeAxis_daimler2opencv(np.array([0,0,0]),-tempVecNorm)


    # roll_pitch_yaw_vector = tempVecNorm
    # pitch_yaw_roll_vector = np.array([roll_pitch_yaw_vector[1], roll_pitch_yaw_vector[2], roll_pitch_yaw_vector[0]])
    print('pitch_yaw_roll_vector',pitch_yaw_roll_vector.T)
    pitch_yaw_roll = changeRotation_unitvec2radian_check('PYR', pitch_yaw_roll_vector,'PYR')
    print('pitch_yaw_roll', pitch_yaw_roll * rad2Deg)
    test2= changeRotation_unitvec2radian_check2('PYR', pitch_yaw_roll_vector,'PYR')
    print('test2', test2 * rad2Deg)

    _, roll_pitch_yaw = changeAxis_opencv2daimler(np.array([0,0,0]), pitch_yaw_roll)
    # roll_pitch_yaw = np.array([pitch_yaw_roll[2], -pitch_yaw_roll[0], pitch_yaw_roll[1]])
    print('roll_pitch_yaw_rad', roll_pitch_yaw ,'\n')
    print('roll_pitch_yaw_deg', roll_pitch_yaw * rad2Deg,'\n')
    # # roll_pitch_yaw = lpupil_roll_pitch_yaw * deg2Rad
    #
    # rt = eulerAnglesToRotationMatrix(headOri_radian)
    rt1 = eulerAnglesToRotationMatrix(np.array([0, 0, 0]))
    rt = rt1
    # rot2 = np.eye(3)
    rot2 = eulerAnglesToRotationMatrix(roll_pitch_yaw)
    # rot2 = eulerAnglesToRotationMatrix(np.array([-roll_pitch_yaw[0],-roll_pitch_yaw[1],roll_pitch_yaw[2]]))
    print('rot2',rot2)
    headDir = np.dot(rot2, np.dot(rt, [1, 0, 0]))
    print('headDir',headDir)


    tview_point5 = intersectionWithPlan(origin, headDir, camPlaneOrthVector, pointOnPlan)
    print('tview_point5', tview_point5)


    # roll_pitch_yaw_vector = tempVecNorm
    # pitch_yaw_roll_vector = np.array([roll_pitch_yaw_vector[1], roll_pitch_yaw_vector[2], roll_pitch_yaw_vector[0]])
    # print('pitch_yaw_roll_vector',pitch_yaw_roll_vector* rad2Deg)
    # pitch_yaw_roll = changeRotation_unitvec2radian_check(pitch_yaw_roll_vector)
    # roll_pitch_yaw = np.array([pitch_yaw_roll[2], -pitch_yaw_roll[0], pitch_yaw_roll[1]])
    # print('pitch_yaw_roll', pitch_yaw_roll * rad2Deg)
    # print('roll_pitch_yaw_rad', roll_pitch_yaw ,'\n')
    # print('roll_pitch_yaw_deg', roll_pitch_yaw * rad2Deg,'\n')
    # # # roll_pitch_yaw = lpupil_roll_pitch_yaw * deg2Rad
    # #
    # # rt = eulerAnglesToRotationMatrix(headOri_radian)
    # rt1 = eulerAnglesToRotationMatrix(np.array([0, 0, 0]))
    # rt = rt1
    # # rot2 = np.eye(3)
    # rot2 = eulerAnglesToRotationMatrix(roll_pitch_yaw)
    # print('rot2',rot2)
    # headDir = np.dot(rot2, np.dot(rt, [1, 0, 0]))
    # print('headDir',headDir)

    print('start',np.array([-1,2,-3]))
    _,zzz = changeAxis_daimler2opencv(np.array([0,0,0]),np.array([-1,2,-3]))
    print('zzz',zzz.T)
    _, zzz2 = changeAxis_opencv2daimler(np.array([0, 0, 0]), zzz)
    print('zzz2', zzz2.T)
    _, zzz3 = changeAxis_daimler2opencv(np.array([0, 0, 0]), zzz2)
    print('zzz3', zzz3.T)
