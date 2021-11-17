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

from tkinter import filedialog
from tkinter import *

#offset설정만큼 끝애서 잘라냄.
FRAMECNT_CROP_CHUNK = 40

# 최대 줄 수 설정
pd.set_option('display.max_rows', 2500)
# 최대 열 수 설정
pd.set_option('display.max_columns', 200)
# 표시할 가로의 길이
pd.set_option('display.width', 160)
# 출력값 소숫점4자리로 설정
pd.options.display.float_format = '{:.4f}'.format

#기존모델에서 meter으로 정의하였기에, 사용된는 단위는 모두 m로 한다
#또한 degree(default)로 함
#basic model use meter, then this code also use unit of m.
#same as use degree(default)

# master camera point on Vehicle coord
cam2veh_rot = np.array([0, -11, 0.99999999999999978])
cam2veh_trans = [1.0058888249122007, -0.35483707652634749, 0.68375427481211271]

# display center point on Vehicle coord
disp2veh_rot = np.array([0, -8, 1])
disp2veh_trans = [1.0252906, -0.4003454, 0.6392974]

# display center point on Camera coord
disp2cam_rot = np.array([0.0, 3.0, 0.0])
disp2cam_trans = [0.00978, -0.04584, -0.04719]

#눈의 Height를 추출 함수 (iris의 높이문제 존재함)
def extract_eyeHeight_from_2d_point(point2d):
    if(len(point2d) == 0):
        return 0.0,0.0
    print("//////////", funcname(), "//////////")
    algo_lefteye_top_x = (point2d[43][0] + point2d[44][0] / 2)
    algo_lefteye_bottom_x = (point2d[47][0] + point2d[46][0] / 2)
    algo_righteye_top_x = (point2d[37][0] + point2d[38][0] / 2)
    algo_righteye_bottom_x = (point2d[41][0] + point2d[40][0] / 2)

    algo_lefteye_top_y = (point2d[43][1] + point2d[44][1] / 2)
    algo_lefteye_bottom_y = (point2d[47][1] + point2d[46][1] / 2)
    algo_righteye_top_y = (point2d[37][1] + point2d[38][1] / 2)
    algo_righteye_bottom_y = (point2d[41][1] + point2d[40][1] / 2)

    algo_lefteye_d = np.sqrt((algo_lefteye_top_x - algo_lefteye_bottom_x) ** 2 +
                             (algo_lefteye_top_y - algo_lefteye_bottom_y) ** 2)
    algo_righteye_d = np.sqrt((algo_righteye_top_x - algo_righteye_bottom_x) ** 2 +
                              (algo_righteye_top_y - algo_righteye_bottom_y) ** 2)
    # algo_lefteye_closure = np.maximum(1 - algo_lefteye_d / dict_gt.lefteye_iris_width, np.zeros_like(algo_lefteye_d))
    # algo_righteye_closure = np.maximum(1 - algo_righteye_d / dict_gt.righteye_iris_width, np.zeros_like(algo_righteye_d))
    return round(algo_lefteye_d,3), round(algo_righteye_d,3)

def extract_availData_from_craft_algo(pJson):
    print("//////////", funcname(), "//////////")
    tTotalList = []
    tframeId = []
    tfacePoints2dICS = []
    theadOri = []
    theadPos3D = []
    tisFaceDetected = []
    irisHeight = []

    tisFusedGazeValid = []
    tfusedGazeDegree2D = []
    tfusedGazeStart3D = []
    tfusedGazeVector3D = []

    tisLeftGazeValid = []
    tleftGazeDegree2D = []
    tleftGazeStart3D = []
    tleftGazeVector3D = []

    tisRightGazeValid = []
    trightGazeDegree2D = []
    trightGazeStart3D = []
    trightGazeVector3D = []

    for no in pJson:
        if(no != 'EOF'):
            for name in pJson[no]:
                if (name == 'frameId'):
                    # print(no, name, pJson[no][name])
                    tframeId.append(pJson[no][name])
                elif(name == 'facePoints2dICS'):
                    print(no, name, pJson[no][name], type(pJson[no][name]))
                    # (43, 44)  (47, 46)    right(37, 38)   (41, 40)
                    # if(int(no) == 2201):
                    #     print(1/0)
                    templist = []
                    if(pJson[no][name] == None):
                        templist = []
                    else:
                        for i, sub_no in enumerate(pJson[no][name]):
                            # if(i==43 or i==44 or i==46 or i==47 or i==37 or i==38 or i==40 or i==41  ):
                            # print(i, sub_no['x'], sub_no['y'])
                            templist.append((sub_no['x'], sub_no['y']))
                    print('templist',templist)
                    tfacePoints2dICS.append(templist)
                    irisHeight.append(extract_eyeHeight_from_2d_point(templist))
                    print("irisHeight",irisHeight)
                # if (name == 'fusedGazeDegree'):
                #     print(no, name, pJson[no][name])
                elif (name == 'headOri'):
                    # print(no, name, pJson[no][name])
                    # print(pJson[no]['headOri']['x'], pJson[no]['headOri']['y'], pJson[no]['headOri']['z'])
                    theadOri.append((pJson[no]['headOri']['x']*rad2Deg, pJson[no]['headOri']['y']*rad2Deg, pJson[no]['headOri']['z']*rad2Deg))
                elif (name == 'headPos3D'):
                    # print(no, name, pJson[no][name])
                    # print(pJson[no]['headPos3D']['x'], pJson[no]['headPos3D']['y'], pJson[no]['headPos3D']['z'])
                    theadPos3D.append((pJson[no]['headPos3D']['x'], pJson[no]['headPos3D']['y'], pJson[no]['headPos3D']['z']))
                elif (name == 'isFaceDetected'):
                    # print(no, name, pJson[no][name])
                    # print(pJson[no]['isFaceDetected'])
                    tisFaceDetected.append(pJson[no]['isFaceDetected'])

                #flag of valid about gaze
                elif (name == 'isFusedGazeValid'):
                    tisFusedGazeValid.append(pJson[no]['isFusedGazeValid'])
                elif (name == 'isLeftGazeValid'):
                    tisLeftGazeValid.append(pJson[no]['isLeftGazeValid'])
                elif (name == 'isRightGazeValid'):
                    tisRightGazeValid.append(pJson[no]['isRightGazeValid'])
                #2d degree about gaze
                elif (name == 'fusedGazeDegree'):
                    tfusedGazeDegree2D.append((pJson[no]['fusedGazeDegree']['x'],pJson[no]['fusedGazeDegree']['y']))
                elif (name == 'leftGazeDegree'):
                    tleftGazeDegree2D.append((pJson[no]['leftGazeDegree']['x'],pJson[no]['leftGazeDegree']['y']))
                elif (name == 'rightGazeDegree'):
                    trightGazeDegree2D.append((pJson[no]['rightGazeDegree']['x'],pJson[no]['rightGazeDegree']['y']))
                #start point on 3d about gaze
                elif (name == 'fusedGazeStart3D'):
                    tfusedGazeStart3D.append((pJson[no]['fusedGazeStart3D']['x'],pJson[no]['fusedGazeStart3D']['y'],pJson[no]['fusedGazeStart3D']['z']))
                elif (name == 'leftGazeStart3D'):
                    tleftGazeStart3D.append((pJson[no]['leftGazeStart3D']['x'],pJson[no]['leftGazeStart3D']['y'],pJson[no]['leftGazeStart3D']['z']))
                elif (name == 'rightGazeStart3D'):
                    trightGazeStart3D.append((pJson[no]['rightGazeStart3D']['x'],pJson[no]['rightGazeStart3D']['y'],pJson[no]['rightGazeStart3D']['z']))
                #vector on 3d about gaze
                elif (name == 'fusedGazeVector'):
                    tfusedGazeVector3D.append((pJson[no]['fusedGazeVector']['x'],pJson[no]['fusedGazeVector']['y'],pJson[no]['fusedGazeVector']['z']))
                elif (name == 'leftGazeVector'):
                    tleftGazeVector3D.append((pJson[no]['leftGazeVector']['x'],pJson[no]['leftGazeVector']['y'],pJson[no]['leftGazeVector']['z']))
                elif (name == 'rightGazeVector'):
                    trightGazeVector3D.append((pJson[no]['rightGazeVector']['x'],pJson[no]['rightGazeVector']['y'],pJson[no]['rightGazeVector']['z']))




            #     "isFusedGazeValid": true,
            #     "isLeftGazeValid": true,
            #     "isRightGazeValid": true,
            print("\n")

    print("final_tframeId", len(tframeId), tframeId)
    print("final_tfacePoints2dICS", len(tfacePoints2dICS), tfacePoints2dICS)
    print("final_theadPos3D", len(theadPos3D), theadPos3D)
    print("final_theadOri_degree", len(theadOri), theadOri)
    print("final_tisFaceDetected", len(tisFaceDetected), tisFaceDetected)

    print("\n/////////////phase2////////")
    print("final_tisFusedGazeValid", len(tisFusedGazeValid), tisFusedGazeValid)
    print("final_tisLeftGazeValid", len(tisLeftGazeValid), tisLeftGazeValid)
    print("final_tisRightGazeValid", len(tisRightGazeValid), tisRightGazeValid)
    print("final_tfusedGazeDegree2D", len(tfusedGazeDegree2D), tfusedGazeDegree2D)
    print("final_tleftGazeDegree2D", len(tleftGazeDegree2D), tleftGazeDegree2D)
    print("final_trightGazeDegree2D", len(trightGazeDegree2D), trightGazeDegree2D)
    print("final_tfusedGazeStart3D", len(tfusedGazeStart3D), tfusedGazeStart3D)
    print("final_tleftGazeStart3D", len(tleftGazeStart3D), tleftGazeStart3D)
    print("final_trightGazeStart3D", len(trightGazeStart3D), trightGazeStart3D)
    print("final_tfusedGazeVector3D", len(tfusedGazeVector3D), tfusedGazeVector3D)
    print("final_tleftGazeVector3D", len(tleftGazeVector3D), tleftGazeVector3D)
    print("final_trightGazeVector3D", len(trightGazeVector3D), trightGazeVector3D)

    # available_columns = ["frameId","facePoints2dICS", "headPos3D", "headOri","isFaceDetected"]
    available_dict = {"isFaceDetected":tisFaceDetected,"headPos3D":theadPos3D, "headOri":theadOri,"facePoints2dICS":tfacePoints2dICS,"irisHeight":irisHeight,
                      "isFusedGazeValid":tisFusedGazeValid,"isLeftGazeValid":tisLeftGazeValid,"isRightGazeValid":tisRightGazeValid,
                      "fusedGazeDegree2D": tfusedGazeDegree2D, "leftGazeDegree2D": tleftGazeDegree2D,"rightGazeDegree2D": trightGazeDegree2D,
                      "fusedGazeStart3D": tfusedGazeStart3D, "leftGazeStart3D": tleftGazeStart3D,"rightGazeStart3D": trightGazeStart3D,
                      "fusedGazeVector3D": tfusedGazeVector3D, "leftGazeVector3D": tleftGazeVector3D,"rightGazeVector3D": trightGazeVector3D
                      }
    # available_df = pd.DataFrame({tframeId,theadPos3D, theadPos3D, theadOri,tisFaceDetected }, index=tframeId, columns=available_columns)  # index 지정

    # available_df = pd.DataFrame({'frameId':tframeId,"theadPos3D":theadPos3D}, columns=["frameId","theadPos3D"])  # index 지정

    available_df = pd.DataFrame(available_dict, index=tframeId)  # index 지정
    print(available_df)
    return available_df

def load_jsonfile_HET(fname):
    print("//////////", funcname(), "//////////")

    fp = open(fname)
    fjs = json.load(fp)
    fp.close()
    # print(fjs)
    return fjs

def load_jsonfile_preValue(fname1, fname2, fname3):
    fp1 = open(fname1)
    fjs1 = json.load(fp1)
    fp1.close()

    fp2 = open(fname2)
    fjs2 = json.load(fp2)
    fp2.close()

    fp3 = open(fname3)
    fjs3 = json.load(fp3)
    fp3.close()

    print('\nLoading...', fname1,'\n', fjs1)
    print('\nLoading...', fname2,'\n',fjs2)
    print('\nLoading...', fname3,'\n', fjs3)
    print('\n')

    global cam2veh_rot, cam2veh_trans, disp2veh_rot, disp2veh_trans, disp2cam_rot, disp2cam_trans
    cam2veh_rot = fjs1['master']['camera_pose']['rot']
    cam2veh_trans = fjs1['master']['camera_pose']['trans']
    disp2veh_rot = fjs2['master']['display_pose']['rot']
    disp2veh_trans = fjs2['master']['display_pose']['trans']
    disp2cam_rot = fjs3['display_pose_wrt_master_camera']['rot']
    disp2cam_trans = fjs3['display_pose_wrt_master_camera']['trans']

    print("update... cam2veh_rot, cam2veh_trans", cam2veh_rot, cam2veh_trans)
    print("update... disp2veh_rot, disp2veh_trans",disp2veh_rot, disp2veh_trans)
    print("update... disp2cam_rot, disp2cam_trans",disp2cam_rot, disp2cam_trans)
    return True

def load_jsonfile_preValue_extend(fname1, fname2):
    fp1 = open(fname1)
    fjs1 = json.load(fp1)
    fp1.close()

    fp2 = open(fname2)
    fjs2 = json.load(fp2)
    fp2.close()

    # fp3 = open(fname3)
    # fjs3 = json.load(fp3)
    # fp3.close()

    print('\nLoading...', fname1,'\n', fjs1)
    print('\nLoading...', fname2,'\n',fjs2)
    # print('\nLoading...', fname3,'\n', fjs3)
    print('\n')

    global cam2veh_rot, cam2veh_trans, disp2veh_rot, disp2veh_trans, disp2cam_rot, disp2cam_trans
    cam2veh_rot = fjs1['master']['camera_pose']['rot']
    cam2veh_trans = fjs1['master']['camera_pose']['trans']
    disp2veh_rot = fjs2['master']['display_pose']['rot']
    disp2veh_trans = fjs2['master']['display_pose']['trans']

    print("update... cam2veh_rot, cam2veh_trans", cam2veh_rot, cam2veh_trans)
    print("update... disp2veh_rot, disp2veh_trans",disp2veh_rot, disp2veh_trans)

    veh2cam_rot, veh2cam_trans = transform_3by3_inverse(cam2veh_rot, cam2veh_trans)
    # print("update... veh2cam_rot, veh2cam_trans", veh2cam_rot , veh2cam_trans)

    disp2cam_matrix, _ = transform_A2Bcoord_and_B2Ccoord(disp2veh_rot, disp2veh_trans, veh2cam_rot , veh2cam_trans )
    # print('disp2cam_matrix',rotationMatrixToEulerAngles(disp2cam_matrix[0:3, 0:3])*rad2Deg, disp2cam_matrix[0:3, 3])

    disp2cam_rot = rotationMatrixToEulerAngles(disp2cam_matrix[0:3, 0:3])*rad2Deg
    disp2cam_trans = disp2cam_matrix[0:3, 3]

    print("update... disp2cam_rot, disp2cam_trans",disp2cam_rot, disp2cam_trans)
    return True

def load_jsonfile_ROI(fname):
    print("//////////", funcname(), "//////////")

    fp = open(fname)
    fjs = json.load(fp)
    fp.close()
    # print(fjs)
    return fjs

def make_prototype_on_pandas(frameId):
    print("//////////", funcname(), "//////////")

    target_columns = ["f_version", "f_frame_counter_left_camera", "f_frame_counter_right_camera",
                      "f_frame_counter_virtual", "f_master_counter_in", "f_eye_model_generation",
                      "eye_left_partial_blockage", "eye_left_blockage", "MS_EyeLeftCameraLeft_old",
                      "MS_EyeLeftCameraRight_old", "eye_right_partial_blockage", "eye_right_blockage",
                      "MS_EyeRightCameraLeft_old", "MS_EyeRightCameraRight_old", "mouth_nose_partial_blockage",
                      "mouth_nose_blockage", "MS_MouthNoseCameraLeft", "MS_MouthNoseCameraRight",
                      "CAN_S_glasses_detected_old", "CAN_S_face_detection", "MS_EyeDetection", "MS_S_Head_rot_X",
                      "MS_S_Head_rot_Y", "MS_S_Head_rot_Z", "HSVL_MS_CAN_S_Head_tracking_status",
                      "HSVL_MS_CAN_S_Head_tracking_mode", "f_primary_face_landmark_camera", "f_left_position_valid0",
                      "f_left_positionX0", "f_left_positionY0", "f_left_position_valid1", "f_left_positionX1",
                      "f_left_positionY1", "f_left_position_valid2", "f_left_positionX2", "f_left_positionY2",
                      "f_left_position_valid3", "f_left_positionX3", "f_left_positionY3", "f_left_position_valid4",
                      "f_left_positionX4", "f_left_positionY4", "f_left_position_valid5", "f_left_positionX5",
                      "f_left_positionY5", "f_left_position_valid6", "f_left_positionX6", "f_left_positionY6",
                      "f_right_position_valid0", "f_right_positionX0", "f_right_positionY0", "f_right_position_valid1",
                      "f_right_positionX1", "f_right_positionY1", "f_right_position_valid2", "f_right_positionX2",
                      "f_right_positionY2", "f_right_position_valid3", "f_right_positionX3", "f_right_positionY3",
                      "f_right_position_valid4", "f_right_positionX4", "f_right_positionY4", "f_right_position_valid5",
                      "f_right_positionX5", "f_right_positionY5", "f_right_position_valid6", "f_right_positionX6",
                      "f_right_positionY6", "CAN_glint_detected", "f_gaze_le_result_valid", "MS_S_Gaze_LE_VA_rot_X",
                      "MS_S_Gaze_LE_VA_rot_Y", "MS_S_Gaze_LE_VA_rot_Z", "MS_S_Gaze_LE_Center_X",
                      "MS_S_Gaze_LE_Center_Y", "MS_S_Gaze_LE_Center_Z", "f_gaze_re_result_valid",
                      "MS_S_Gaze_RE_VA_rot_X", "MS_S_Gaze_RE_VA_rot_Y", "MS_S_Gaze_RE_VA_rot_Z",
                      "MS_S_Gaze_RE_Center_X", "MS_S_Gaze_RE_Center_Y", "MS_S_Gaze_RE_Center_Z", "CAN_eye_closure_left",
                      "CAN_eye_closure_left_conf", "CAN_eye_closure_right", "CAN_eye_closure_right_conf",
                      "CAN_eye_closure", "MS_eye_closed", "CAN_long_eyeclosure", "f_long_eyeclosure_counter",
                      "MS_eye_closed_AAS", "MS_PERCLOS_AAS", "MS_PERCLOS_AAS_conf", "MS_PERCLOS_strict",
                      "MS_eye_closed_strict", "CAN_eye_blink_conf", "MS_Eye_blink_freq_conf", "MS_Eye_blink_freq",
                      "CAN_eye_blink_t_closing", "CAN_eye_blink_t_opening", "CAN_eye_blink_duration",
                      "CAN_eye_blink_A_closing", "CAN_eye_blink_A_opening", "CAN_eye_blink_counter", "CAN_S_Gaze_ROI",
                      "CAN_S_Gaze_ROI_X", "CAN_S_Gaze_ROI_Y", "f_roi_id", "MS_S_Gaze_ROI_X_Raw", "MS_S_Gaze_ROI_Y_Raw",
                      "HSVL_MS_S_Head_Pos_Veh_X", "HSVL_MS_S_Head_Pos_Veh_Y", "HSVL_MS_S_Head_Pos_Veh_Z",
                      "MS_S_Gaze_rot_X", "MS_S_Gaze_rot_Y", "MS_S_Gaze_rot_Z", "HSVL_MS_CAN_S_Eye_dist",
                      "f_camera_left_measured_brightness", "f_camera_left_target_brightness",
                      "f_camera_left_shutter_us", "f_camera_left_column_gain", "f_camera_left_digital_gain",
                      "f_camera_right_measured_brightness", "f_camera_right_target_brightness",
                      "f_camera_right_shutter_us", "f_camera_right_column_gain", "f_camera_right_digital_gain",
                      "f_raw_result_age", "HSVL_S_Head_dir_h", "HSVL_S_Head_dir_v", "HSVL_MS_S_Head_Pos_Disp_X",
                      "HSVL_MS_S_Head_Pos_Disp_Y", "HSVL_MS_S_Head_Pos_Disp_Z", "LGE_BD_frame_count",
                      "CAN_S_camera_close_blocked", "CAN_S_glasses_detected", "MS_camera_blockage_detection",
                      "MS_EyeLeftCameraLeft", "MS_EyeLeftCameraRight", "MS_EyeRightCameraLeft",
                      "MS_EyeRightCameraRight", "MouthNoseCameraLeft", "MouthNoseCameraRight",
                      "disp_left_cam_blocked_0", "disp_left_cam_blocked_1", "disp_left_cam_blocked_2",
                      "disp_left_cam_blocked_3", "disp_right_cam_blocked_0", "disp_right_cam_blocked_1",
                      "disp_right_cam_blocked_2", "disp_right_cam_blocked_3", "LGE_OOF_frame_count", "bOutOfFocus",
                      "CAN_S_drcam_status", "LGE_SWBA_frame_count", "CAN_S_StWhl_adjust_occlusion", "MS_HeadOcclusion",
                      "Absorber_Left_Center_nX", "Absorber_Left_Center_nY", "Absorber_Right_Center_nX",
                      "Absorber_Right_Center_nY", "nAbsorber_Radius", "Wheel_Left_Center_nX", "Wheel_Left_Center_nY",
                      "Wheel_Right_Center_nX", "Wheel_Right_Center_nY", "nWheel_Radius", "LGE_DB_frame_count",
                      "bIsHeadMoving", "MS_Intended_head_movement", "CAN_Driver_is_responsive", "MS_nod_head_gesture",
                      "MS_shake_head_gesture", "bIsHeadGestureResult", "LGE_DI_frame_count", "CAN_S_Driver_ID_Top_1",
                      "CAN_S_Driver_ID_Top_2", "CAN_S_Driver_ID_Top_3", "CAN_S_Driver_ID_Confidence_1",
                      "CAN_S_Driver_ID_Confidence_2", "CAN_S_Driver_ID_Confidence_3", "arrDrviers_01", "arrDrviers_02",
                      "arrDrviers_03", "arrDrviers_04", "arrDrviers_05", "arrDrviers_06", "arrDrviers_07",
                      "arrDrviers_08", "arrDrviers_09", "arrDrviers_10", "arrDrviers_11", "arrDrviers_12",
                      "arrDrviers_13", "status", "numOfDriver", "LGE_DOT_frame_count", "CAN_S_Head_Pos_X",
                      "CAN_S_Head_Pos_Y", "CAN_S_Head_Pos_Z", "CAN_S_Head_Pos_type", "bLeftIRLight",
                      "LGE_SP_frame_count", "MS_spoofing_detected", "CurrentStatus", "OutConfidenceSpoof",
                      "OutConfidenceGenuine", "f_raw_result_age", "S_Head_dir_h", "S_Head_dir_v", "S_Head_Pos_Disp_x",
                      "S_Head_Pos_Disp_y", "S_Head_Pos_Disp_z", "f_left_position_valid7", "f_left_positionX7",
                      "f_left_positionY7", "f_right_position_valid7", "f_right_positionX7", "f_right_positionY7",
                      "FrameHistory", "FrameDiff", "FrameTimeStamp", "f_head_pose_confidence",
                      "f_early_head_pose_confidence", "LGE_DI_Ext_frame_count", "CAN_S_EnrollAndDeleteStatus",
                      "CAN_S_HasStoredIDs", "CAN_S_driverID_MsgCnt", "ResultDataType", "timeEnrollment",
                      "bFilteringFlag", "f_le_iris_diameter", "f_re_iris_diameter",
                      "HetAlgoGlintPosition.MS_S_Gaze_LE_Cornea_Center_X",
                      "HetAlgoGlintPosition.MS_S_Gaze_LE_Cornea_Center_Y",
                      "HetAlgoGlintPosition.MS_S_Gaze_LE_Cornea_Center_Z",
                      "HetAlgoGlintPosition.f_le_glint_position_idx_0_X",
                      "HetAlgoGlintPosition.f_le_glint_position_idx_0_Y",
                      "HetAlgoGlintPosition.f_le_glint_position_idx_1_X",
                      "HetAlgoGlintPosition.f_le_glint_position_idx_1_Y",
                      "HetAlgoGlintPosition.MS_S_Gaze_RE_Cornea_Center_X",
                      "HetAlgoGlintPosition.MS_S_Gaze_RE_Cornea_Center_Y",
                      "HetAlgoGlintPosition.MS_S_Gaze_RE_Cornea_Center_Z",
                      "HetAlgoGlintPosition.f_re_glint_position_idx_0_X",
                      "HetAlgoGlintPosition.f_re_glint_position_idx_0_Y",
                      "HetAlgoGlintPosition.f_re_glint_position_idx_1_X",
                      "HetAlgoGlintPosition.f_re_glint_position_idx_1_Y",
                      "HetAlgoEyelidPosition.f_le_lc_position_valid0", "HetAlgoEyelidPosition.f_le_lc_positionX0",
                      "HetAlgoEyelidPosition.f_le_lc_positionY0", "HetAlgoEyelidPosition.f_le_lc_position_valid1",
                      "HetAlgoEyelidPosition.f_le_lc_positionX1", "HetAlgoEyelidPosition.f_le_lc_positionY1",
                      "HetAlgoEyelidPosition.f_le_lc_position_valid2", "HetAlgoEyelidPosition.f_le_lc_positionX2",
                      "HetAlgoEyelidPosition.f_le_lc_positionY2", "HetAlgoEyelidPosition.f_le_lc_position_valid3",
                      "HetAlgoEyelidPosition.f_le_lc_positionX3", "HetAlgoEyelidPosition.f_le_lc_positionY3",
                      "HetAlgoEyelidPosition.f_le_lc_position_valid4", "HetAlgoEyelidPosition.f_le_lc_positionX4",
                      "HetAlgoEyelidPosition.f_le_lc_positionY4", "HetAlgoEyelidPosition.f_le_lc_position_valid5",
                      "HetAlgoEyelidPosition.f_le_lc_positionX5", "HetAlgoEyelidPosition.f_le_lc_positionY5",
                      "HetAlgoEyelidPosition.f_le_lc_position_valid6", "HetAlgoEyelidPosition.f_le_lc_positionX6",
                      "HetAlgoEyelidPosition.f_le_lc_positionY6", "HetAlgoEyelidPosition.f_le_rc_position_valid0",
                      "HetAlgoEyelidPosition.f_le_rc_positionX0", "HetAlgoEyelidPosition.f_le_rc_positionY0",
                      "HetAlgoEyelidPosition.f_le_rc_position_valid1", "HetAlgoEyelidPosition.f_le_rc_positionX1",
                      "HetAlgoEyelidPosition.f_le_rc_positionY1", "HetAlgoEyelidPosition.f_le_rc_position_valid2",
                      "HetAlgoEyelidPosition.f_le_rc_positionX2", "HetAlgoEyelidPosition.f_le_rc_positionY2",
                      "HetAlgoEyelidPosition.f_le_rc_position_valid3", "HetAlgoEyelidPosition.f_le_rc_positionX3",
                      "HetAlgoEyelidPosition.f_le_rc_positionY3", "HetAlgoEyelidPosition.f_le_rc_position_valid4",
                      "HetAlgoEyelidPosition.f_le_rc_positionX4", "HetAlgoEyelidPosition.f_le_rc_positionY4",
                      "HetAlgoEyelidPosition.f_le_rc_position_valid5", "HetAlgoEyelidPosition.f_le_rc_positionX5",
                      "HetAlgoEyelidPosition.f_le_rc_positionY5", "HetAlgoEyelidPosition.f_le_rc_position_valid6",
                      "HetAlgoEyelidPosition.f_le_rc_positionX6", "HetAlgoEyelidPosition.f_le_rc_positionY6",
                      "HetAlgoEyelidPosition.f_re_lc_position_valid0", "HetAlgoEyelidPosition.f_re_lc_positionX0",
                      "HetAlgoEyelidPosition.f_re_lc_positionY0", "HetAlgoEyelidPosition.f_re_lc_position_valid1",
                      "HetAlgoEyelidPosition.f_re_lc_positionX1", "HetAlgoEyelidPosition.f_re_lc_positionY1",
                      "HetAlgoEyelidPosition.f_re_lc_position_valid2", "HetAlgoEyelidPosition.f_re_lc_positionX2",
                      "HetAlgoEyelidPosition.f_re_lc_positionY2", "HetAlgoEyelidPosition.f_re_lc_position_valid3",
                      "HetAlgoEyelidPosition.f_re_lc_positionX3", "HetAlgoEyelidPosition.f_re_lc_positionY3",
                      "HetAlgoEyelidPosition.f_re_lc_position_valid4", "HetAlgoEyelidPosition.f_re_lc_positionX4",
                      "HetAlgoEyelidPosition.f_re_lc_positionY4", "HetAlgoEyelidPosition.f_re_lc_position_valid5",
                      "HetAlgoEyelidPosition.f_re_lc_positionX5", "HetAlgoEyelidPosition.f_re_lc_positionY5",
                      "HetAlgoEyelidPosition.f_re_lc_position_valid6", "HetAlgoEyelidPosition.f_re_lc_positionX6",
                      "HetAlgoEyelidPosition.f_re_lc_positionY6", "HetAlgoEyelidPosition.f_re_rc_position_valid0",
                      "HetAlgoEyelidPosition.f_re_rc_positionX0", "HetAlgoEyelidPosition.f_re_rc_positionY0",
                      "HetAlgoEyelidPosition.f_re_rc_position_valid1", "HetAlgoEyelidPosition.f_re_rc_positionX1",
                      "HetAlgoEyelidPosition.f_re_rc_positionY1", "HetAlgoEyelidPosition.f_re_rc_position_valid2",
                      "HetAlgoEyelidPosition.f_re_rc_positionX2", "HetAlgoEyelidPosition.f_re_rc_positionY2",
                      "HetAlgoEyelidPosition.f_re_rc_position_valid3", "HetAlgoEyelidPosition.f_re_rc_positionX3",
                      "HetAlgoEyelidPosition.f_re_rc_positionY3", "HetAlgoEyelidPosition.f_re_rc_position_valid4",
                      "HetAlgoEyelidPosition.f_re_rc_positionX4", "HetAlgoEyelidPosition.f_re_rc_positionY4",
                      "HetAlgoEyelidPosition.f_re_rc_position_valid5", "HetAlgoEyelidPosition.f_re_rc_positionX5",
                      "HetAlgoEyelidPosition.f_re_rc_positionY5", "HetAlgoEyelidPosition.f_re_rc_position_valid6",
                      "HetAlgoEyelidPosition.f_re_rc_positionX6", "HetAlgoEyelidPosition.f_re_rc_positionY6",
                      "Het2DLandmark.f_left_position_valid0", "Het2DLandmark.f_left_positionX0",
                      "Het2DLandmark.f_left_positionY0", "Het2DLandmark.f_left_position_valid1",
                      "Het2DLandmark.f_left_positionX1", "Het2DLandmark.f_left_positionY1",
                      "Het2DLandmark.f_left_position_valid2", "Het2DLandmark.f_left_positionX2",
                      "Het2DLandmark.f_left_positionY2", "Het2DLandmark.f_left_position_valid3",
                      "Het2DLandmark.f_left_positionX3", "Het2DLandmark.f_left_positionY3",
                      "Het2DLandmark.f_left_position_valid4", "Het2DLandmark.f_left_positionX4",
                      "Het2DLandmark.f_left_positionY4", "Het2DLandmark.f_left_position_valid5",
                      "Het2DLandmark.f_left_positionX5", "Het2DLandmark.f_left_positionY5",
                      "Het2DLandmark.f_left_position_valid6", "Het2DLandmark.f_left_positionX6",
                      "Het2DLandmark.f_left_positionY6", "Het2DLandmark.f_left_position_valid7",
                      "Het2DLandmark.f_left_positionX7", "Het2DLandmark.f_left_positionY7",
                      "Het2DLandmark.f_right_position_valid0", "Het2DLandmark.f_right_positionX0",
                      "Het2DLandmark.f_right_positionY0", "Het2DLandmark.f_right_position_valid1",
                      "Het2DLandmark.f_right_positionX1", "Het2DLandmark.f_right_positionY1",
                      "Het2DLandmark.f_right_position_valid2", "Het2DLandmark.f_right_positionX2",
                      "Het2DLandmark.f_right_positionY2", "Het2DLandmark.f_right_position_valid3",
                      "Het2DLandmark.f_right_positionX3", "Het2DLandmark.f_right_positionY3",
                      "Het2DLandmark.f_right_position_valid4", "Het2DLandmark.f_right_positionX4",
                      "Het2DLandmark.f_right_positionY4", "Het2DLandmark.f_right_position_valid5",
                      "Het2DLandmark.f_right_positionX5", "Het2DLandmark.f_right_positionY5",
                      "Het2DLandmark.f_right_position_valid6", "Het2DLandmark.f_right_positionX6",
                      "Het2DLandmark.f_right_positionY6", "Het2DLandmark.f_right_position_valid7",
                      "Het2DLandmark.f_right_positionX7", "Het2DLandmark.f_right_positionY7",
                      "Het2DLandmark.f_stereo_face_plane_nose_residual_mm", "f_headpose_tracking_mode",
                      "S_Head_tracking_status_from_early", "Number_of_eye_blinks_haf", "timestamp_result_receive",
                      "eyeclose_height_left", "eyeclose_height_right"]


    toutput = pd.DataFrame(data=0, index=frameId, columns=target_columns )  # index 지정
    print(toutput)
    # df = df.sort_values(['title', 'point_name', 'number'], ascending=(True, True, True))
    return toutput



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

def convert_xyz_To_hv( pos3D):
    print("need to set Head_dir_h, Head_dir_v -0.287461, 18.055828\n")
    nPos = np.zeros((3, 1))
    nPos = pos3D * deg2Rad
    # daimler    x    y    z axis-> opencv   z   -x     -y
    # daimler roll, pitch, yaw   -> opencv roll -pitch  -yaw
    #opencv     x    y    z axis-> daimler   -y    -z   x
    #opencv  pitch, yaw, roll   -> daimler -pitch -yaw roll
    posNorm = np.linalg.norm( pos3D )
    gaze_vect = pos3D/posNorm
    gH = math.asin( gaze_vect[2] )
    gV = math.atan2( -gaze_vect[1], gaze_vect[0] )
    print('gH, gV', gH, gV)
    gH2 = math.cos( gaze_vect[1] ) * math.sin( gaze_vect[2] )
    gV2 = math.asin( gaze_vect[2] )
    print('gH2, gV2', gH2*rad2Deg, gV2*rad2Deg)

    # gH = math.asin( gaze_vect[0] )
    # gV = math.atan2( -gaze_vect[1], gaze_vect[2] )
    # -0.287461
    # 18.055828
    if ( gV > math.pi / 2 ):
        gV -= math.pi

    if ( gV < -math.pi / 2 ):
        gV += math.pi

    return gH*rad2Deg, gV*rad2Deg

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

def distort(x, y): #input: homogenous or 이미지 센터에서 뺸 후 width.heght 나누기
    r2 = x * x + y * y
    dist = 1.0 + ((distortion[4] * r2 + distortion[1]) * r2 + distortion[0]) * r2
    deltaX = 2.0 * distortion[2] * x * y + distortion[3] * (r2 + 2.0 * x * x)
    deltaY = distortion[2] * (r2 + 2.0 * y * y) + 2.0 * distortion[3] * x * y

    x = x * dist + deltaX
    y = y * dist + deltaY

    return x, y

def modify_frameId_craft_to_daimler(extData, tpd):
    for tindex in extData.index.values:
        tpd.loc[tindex, 'f_frame_counter_left_camera'] = tindex
    print("\n\n")
    # print(tpd)
    return tpd

def modify_HeadObject_from_craft_to_daimler(extData, tpd):
    print('extData.headOri', extData.headOri, '\n\n')
    print('extData.headPos3D', extData.headPos3D, '\n\n')

    for tindex in extData.index.values:
        if (extData.isFaceDetected[tindex] != True):
            tpd.loc[tindex, 'HSVL_MS_S_Head_Pos_Veh_X'] = 'nan'
            tpd.loc[tindex, 'HSVL_MS_S_Head_Pos_Veh_Y'] = 'nan'
            tpd.loc[tindex, 'HSVL_MS_S_Head_Pos_Veh_Z'] = 'nan'
            tpd.loc[tindex, 'S_Head_Pos_Disp_x'] = 'nan'
            tpd.loc[tindex, 'S_Head_Pos_Disp_y'] = 'nan'
            tpd.loc[tindex, 'S_Head_Pos_Disp_z'] = 'nan'
            tpd.loc[tindex, 'HSVL_MS_S_Head_Pos_Disp_X'] = 'nan'
            tpd.loc[tindex, 'HSVL_MS_S_Head_Pos_Disp_Y'] = 'nan'
            tpd.loc[tindex, 'HSVL_MS_S_Head_Pos_Disp_Z'] = 'nan'
            tpd.loc[tindex, 'MS_S_Head_rot_X'] = 'nan'
            tpd.loc[tindex, 'MS_S_Head_rot_Y'] = 'nan'
            tpd.loc[tindex, 'MS_S_Head_rot_Z'] = 'nan'
            tpd.loc[tindex, 'HSVL_S_Head_dir_h'] = 'nan'
            tpd.loc[tindex, 'HSVL_S_Head_dir_v'] = 'nan'
        else:
            tR, tT = changeAxis_opencv2daimler(extData.headOri.loc[tindex], extData.headPos3D.loc[tindex])
            print('tR', tR, 'tT', tT)

            # mat_C2V, _ = transform_A2Bcoord_with_point_Of_Acoord(cam2veh_rot, cam2veh_trans, tT)
            mat_C2V_44, _ = transform_A2Bcoord_with_Object_Of_Acoord(cam2veh_rot, cam2veh_trans, tR, tT.T)
            print("mat_C2V_44", mat_C2V_44)

            cam2disp_rot, cam2disp_trans = transform_3by3_inverse(disp2cam_rot, disp2cam_trans)

            # _, mat_C2D = transform_A2Bcoord_with_point_Of_Acoord(disp2cam_rot, disp2cam_trans, tT)
            mat_C2D_44, _  = transform_A2Bcoord_with_Object_Of_Acoord(cam2disp_rot, cam2disp_trans, tR, tT.T)
            print("mat_C2D_44", mat_C2D_44)

                        # mat_C2V, _ = transform_A2Bcoord_with_point_Of_Acoord(cam2veh_rot, cam2veh_trans, tT)
                        # # mat_C2V_44, _ = transform_A2Bcoord_with_point_Of_Acoord(cam2veh_rot, cam2veh_trans, tR, tT.T)
                        # print("mat_C2V", mat_C2V)
                        #
                        # cam2disp_rot, cam2disp_trans = transform_3by3_inverse(disp2cam_rot, disp2cam_trans)
                        #
                        # mat_C2D, _ = transform_A2Bcoord_with_point_Of_Acoord(cam2disp_rot, cam2disp_trans, tT)
                        # # mat_C2D_44, _ = transform_A2Bcoord_with_Object_Of_Acoord(cam2disp_rot, cam2disp_trans, tR, tT.T)
                        # print("mat_C2D", mat_C2D)
                        #
                        # tpd.loc[tindex, 'HSVL_MS_S_Head_Pos_Veh_X'] = mat_C2V[0]*1000
                        # tpd.loc[tindex, 'HSVL_MS_S_Head_Pos_Veh_Y'] = mat_C2V[1]*1000
                        # tpd.loc[tindex, 'HSVL_MS_S_Head_Pos_Veh_Z'] = mat_C2V[2]*1000
                        # tpd.loc[tindex, 'S_Head_Pos_Disp_x'] = mat_C2D[0]*1000
                        # tpd.loc[tindex, 'S_Head_Pos_Disp_y'] = mat_C2D[1]*1000
                        # tpd.loc[tindex, 'S_Head_Pos_Disp_z'] = mat_C2D[2]*1000
                        # tpd.loc[tindex, 'HSVL_MS_S_Head_Pos_Disp_X'] = np.int64(mat_C2D[0]*1000)
                        # tpd.loc[tindex, 'HSVL_MS_S_Head_Pos_Disp_Y'] = np.int64(mat_C2D[1]*1000)
                        # tpd.loc[tindex, 'HSVL_MS_S_Head_Pos_Disp_Z'] = np.int64(mat_C2D[2]*1000)
                        #
                        # tpd.loc[tindex, 'MS_S_Head_rot_X'] = tR[0]
                        # tpd.loc[tindex, 'MS_S_Head_rot_Y'] = tR[1]
                        # tpd.loc[tindex, 'MS_S_Head_rot_Z'] = tR[2]
                        #
                        # tpd.loc[tindex, 'HSVL_S_Head_dir_h'] = math.atan(mat_C2D[1]/mat_C2D[0]) *rad2Deg
                        # tpd.loc[tindex, 'HSVL_S_Head_dir_v'] = math.atan(mat_C2D[2]/mat_C2D[0]) *rad2Deg



            tpd.loc[tindex, 'HSVL_MS_S_Head_Pos_Veh_X'] = mat_C2V_44[0,3]*1000
            tpd.loc[tindex, 'HSVL_MS_S_Head_Pos_Veh_Y'] = mat_C2V_44[1,3]*1000
            tpd.loc[tindex, 'HSVL_MS_S_Head_Pos_Veh_Z'] = mat_C2V_44[2,3]*1000
            tpd.loc[tindex, 'S_Head_Pos_Disp_x'] = mat_C2D_44[0,3]*1000
            tpd.loc[tindex, 'S_Head_Pos_Disp_y'] = mat_C2D_44[1,3]*1000
            tpd.loc[tindex, 'S_Head_Pos_Disp_z'] = mat_C2D_44[2,3]*1000
            tpd.loc[tindex, 'HSVL_MS_S_Head_Pos_Disp_X'] = np.int64(mat_C2D_44[0,3]*1000)
            tpd.loc[tindex, 'HSVL_MS_S_Head_Pos_Disp_Y'] = np.int64(mat_C2D_44[1,3]*1000)
            tpd.loc[tindex, 'HSVL_MS_S_Head_Pos_Disp_Z'] = np.int64(mat_C2D_44[2,3]*1000)

            tpd.loc[tindex, 'MS_S_Head_rot_X'] = rotationMatrixToEulerAngles(mat_C2V_44[0:3, 0:3])[0]*rad2Deg
            tpd.loc[tindex, 'MS_S_Head_rot_Y'] = rotationMatrixToEulerAngles(mat_C2V_44[0:3, 0:3])[1]*rad2Deg
            tpd.loc[tindex, 'MS_S_Head_rot_Z'] = rotationMatrixToEulerAngles(mat_C2V_44[0:3, 0:3])[2]*rad2Deg

            tpd.loc[tindex, 'HSVL_S_Head_dir_h'] = math.atan(mat_C2D_44[1,3]/mat_C2D_44[0,3]) *rad2Deg
            tpd.loc[tindex, 'HSVL_S_Head_dir_v'] = math.atan(mat_C2D_44[2,3]/mat_C2D_44[0,3]) *rad2Deg


    print('\n', tpd['S_Head_Pos_Disp_x'], '\n', tpd['S_Head_Pos_Disp_y'], '\n', tpd['S_Head_Pos_Disp_z'], '\n')
    print('\n', tpd['HSVL_MS_S_Head_Pos_Disp_X'], '\n', tpd['HSVL_MS_S_Head_Pos_Disp_Y'], '\n', tpd['HSVL_MS_S_Head_Pos_Disp_Z'], '\n')
    print('\n', tpd['HSVL_MS_S_Head_Pos_Veh_X'], '\n', tpd['HSVL_MS_S_Head_Pos_Veh_Y'], '\n', tpd['HSVL_MS_S_Head_Pos_Veh_Z'], '\n')
    print('\n',tpd['MS_S_Head_rot_X'],'\n',tpd['MS_S_Head_rot_Y'],'\n',tpd['MS_S_Head_rot_Z'],'\n')


    print('\n', tpd['HSVL_S_Head_dir_h'], '\n', tpd['HSVL_S_Head_dir_v'], '\n')

    return tpd

def modify_EyeClosureHeight_craft_to_daimler(extData, tpd):
    # print('extData.irisHeight', extData.irisHeight,'\n\n')

    for tindex in extData.index.values:
        # print(extData.irisHeight.loc[tindex])
        tpd.loc[tindex, 'eyeclose_height_left'] = extData.irisHeight.loc[tindex][0]
        tpd.loc[tindex, 'eyeclose_height_right'] = extData.irisHeight.loc[tindex][1]

    print('\n',tpd['eyeclose_height_left'],'\n',tpd['eyeclose_height_right'],'\n')

    return tpd


def modify_frameCounterVirtual_craft_to_daimler(extData, tpd):
    for tindex in extData.index.values:
        tpd.loc[tindex, 'f_frame_counter_virtual'] = tindex - extData.index.values[0] + 1
    print("\n\n")
    print(tpd.f_frame_counter_virtual)
    return tpd

def modify_HeadDetect_craft_to_daimler(extData, tpd):
    for tindex in extData.index.values:
        valueFace = 0
        valueEye = 0
        if(extData.isFaceDetected[tindex]==True):
            valueFace = 1
        if(extData.isLeftGazeValid[tindex]==True):
            valueEye += 1
        if (extData.isRightGazeValid[tindex] == True):
            valueEye += 1
        tpd.loc[tindex, 'CAN_S_face_detection'] = valueFace
        tpd.loc[tindex, 'HSVL_MS_CAN_S_Head_tracking_status'] = valueFace
        tpd.loc[tindex, 'MS_EyeDetection'] = valueEye

    print('\n', tpd['CAN_S_face_detection'], '\n')
    print('\n', tpd['HSVL_MS_CAN_S_Head_tracking_status'], '\n')
    print('\n', tpd['MS_EyeDetection'], '\n')
    return tpd


def modify_EyeGaze_craft_to_daimler(extData, tpd):
    # if(flag_mra2_match == False):

    print('extData.isFusedGazeValid', extData.isFusedGazeValid, '\n\n')
    # print('extData.fusedGazeStart3D', extData.fusedGazeStart3D, '\n\n')
    # print('extData.fusedGazeVector3D', extData.fusedGazeVector3D, '\n\n')
    print('extData.isLeftGazeValid', extData.isLeftGazeValid, '\n\n')
    print('extData.leftGazeStart3D', extData.leftGazeStart3D, '\n\n')
    print('extData.leftGazeVector3D', extData.leftGazeVector3D, '\n\n')
    # print('extData.isRightGazeValid', extData.isRightGazeValid, '\n\n')
    # print('extData.rightGazeStart3D', extData.rightGazeStart3D, '\n\n')
    # print('extData.rightGazeVector3D', extData.rightGazeVector3D, '\n\n')


    for tindex in extData.index.values:
        tpd.loc[tindex, 'MS_S_Gaze_LE_VA_rot_X'] = 'nan'
        tpd.loc[tindex, 'MS_S_Gaze_LE_VA_rot_Y'] = 'nan'
        tpd.loc[tindex, 'MS_S_Gaze_LE_VA_rot_Z'] = 'nan'
        tpd.loc[tindex, 'MS_S_Gaze_LE_Center_X'] = 'nan'
        tpd.loc[tindex, 'MS_S_Gaze_LE_Center_Y'] = 'nan'
        tpd.loc[tindex, 'MS_S_Gaze_LE_Center_Z'] = 'nan'
        tpd.loc[tindex, 'MS_S_Gaze_RE_VA_rot_X'] = 'nan'
        tpd.loc[tindex, 'MS_S_Gaze_RE_VA_rot_Y'] = 'nan'
        tpd.loc[tindex, 'MS_S_Gaze_RE_VA_rot_Z'] = 'nan'
        tpd.loc[tindex, 'MS_S_Gaze_RE_Center_X'] = 'nan'
        tpd.loc[tindex, 'MS_S_Gaze_RE_Center_Y'] = 'nan'
        tpd.loc[tindex, 'MS_S_Gaze_RE_Center_Z'] = 'nan'
        tpd.loc[tindex, 'MS_S_Gaze_rot_X'] = 'nan'
        tpd.loc[tindex, 'MS_S_Gaze_rot_Y'] = 'nan'
        tpd.loc[tindex, 'MS_S_Gaze_rot_Z'] = 'nan'
        tpd.loc[tindex, 'HSVL_MS_CAN_S_Eye_dist'] = 'nan'
        flag_gaze_fused = False
        flag_gaze_left = False
        flag_gaze_right = False
        # "fusedGazeStart3D":
        # "fusedGazeVector":

        if (extData.isLeftGazeValid[tindex] == True):
            flag_gaze_left = True
            tpd.loc[tindex, 'f_gaze_le_result_valid'] = 1
            # tvec2ang = changeRotation_unitvec2radian_check(extData.leftGazeVector3D.loc[tindex])
            tvec2ang = changeRotation_unitvec2radian('PYR', extData.leftGazeVector3D.loc[tindex], 'PYR')

            tR, tT = changeAxis_opencv2daimler(tvec2ang * rad2Deg , extData.leftGazeStart3D.loc[tindex])
            print('Gaze ')
            print('tR', tR, 'tT', tT)
            mat_C2V, _ = transform_A2Bcoord_with_point_Of_Acoord(cam2veh_rot, cam2veh_trans, tT)
            print("mat_C2V", mat_C2V)
            tpd.loc[tindex, 'MS_S_Gaze_LE_VA_rot_X'] = tR[0]
            tpd.loc[tindex, 'MS_S_Gaze_LE_VA_rot_Y'] = tR[1]
            tpd.loc[tindex, 'MS_S_Gaze_LE_VA_rot_Z'] = tR[2]
            tpd.loc[tindex, 'MS_S_Gaze_LE_Center_X'] = mat_C2V[0]*1000
            tpd.loc[tindex, 'MS_S_Gaze_LE_Center_Y'] = mat_C2V[1]*1000
            tpd.loc[tindex, 'MS_S_Gaze_LE_Center_Z'] = mat_C2V[2]*1000

        if (extData.isRightGazeValid[tindex] == True):
            flag_gaze_right = True
            tpd.loc[tindex, 'f_gaze_re_result_valid'] = 1
            # tvec2ang = changeRotation_unitvec2radian_check(extData.rightGazeVector3D.loc[tindex])
            tvec2ang = changeRotation_unitvec2radian('PYR', extData.rightGazeVector3D.loc[tindex], 'PYR')

            tR, tT = changeAxis_opencv2daimler(tvec2ang * rad2Deg, extData.rightGazeStart3D.loc[tindex])
            print('Gaze ')
            print('tR', tR, 'tT', tT)
            mat_C2V, _ = transform_A2Bcoord_with_point_Of_Acoord(cam2veh_rot, cam2veh_trans, tT)
            print("mat_C2V", mat_C2V)
            tpd.loc[tindex, 'MS_S_Gaze_RE_VA_rot_X'] = tR[0]
            tpd.loc[tindex, 'MS_S_Gaze_RE_VA_rot_Y'] = tR[1]
            tpd.loc[tindex, 'MS_S_Gaze_RE_VA_rot_Z'] = tR[2]
            tpd.loc[tindex, 'MS_S_Gaze_RE_Center_X'] = mat_C2V[0]*1000
            tpd.loc[tindex, 'MS_S_Gaze_RE_Center_Y'] = mat_C2V[1]*1000
            tpd.loc[tindex, 'MS_S_Gaze_RE_Center_Z'] = mat_C2V[2]*1000

        if (extData.isLeftGazeValid[tindex] == True and extData.isRightGazeValid[tindex] == True):
            x_d = tpd.loc[tindex, 'MS_S_Gaze_RE_Center_X'] - tpd.loc[tindex, 'MS_S_Gaze_LE_Center_X']
            y_d = tpd.loc[tindex, 'MS_S_Gaze_RE_Center_Y'] - tpd.loc[tindex, 'MS_S_Gaze_LE_Center_Y']
            z_d = tpd.loc[tindex, 'MS_S_Gaze_RE_Center_Z'] - tpd.loc[tindex, 'MS_S_Gaze_LE_Center_Z']
            # x_d = extData.rightGazeStart3D.loc[tindex][0] - extData.leftGazeStart3D.loc[tindex][0]
            # y_d = extData.rightGazeStart3D.loc[tindex][1] - extData.leftGazeStart3D.loc[tindex][1]
            # z_d = extData.rightGazeStart3D.loc[tindex][2] - extData.leftGazeStart3D.loc[tindex][2]
            tpd.loc[tindex, 'HSVL_MS_CAN_S_Eye_dist'] = math.sqrt((x_d*x_d) + (y_d*y_d) + (z_d*z_d))


        if (extData.isFusedGazeValid[tindex] == True):
            flag_gaze_fused = True
            # tvec2ang = changeRotation_unitvec2radian_check(extData.fusedGazeVector3D.loc[tindex])
            tvec2ang = changeRotation_unitvec2radian('PYR', extData.fusedGazeVector3D.loc[tindex], 'PYR')

            tR, tT = changeAxis_opencv2daimler(tvec2ang * rad2Deg, extData.fusedGazeStart3D.loc[tindex])
            print('Gaze ')
            print('tR', tR, 'tT', tT)
            # mat_C2V, _ = transform_A2Bcoord_with_point_Of_Acoord(cam2veh_rot, cam2veh_trans, tT)
            # print("mat_C2V", mat_C2V)
            tpd.loc[tindex, 'MS_S_Gaze_rot_X'] = tR[0]
            tpd.loc[tindex, 'MS_S_Gaze_rot_Y'] = tR[1]
            tpd.loc[tindex, 'MS_S_Gaze_rot_Z'] = tR[2]



    print('\n', tpd['MS_S_Gaze_LE_VA_rot_X'], '\n', tpd['MS_S_Gaze_LE_VA_rot_Y'], '\n', tpd['MS_S_Gaze_LE_VA_rot_Z'], '\n')
    print('\n', tpd['MS_S_Gaze_LE_Center_X'], '\n', tpd['MS_S_Gaze_LE_Center_Y'], '\n', tpd['MS_S_Gaze_LE_Center_Z'], '\n')

    print('\n', tpd['MS_S_Gaze_RE_VA_rot_X'], '\n', tpd['MS_S_Gaze_RE_VA_rot_Y'], '\n', tpd['MS_S_Gaze_RE_VA_rot_Z'], '\n')
    print('\n', tpd['MS_S_Gaze_RE_Center_X'], '\n', tpd['MS_S_Gaze_RE_Center_Y'], '\n', tpd['MS_S_Gaze_RE_Center_Z'], '\n')

    print('\n', tpd['MS_S_Gaze_rot_X'], '\n', tpd['MS_S_Gaze_rot_Y'], '\n', tpd['MS_S_Gaze_rot_Z'], '\n')
    print('\n', tpd['HSVL_MS_CAN_S_Eye_dist'], '\n')

    return tpd

def merge_gazeroi_with_mra2roi(extData, tpd):
    extData['CAN_S_Gaze_ROI']= 0
    extData['MS_S_Gaze_ROI_X_Raw']= 0
    extData['MS_S_Gaze_ROI_Y_Raw']= 0
    for tindex in extData.index.values:
        print(tindex)
        print(tpd.loc[tindex, 'f_frame_counter_left_camera'])
        extData.loc[tindex, 'CAN_S_Gaze_ROI'] = tpd.loc[tindex, 'CAN_S_Gaze_ROI']
        extData.loc[tindex, 'MS_S_Gaze_ROI_X_Raw'] = tpd.loc[tindex, 'MS_S_Gaze_ROI_X_Raw']
        extData.loc[tindex, 'MS_S_Gaze_ROI_Y_Raw'] = tpd.loc[tindex, 'MS_S_Gaze_ROI_Y_Raw']

        # = tindex - extData.index.values[0] + 1
    print("\n\n")
    print(tpd.f_frame_counter_left_camera)
    return tpd

def make_GT_gazeroi(extData, gt_idx):
    extData['gt_s_gaze_roi_das'] = gt_idx
    return extData


# def changeRotation_unitvec2radian_check(nR_unitvec):
#     print("//////////", funcname(), "//////////")
#
#     # nR_unitvec[0] = 0.048387713730335236
#     # nR_unitvec[1] = -0.30887135863304138
#     # nR_unitvec[2] = -0.94987112283706665
#
#     # alpha_yaw = math.atan(nR_unitvec[0] / nR_unitvec[2])
#     # beta_tilt = math.acos(nR_unitvec[1])
#     # print('yaw',math.atan(nR_unitvec[0] / nR_unitvec[2]), alpha_yaw*rad2Deg)
#     # print('tilt',math.acos(nR_unitvec[1]), beta_tilt*rad2Deg)
#     # print('r3', [math.sin(alpha_yaw)*math.sin(beta_tilt), math.cos(beta_tilt), math.cos(alpha_yaw)*math.sin(beta_tilt) ])
#     # print("degree", ret2 * rad2Deg)
#     print('\n\n')
#     alpha_yaw = math.atan(nR_unitvec[0] / nR_unitvec[2])
#     beta_tilt = math.asin(nR_unitvec[1])
#     print('yaw',math.atan(nR_unitvec[0] / nR_unitvec[2]), alpha_yaw*rad2Deg)
#     print('tilt',math.asin(nR_unitvec[1]), beta_tilt*rad2Deg)
#     print('r3', [math.sin(alpha_yaw)*math.cos(beta_tilt), math.sin(beta_tilt), math.cos(alpha_yaw)*math.cos(beta_tilt) ])
#
#     #opencv  pitch, yaw, roll - return sequence
#     #daimler roll, pitch, yaw
#     return np.array([beta_tilt, alpha_yaw, 0])


if __name__ == '__main__':
    # D:/Project/CVT/demo/1_DRCAM_KOR40BU4578_20190219_114431_0002_CTO/HetData_1614867983883741729.json
    # D:/Project/CVT/demo/1_DRCAM_KOR40BU4578_20190219_114431_0002_CTO/Result0001/pose_config.json
    # D:/Project/CVT/demo/1_DRCAM_KOR40BU4578_20190219_114431_0002_CTO/Result0001/display_config.json
    # D:/Project/CVT/demo/2_DRCAM_KOR40BU4578_20190412_092408_0012_CTO/HetData_1614909178201537446.json
    # D:/Project/CVT/demo/2_DRCAM_KOR40BU4578_20190412_092408_0012_CTO/Result0001/pose_config.json
    # D:/Project/CVT/demo/2_DRCAM_KOR40BU4578_20190412_092408_0012_CTO/Result0001/display_config.json
    # D:/Project/CVT/demo/3_DRCAM_MTKLG690_20190226_084627_0016_CTO/HetData_1615223742700610766.json
    # D:/Project/CVT/demo/3_DRCAM_MTKLG690_20190226_084627_0016_CTO/Result0001/pose_config.json
    # D:/Project/CVT/demo/3_DRCAM_MTKLG690_20190226_084627_0016_CTO/Result0001/display_config.json
    # D:/Project/CVT/demo/4_DRCAM_MTKLG690_20180903_170057_0007_CTO/HetData_1614967017269846722.json
    # D:/Project/CVT/demo/4_DRCAM_MTKLG690_20180903_170057_0007_CTO/Result0001/pose_config.json
    # D:/Project/CVT/demo/4_DRCAM_MTKLG690_20180903_170057_0007_CTO/Result0001/display_config.json
    # D:/Project/CVT/demo/5_DRCAM_MTKLG690_20181109_121016_0016_CTO/HetData_1614987673928434516.json
    # D:/Project/CVT/demo/5_DRCAM_MTKLG690_20181109_121016_0016_CTO/Result0001/pose_config.json
    # D:/Project/CVT/demo/5_DRCAM_MTKLG690_20181109_121016_0016_CTO/Result0001/display_config.json
    # D:/Project/CVT/demo/6_DRCAM_MTKLG690_20181107_220127_0006_CTO/HetData_1614985952166460937.json
    # D:/Project/CVT/demo/6_DRCAM_MTKLG690_20181107_220127_0006_CTO/Result0001/pose_config.json
    # D:/Project/CVT/demo/6_DRCAM_MTKLG690_20181107_220127_0006_CTO/Result0001/display_config.json

    if (0):
        sys.stdout = open('DebugLog.txt', 'w')
    intrinsic_matrix = np.array([[-1479.36, 0., 640.91], [0., -1479.39, 488.959], [0., 0., -1.]])
    distortion = np.array([-0.0897939, -0.405596, 0, 0, 0, 0, 0, 0])

    fold_names = filedialog.askdirectory()
    print("select folder", fold_names)
    files_to_replace_base = []

    if (1):
        for dirpath, dirnames, filenames in os.walk(fold_names):
            for filename in [f for f in filenames if f.endswith(".json")]:
                if (filename.__contains__("HetData") == True):
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
    # print(sorted(files_to_replace_base,key=lambda x: str(x).split() ))
    # print(files_to_replace_base)
    # print(1 / 0)

    # "f_version","f_frame_counter_left_camera","f_frame_counter_right_camera","f_frame_counter_virtual","f_master_counter_in","f_eye_model_generation","eye_left_partial_blockage","eye_left_blockage","MS_EyeLeftCameraLeft_old","MS_EyeLeftCameraRight_old","eye_right_partial_blockage","eye_right_blockage","MS_EyeRightCameraLeft_old","MS_EyeRightCameraRight_old","mouth_nose_partial_blockage","mouth_nose_blockage","MS_MouthNoseCameraLeft","MS_MouthNoseCameraRight","CAN_S_glasses_detected_old","CAN_S_face_detection","MS_EyeDetection","MS_S_Head_rot_X","MS_S_Head_rot_Y","MS_S_Head_rot_Z","HSVL_MS_CAN_S_Head_tracking_status","HSVL_MS_CAN_S_Head_tracking_mode","f_primary_face_landmark_camera","f_left_position_valid0","f_left_positionX0","f_left_positionY0","f_left_position_valid1","f_left_positionX1","f_left_positionY1","f_left_position_valid2","f_left_positionX2","f_left_positionY2","f_left_position_valid3","f_left_positionX3","f_left_positionY3","f_left_position_valid4","f_left_positionX4","f_left_positionY4","f_left_position_valid5","f_left_positionX5","f_left_positionY5","f_left_position_valid6","f_left_positionX6","f_left_positionY6","f_right_position_valid0","f_right_positionX0","f_right_positionY0","f_right_position_valid1","f_right_positionX1","f_right_positionY1","f_right_position_valid2","f_right_positionX2","f_right_positionY2","f_right_position_valid3","f_right_positionX3","f_right_positionY3","f_right_position_valid4","f_right_positionX4","f_right_positionY4","f_right_position_valid5","f_right_positionX5","f_right_positionY5","f_right_position_valid6","f_right_positionX6","f_right_positionY6","CAN_glint_detected","f_gaze_le_result_valid","MS_S_Gaze_LE_VA_rot_X","MS_S_Gaze_LE_VA_rot_Y","MS_S_Gaze_LE_VA_rot_Z","MS_S_Gaze_LE_Center_X","MS_S_Gaze_LE_Center_Y","MS_S_Gaze_LE_Center_Z","f_gaze_re_result_valid","MS_S_Gaze_RE_VA_rot_X","MS_S_Gaze_RE_VA_rot_Y","MS_S_Gaze_RE_VA_rot_Z","MS_S_Gaze_RE_Center_X","MS_S_Gaze_RE_Center_Y","MS_S_Gaze_RE_Center_Z","CAN_eye_closure_left","CAN_eye_closure_left_conf","CAN_eye_closure_right","CAN_eye_closure_right_conf","CAN_eye_closure","MS_eye_closed","CAN_long_eyeclosure","f_long_eyeclosure_counter","MS_eye_closed_AAS","MS_PERCLOS_AAS","MS_PERCLOS_AAS_conf","MS_PERCLOS_strict","MS_eye_closed_strict","CAN_eye_blink_conf","MS_Eye_blink_freq_conf","MS_Eye_blink_freq","CAN_eye_blink_t_closing","CAN_eye_blink_t_opening","CAN_eye_blink_duration","CAN_eye_blink_A_closing","CAN_eye_blink_A_opening","CAN_eye_blink_counter","CAN_S_Gaze_ROI","CAN_S_Gaze_ROI_X","CAN_S_Gaze_ROI_Y","f_roi_id","MS_S_Gaze_ROI_X_Raw","MS_S_Gaze_ROI_Y_Raw","HSVL_MS_S_Head_Pos_Veh_X","HSVL_MS_S_Head_Pos_Veh_Y","HSVL_MS_S_Head_Pos_Veh_Z","MS_S_Gaze_rot_X","MS_S_Gaze_rot_Y","MS_S_Gaze_rot_Z","HSVL_MS_CAN_S_Eye_dist","f_camera_left_measured_brightness","f_camera_left_target_brightness","f_camera_left_shutter_us","f_camera_left_column_gain","f_camera_left_digital_gain","f_camera_right_measured_brightness","f_camera_right_target_brightness","f_camera_right_shutter_us","f_camera_right_column_gain","f_camera_right_digital_gain","f_raw_result_age","HSVL_S_Head_dir_h","HSVL_S_Head_dir_v","HSVL_MS_S_Head_Pos_Disp_X","HSVL_MS_S_Head_Pos_Disp_Y","HSVL_MS_S_Head_Pos_Disp_Z","LGE_BD_frame_count","CAN_S_camera_close_blocked","CAN_S_glasses_detected","MS_camera_blockage_detection","MS_EyeLeftCameraLeft","MS_EyeLeftCameraRight","MS_EyeRightCameraLeft","MS_EyeRightCameraRight","MouthNoseCameraLeft","MouthNoseCameraRight","disp_left_cam_blocked_0","disp_left_cam_blocked_1","disp_left_cam_blocked_2","disp_left_cam_blocked_3","disp_right_cam_blocked_0","disp_right_cam_blocked_1","disp_right_cam_blocked_2","disp_right_cam_blocked_3","LGE_OOF_frame_count","bOutOfFocus","CAN_S_drcam_status","LGE_SWBA_frame_count","CAN_S_StWhl_adjust_occlusion","MS_HeadOcclusion","Absorber_Left_Center_nX","Absorber_Left_Center_nY","Absorber_Right_Center_nX","Absorber_Right_Center_nY","nAbsorber_Radius","Wheel_Left_Center_nX","Wheel_Left_Center_nY","Wheel_Right_Center_nX","Wheel_Right_Center_nY","nWheel_Radius","LGE_DB_frame_count","bIsHeadMoving","MS_Intended_head_movement","CAN_Driver_is_responsive","MS_nod_head_gesture","MS_shake_head_gesture","bIsHeadGestureResult","LGE_DI_frame_count","CAN_S_Driver_ID_Top_1","CAN_S_Driver_ID_Top_2","CAN_S_Driver_ID_Top_3","CAN_S_Driver_ID_Confidence_1","CAN_S_Driver_ID_Confidence_2","CAN_S_Driver_ID_Confidence_3","arrDrviers_01","arrDrviers_02","arrDrviers_03","arrDrviers_04","arrDrviers_05","arrDrviers_06","arrDrviers_07","arrDrviers_08","arrDrviers_09","arrDrviers_10","arrDrviers_11","arrDrviers_12","arrDrviers_13","status","numOfDriver","LGE_DOT_frame_count","CAN_S_Head_Pos_X","CAN_S_Head_Pos_Y","CAN_S_Head_Pos_Z","CAN_S_Head_Pos_type","bLeftIRLight","LGE_SP_frame_count","MS_spoofing_detected","CurrentStatus","OutConfidenceSpoof","OutConfidenceGenuine","f_raw_result_age","S_Head_dir_h","S_Head_dir_v","S_Head_Pos_Disp_x","S_Head_Pos_Disp_y","S_Head_Pos_Disp_z","f_left_position_valid7","f_left_positionX7","f_left_positionY7","f_right_position_valid7","f_right_positionX7","f_right_positionY7","FrameHistory","FrameDiff","FrameTimeStamp","f_head_pose_confidence","f_early_head_pose_confidence","LGE_DI_Ext_frame_count","CAN_S_EnrollAndDeleteStatus","CAN_S_HasStoredIDs","CAN_S_driverID_MsgCnt","ResultDataType","timeEnrollment","bFilteringFlag","f_le_iris_diameter","f_re_iris_diameter","HetAlgoGlintPosition.MS_S_Gaze_LE_Cornea_Center_X","HetAlgoGlintPosition.MS_S_Gaze_LE_Cornea_Center_Y","HetAlgoGlintPosition.MS_S_Gaze_LE_Cornea_Center_Z","HetAlgoGlintPosition.f_le_glint_position_idx_0_X","HetAlgoGlintPosition.f_le_glint_position_idx_0_Y","HetAlgoGlintPosition.f_le_glint_position_idx_1_X","HetAlgoGlintPosition.f_le_glint_position_idx_1_Y","HetAlgoGlintPosition.MS_S_Gaze_RE_Cornea_Center_X","HetAlgoGlintPosition.MS_S_Gaze_RE_Cornea_Center_Y","HetAlgoGlintPosition.MS_S_Gaze_RE_Cornea_Center_Z","HetAlgoGlintPosition.f_re_glint_position_idx_0_X","HetAlgoGlintPosition.f_re_glint_position_idx_0_Y","HetAlgoGlintPosition.f_re_glint_position_idx_1_X","HetAlgoGlintPosition.f_re_glint_position_idx_1_Y","HetAlgoEyelidPosition.f_le_lc_position_valid0","HetAlgoEyelidPosition.f_le_lc_positionX0","HetAlgoEyelidPosition.f_le_lc_positionY0","HetAlgoEyelidPosition.f_le_lc_position_valid1","HetAlgoEyelidPosition.f_le_lc_positionX1","HetAlgoEyelidPosition.f_le_lc_positionY1","HetAlgoEyelidPosition.f_le_lc_position_valid2","HetAlgoEyelidPosition.f_le_lc_positionX2","HetAlgoEyelidPosition.f_le_lc_positionY2","HetAlgoEyelidPosition.f_le_lc_position_valid3","HetAlgoEyelidPosition.f_le_lc_positionX3","HetAlgoEyelidPosition.f_le_lc_positionY3","HetAlgoEyelidPosition.f_le_lc_position_valid4","HetAlgoEyelidPosition.f_le_lc_positionX4","HetAlgoEyelidPosition.f_le_lc_positionY4","HetAlgoEyelidPosition.f_le_lc_position_valid5","HetAlgoEyelidPosition.f_le_lc_positionX5","HetAlgoEyelidPosition.f_le_lc_positionY5","HetAlgoEyelidPosition.f_le_lc_position_valid6","HetAlgoEyelidPosition.f_le_lc_positionX6","HetAlgoEyelidPosition.f_le_lc_positionY6","HetAlgoEyelidPosition.f_le_rc_position_valid0","HetAlgoEyelidPosition.f_le_rc_positionX0","HetAlgoEyelidPosition.f_le_rc_positionY0","HetAlgoEyelidPosition.f_le_rc_position_valid1","HetAlgoEyelidPosition.f_le_rc_positionX1","HetAlgoEyelidPosition.f_le_rc_positionY1","HetAlgoEyelidPosition.f_le_rc_position_valid2","HetAlgoEyelidPosition.f_le_rc_positionX2","HetAlgoEyelidPosition.f_le_rc_positionY2","HetAlgoEyelidPosition.f_le_rc_position_valid3","HetAlgoEyelidPosition.f_le_rc_positionX3","HetAlgoEyelidPosition.f_le_rc_positionY3","HetAlgoEyelidPosition.f_le_rc_position_valid4","HetAlgoEyelidPosition.f_le_rc_positionX4","HetAlgoEyelidPosition.f_le_rc_positionY4","HetAlgoEyelidPosition.f_le_rc_position_valid5","HetAlgoEyelidPosition.f_le_rc_positionX5","HetAlgoEyelidPosition.f_le_rc_positionY5","HetAlgoEyelidPosition.f_le_rc_position_valid6","HetAlgoEyelidPosition.f_le_rc_positionX6","HetAlgoEyelidPosition.f_le_rc_positionY6","HetAlgoEyelidPosition.f_re_lc_position_valid0","HetAlgoEyelidPosition.f_re_lc_positionX0","HetAlgoEyelidPosition.f_re_lc_positionY0","HetAlgoEyelidPosition.f_re_lc_position_valid1","HetAlgoEyelidPosition.f_re_lc_positionX1","HetAlgoEyelidPosition.f_re_lc_positionY1","HetAlgoEyelidPosition.f_re_lc_position_valid2","HetAlgoEyelidPosition.f_re_lc_positionX2","HetAlgoEyelidPosition.f_re_lc_positionY2","HetAlgoEyelidPosition.f_re_lc_position_valid3","HetAlgoEyelidPosition.f_re_lc_positionX3","HetAlgoEyelidPosition.f_re_lc_positionY3","HetAlgoEyelidPosition.f_re_lc_position_valid4","HetAlgoEyelidPosition.f_re_lc_positionX4","HetAlgoEyelidPosition.f_re_lc_positionY4","HetAlgoEyelidPosition.f_re_lc_position_valid5","HetAlgoEyelidPosition.f_re_lc_positionX5","HetAlgoEyelidPosition.f_re_lc_positionY5","HetAlgoEyelidPosition.f_re_lc_position_valid6","HetAlgoEyelidPosition.f_re_lc_positionX6","HetAlgoEyelidPosition.f_re_lc_positionY6","HetAlgoEyelidPosition.f_re_rc_position_valid0","HetAlgoEyelidPosition.f_re_rc_positionX0","HetAlgoEyelidPosition.f_re_rc_positionY0","HetAlgoEyelidPosition.f_re_rc_position_valid1","HetAlgoEyelidPosition.f_re_rc_positionX1","HetAlgoEyelidPosition.f_re_rc_positionY1","HetAlgoEyelidPosition.f_re_rc_position_valid2","HetAlgoEyelidPosition.f_re_rc_positionX2","HetAlgoEyelidPosition.f_re_rc_positionY2","HetAlgoEyelidPosition.f_re_rc_position_valid3","HetAlgoEyelidPosition.f_re_rc_positionX3","HetAlgoEyelidPosition.f_re_rc_positionY3","HetAlgoEyelidPosition.f_re_rc_position_valid4","HetAlgoEyelidPosition.f_re_rc_positionX4","HetAlgoEyelidPosition.f_re_rc_positionY4","HetAlgoEyelidPosition.f_re_rc_position_valid5","HetAlgoEyelidPosition.f_re_rc_positionX5","HetAlgoEyelidPosition.f_re_rc_positionY5","HetAlgoEyelidPosition.f_re_rc_position_valid6","HetAlgoEyelidPosition.f_re_rc_positionX6","HetAlgoEyelidPosition.f_re_rc_positionY6","Het2DLandmark.f_left_position_valid0","Het2DLandmark.f_left_positionX0","Het2DLandmark.f_left_positionY0","Het2DLandmark.f_left_position_valid1","Het2DLandmark.f_left_positionX1","Het2DLandmark.f_left_positionY1","Het2DLandmark.f_left_position_valid2","Het2DLandmark.f_left_positionX2","Het2DLandmark.f_left_positionY2","Het2DLandmark.f_left_position_valid3","Het2DLandmark.f_left_positionX3","Het2DLandmark.f_left_positionY3","Het2DLandmark.f_left_position_valid4","Het2DLandmark.f_left_positionX4","Het2DLandmark.f_left_positionY4","Het2DLandmark.f_left_position_valid5","Het2DLandmark.f_left_positionX5","Het2DLandmark.f_left_positionY5","Het2DLandmark.f_left_position_valid6","Het2DLandmark.f_left_positionX6","Het2DLandmark.f_left_positionY6","Het2DLandmark.f_left_position_valid7","Het2DLandmark.f_left_positionX7","Het2DLandmark.f_left_positionY7","Het2DLandmark.f_right_position_valid0","Het2DLandmark.f_right_positionX0","Het2DLandmark.f_right_positionY0","Het2DLandmark.f_right_position_valid1","Het2DLandmark.f_right_positionX1","Het2DLandmark.f_right_positionY1","Het2DLandmark.f_right_position_valid2","Het2DLandmark.f_right_positionX2","Het2DLandmark.f_right_positionY2","Het2DLandmark.f_right_position_valid3","Het2DLandmark.f_right_positionX3","Het2DLandmark.f_right_positionY3","Het2DLandmark.f_right_position_valid4","Het2DLandmark.f_right_positionX4","Het2DLandmark.f_right_positionY4","Het2DLandmark.f_right_position_valid5","Het2DLandmark.f_right_positionX5","Het2DLandmark.f_right_positionY5","Het2DLandmark.f_right_position_valid6","Het2DLandmark.f_right_positionX6","Het2DLandmark.f_right_positionY6","Het2DLandmark.f_right_position_valid7","Het2DLandmark.f_right_positionX7","Het2DLandmark.f_right_positionY7","Het2DLandmark.f_stereo_face_plane_nose_residual_mm","f_headpose_tracking_mode","S_Head_tracking_status_from_early","Number_of_eye_blinks_haf","timestamp_result_receive","eyeclose_height_left","eyeclose_height_right"

    inputPath_HET = "./refer/gaze_GT/01_display_driver/EVA2DAS_01a_DRCAM_CONT_20200120_104521_0000/HetData_1634598329073106478.json"
    # inputPath_HET = "D:/Project/CVT/demo/result101/HetData_01.json"
    # inputPath_HET = "D:/Project/CVT/성능비교/HetData_test.json"
    # inputPath_HET = "./input/HetData_mid.json"

    inputPath_C2V = "./refer/gaze_config/pose_config.json"
    # inputPath_C2V = "D:/Project/CVT/성능비교/DRCAM_KOR40BU4578_20190214_113824_0021_2/Result0001_20210205/pose_config.json"
    # pose_config.json

    inputPath_D2V = "./refer/gaze_config/display_config.json"
    # inputPath_D2V = "D:/Project/CVT/성능비교/DRCAM_KOR40BU4578_20190214_113824_0021_2/Result0001_20210205/display_config.json"
    # display_config.json

    inputPath_D2C = ""
    # "D:/Project/CVT/성능비교/DRCAM_KOR40BU4578_20190214_113824_0021_2/Result0001_20210205/CamToDisplay_config.json"
    # CamToDisplay_config.json

    inputPath_ROI = "./refer/gaze_config/roi_config_eva5.json"
    # roi_config.json

    # ret = load_jsonfile_preValue(inputPath_C2V, inputPath_D2V, inputPath_D2C)
    ret = load_jsonfile_preValue_extend(inputPath_C2V, inputPath_D2V)

    df_merge = pd.DataFrame()
    if(0):
        ret_json = load_jsonfile_HET(inputPath_HET)
        ret_ExtData = extract_availData_from_craft_algo(ret_json)

        # offset설정만큼 끝애서 잘라냄.
        ret_ExtData = ret_ExtData[-FRAMECNT_CROP_CHUNK:]

        print(ret_ExtData.index.values)
        # print(ret_ExtData.index.values[-40:])

        tout = make_prototype_on_pandas(ret_ExtData.index.values)
        # todo : 데이터 업데이트 추가 필요

        tout = modify_frameId_craft_to_daimler(ret_ExtData, tout)
        tout = modify_HeadObject_from_craft_to_daimler(ret_ExtData, tout)

        tout = modify_EyeClosureHeight_craft_to_daimler(ret_ExtData, tout)
        tout = modify_frameCounterVirtual_craft_to_daimler(ret_ExtData, tout)
        tout = modify_HeadDetect_craft_to_daimler(ret_ExtData, tout)

        tout = modify_EyeGaze_craft_to_daimler(ret_ExtData, tout)

        df_merge = pd.concat([df_merge, tout]).reset_index(drop=True)
    else:
        for i, tname in enumerate(files_to_replace_base):
            print(i , tname)
            ret_json = load_jsonfile_HET(tname)
            ret_ExtData = extract_availData_from_craft_algo(ret_json)

            # #offset설정만큼 끝애서 잘라냄.
            ret_ExtData = ret_ExtData[-FRAMECNT_CROP_CHUNK:]
            print(ret_ExtData.index.values)
            # print(ret_ExtData.index.values[-40:])
            #
            tout = make_prototype_on_pandas(ret_ExtData.index.values)
            #todo : 데이터 업데이트 추가 필요

            tout = modify_frameId_craft_to_daimler(ret_ExtData, tout)
            tout = modify_HeadObject_from_craft_to_daimler(ret_ExtData, tout)

            tout = modify_EyeClosureHeight_craft_to_daimler(ret_ExtData, tout)
            tout = modify_frameCounterVirtual_craft_to_daimler(ret_ExtData, tout)
            tout = modify_HeadDetect_craft_to_daimler(ret_ExtData, tout)

            tout = modify_EyeGaze_craft_to_daimler(ret_ExtData, tout)

            tempIdx = tname.rfind('EVA2DAS_')
            if (tempIdx == -1):
                print("This file{} is not exist GT", tname)
                print("Exit!!")
                print(1 / 0)
            # print('tempIdx', tempIdx)
            gt_value = int(tname[tempIdx + len('EVA2DAS_'):tempIdx + len('EVA2DAS_') + 2:])
            print(gt_value)
            tout = make_GT_gazeroi(tout, gt_value)

            #파일의 상위폴더 split하여 저장함 ex) EVA2DAS_01a_DRCAM_CONT_20200120_104521_0000
            tout['Load_file'] =  os.path.dirname(tname).replace("\\","/").split(sep='/')[-1]
            df_merge = pd.concat([df_merge, tout]).reset_index(drop=True)


    save_csvfile(df_merge, fold_names + "/" + "extract_output.csv")
    print(1/0)

    if (1):
        #inputPath_GT = "./refer/GT_3531_96_670222_0001_all.csv"
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

        # inputPath_ROI = "./refer/roi_config.json"
        # roi_config.json
        # obj = make_gaze_and_roi()
        objgaze = mg.make_gaze_and_roi()
        ret_roi = objgaze.load_jsonfile_ROI(inputPath_ROI)

        ret_ExtROI = objgaze.extract_availData_from_3D_target_ROI(ret_roi)

        # ret_ExtGT = objgaze.extract_availData_from_GT(inputPath_GT)
        ret_df_tout = objgaze.extract_availData_from_pandas(tout)
        # print('ret_df_tout\n\n', ret_df_tout)
        # print(1/0)
        # save_csvfile(ret_df_tout, "./final_output.csv")
        # print(1/0)

        ret_ExtGT_with_direction = objgaze.retcalcuate_head_eye_direction(ret_df_tout)

        print('\n\n', ret_ExtGT_with_direction)
        ret_match = objgaze.check_match_roi_cto(ret_ExtGT_with_direction, ret_ExtROI, 100)
        ret_match = ret_match.drop(columns=['CAN_S_Gaze_ROI', 'MS_S_Gaze_ROI_X_Raw', 'MS_S_Gaze_ROI_Y_Raw'])

        tempIdx = inputPath_HET.rfind('EVA2DAS_')
        if (tempIdx == -1):
            print("This file{} is not exist GT", inputPath_HET)
            print("Exit!!")
            print(1 / 0)
        # print('tempIdx', tempIdx)
        gt_value = int(inputPath_HET[tempIdx + len('EVA2DAS_'):tempIdx + len('EVA2DAS_') + 2:])
        print(gt_value)
        ret_match = make_GT_gazeroi(ret_match, gt_value)

        objgaze.save_csvfile(ret_match, "./roi_output.csv")
        print(1/0)

        #GT값을 name에서 추출하자. 값을 추출하자
        ret_resultGT = objgaze.extract_resultRoi_from_GT(inputPath_GT, "MRA2_")

        ret_roi_result = pd.merge(ret_match, ret_resultGT, how='left', left_on="f_frame_counter_left_camera", right_on="f_frame_counter_left_camera")
        # test = pd.concat([ret_match, ret_resultGT], axis=1)
        print(ret_roi_result)
        # print(1/0)

	    # merge_gazeroi_with_mra2roi(ret_match, ret_resultGT)
	    # print(test)
	    # "f_frame_counter_left_camera"
	    # ret_match.merge(ret_resultGT, on="f_frame_counter_left_camera")
        objgaze.save_csvfile(ret_roi_result, "./roi_output.csv")
        # ret_match.to_csv("filename.csv", mode='w', index=False, header=False, sep=',', quotechar=" ",
        #                  float_format='%.4f')

        # objgaze.rendering_roi_with_head_gaze(ret_ExtROI, ret_match)


    print(1/0)



    # print(1 / 0)


