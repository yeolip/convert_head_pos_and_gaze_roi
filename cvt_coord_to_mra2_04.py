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


#mra2 세팅에 맞춰서 저장함
flag_mra2_match = False

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

deg2Rad = math.pi/180
rad2Deg = 180/math.pi

# master camera point on Vehicle coord
cam2veh_rot = np.array([0, -11, 0.99999999999999978])
cam2veh_trans = [1.0058888249122007, -0.35483707652634749, 0.68375427481211271]

# display center point on Vehicle coord
disp2veh_rot = np.array([0, -8, 1])
disp2veh_trans = [1.0252906, -0.4003454, 0.6392974]

# display center point on Camera coord
disp2cam_rot = np.array([0.0, 3.0, 0.0])
disp2cam_trans = [0.00978, -0.04584, -0.04719]

def funcname():
    return sys._getframe(1).f_code.co_name + "()"

def callername():
    return sys._getframe(2).f_code.co_name + "()"

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

'''reverse direction_from A to B -> from B to A'''
def transform_3by3_inverse(nR, nT):
    t_matrix_1 = np.eye(4)
    s_trot = np.zeros((3, 1))
    s_trot[0] = deg2Rad * nR[0]
    s_trot[1] = deg2Rad * nR[1]
    s_trot[2] = deg2Rad * nR[2]
    t_matrix_1[0:3, 0:3] = eulerAnglesToRotationMatrix(s_trot)
    t_matrix_1[0:3, 3] = nT
    mat_result_inv = np.linalg.inv(t_matrix_1)

    return (rotationMatrixToEulerAngles(mat_result_inv[0:3, 0:3]) * rad2Deg), (mat_result_inv[0:3, 3])

'''transform point on A coordination to B coordination'''
def transform_A2Bcoord_with_point_Of_Acoord(nR, nT, pos3D):
    t_matrix = np.eye(4)
    s_trot = np.zeros((3, 1))
    s_trot[0] = deg2Rad * nR[0]
    s_trot[1] = deg2Rad * nR[1]
    s_trot[2] = deg2Rad * nR[2]
    t_matrix[0:3, 0:3] = eulerAnglesToRotationMatrix(s_trot)
    t_matrix[0:3, 3] = nT
    t_matrix_inv = np.linalg.inv(t_matrix)

    s_pos3D = np.ones((4, 1))
    # print("pos3D", pos3D)
    s_pos3D[0] = pos3D[0]
    s_pos3D[1] = pos3D[1]
    s_pos3D[2] = pos3D[2]
    # print("s_pos3D", s_pos3D)
    mat_result = np.matmul(t_matrix, s_pos3D)
    mat_result_inv = np.matmul(t_matrix_inv, s_pos3D)
    return mat_result[0:3,0], mat_result_inv[0:3,0]

'''make transform-matrix about A2B2C that mean from A to C coordination '''
def transform_A2Bcoord_and_B2Ccoord(nR_A2B, nT_A2B, nR2_B2C, nT2_B2C):
    t_matrix_1 = np.eye(4)
    s_trot = np.zeros((3, 1))
    s_trot[0] = deg2Rad * nR_A2B[0]
    s_trot[1] = deg2Rad * nR_A2B[1]
    s_trot[2] = deg2Rad * nR_A2B[2]
    t_matrix_1[0:3, 0:3] = eulerAnglesToRotationMatrix(s_trot)
    t_matrix_1[0:3, 3] = nT_A2B

    t_matrix_2 = np.eye(4)
    s_trot2 = np.zeros((3, 1))
    s_trot2[0] = deg2Rad * nR2_B2C[0]
    s_trot2[1] = deg2Rad * nR2_B2C[1]
    s_trot2[2] = deg2Rad * nR2_B2C[2]
    t_matrix_2[0:3, 0:3] = eulerAnglesToRotationMatrix(s_trot2)
    t_matrix_2[0:3, 3] = nT2_B2C

    mat_result = np.matmul(t_matrix_2, t_matrix_1)
    mat_result_inv = np.linalg.inv(mat_result)
    return mat_result, mat_result_inv

'''transform object(included R&T) on A coordination to B coordination'''
def transform_A2Bcoord_with_Object_Of_Acoord(nR, nT, nObjR2, nObjT2):
    t_matrix_1 = np.eye(4)
    s_trot = np.zeros((3, 1))
    s_trot[0] = deg2Rad * nR[0]
    s_trot[1] = deg2Rad * nR[1]
    s_trot[2] = deg2Rad * nR[2]
    t_matrix_1[0:3, 0:3] = eulerAnglesToRotationMatrix(s_trot)
    t_matrix_1[0:3, 3] = nT

    t_matrix_2 = np.eye(4)
    s_trot2 = np.zeros((3, 1))
    s_trot2[0] = deg2Rad * nObjR2[0]
    s_trot2[1] = deg2Rad * nObjR2[1]
    s_trot2[2] = deg2Rad * nObjR2[2]
    t_matrix_2[0:3, 0:3] = eulerAnglesToRotationMatrix(s_trot2)
    t_matrix_2[0:3, 3] = nObjT2

    mat_result = np.matmul(t_matrix_1, t_matrix_2)
    mat_result_inv = np.linalg.inv(mat_result)
    return mat_result, mat_result_inv

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

def rotateX(vect, angle):
    matrix = np.array([[1, 0, 0],
                       [0, np.cos(angle), -np.sin(angle)],
                       [0, np.sin(angle), np.cos(angle)]])
    return np.matmul(matrix, vect)


def rotateY(vect, angle):
    matrix = np.array([[np.cos(angle), 0, np.sin(angle)],
                       [0.0, 1.0, 0.0],
                       [-np.sin(angle), 0.0, np.cos(angle)]])
    return np.matmul(matrix, vect)


def rotateZ(vect, angle):
    matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                       [np.sin(angle), np.cos(angle), 0],
                       [0, 0, 1]])
    return np.matmul(matrix, vect)


def distort(x, y): #input: homogenous or 이미지 센터에서 뺸 후 width.heght 나누기
    r2 = x * x + y * y
    dist = 1.0 + ((distortion[4] * r2 + distortion[1]) * r2 + distortion[0]) * r2
    deltaX = 2.0 * distortion[2] * x * y + distortion[3] * (r2 + 2.0 * x * x)
    deltaY = distortion[2] * (r2 + 2.0 * y * y) + 2.0 * distortion[3] * x * y

    x = x * dist + deltaX
    y = y * dist + deltaY

    return x, y


def applyPoseRotation(vect, pitch, yaw, roll):
    vect = rotateZ(vect, roll)
    vect = rotateY(vect, yaw)
    vect = rotateX(vect, pitch)

    return vect


def applyPoseRotation_ZXY(vect, pitch, yaw, roll):
    vect = rotateY(vect, yaw)
    vect = rotateX(vect, pitch)
    vect = rotateZ(vect, roll)

    return vect


def applyPoseTranslation(vect, translationVect):
    vect += translationVect

    return vect

# Calculates Rotation Matrix given euler angles.
def eulerAnglesToRotationMatrix(theta):
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])

    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])

    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])

    R = np.dot(R_z, np.dot(R_y, R_x))

    return R

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

            # cam->disp
            #
            # transform_A2Bcoord_with_point_Of_Acoord(nR, nT, pos3D):
            # cam->veh
            # mat_C2V, _ = transform_A2Bcoord_with_point_Of_Acoord(cam2veh_rot, cam2veh_trans, tT)
            mat_C2V_44, _ = transform_A2Bcoord_with_Object_Of_Acoord(cam2veh_rot, cam2veh_trans, tR, tT.T)
            print("mat_C2V_44", mat_C2V_44)

            cam2disp_rot, cam2disp_trans = transform_3by3_inverse(disp2cam_rot, disp2cam_trans)

            # _, mat_C2D = transform_A2Bcoord_with_point_Of_Acoord(disp2cam_rot, disp2cam_trans, tT)
            mat_C2D_44, _  = transform_A2Bcoord_with_Object_Of_Acoord(cam2disp_rot, cam2disp_trans, tR, tT.T)
            print("mat_C2D_44", mat_C2D_44)

            # cam->disp->veh

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

if __name__ == '__main__':

    inputPath_HET = "D:/Project/CVT/demo/result101/HetData_01.json"
    inputPath_C2V = "D:/Project/CVT/demo/Standalone_Player/1_DRCAM_KOR40BU4578_20190219_114431/pose_config.json"
    # inputPath_C2V = "D:/Project/CVT/성능비교/DRCAM_KOR40BU4578_20190214_113824_0021_2/Result0001_20210205/pose_config.json"
    # pose_config.json

    inputPath_D2V = "D:/Project/CVT/demo/Standalone_Player/1_DRCAM_KOR40BU4578_20190219_114431/display_config.json"
    # inputPath_D2V = "D:/Project/CVT/성능비교/DRCAM_KOR40BU4578_20190214_113824_0021_2/Result0001_20210205/display_config.json"
    # display_config.json

    inputPath_D2C = ""
    # "D:/Project/CVT/성능비교/DRCAM_KOR40BU4578_20190214_113824_0021_2/Result0001_20210205/CamToDisplay_config.json"
    # CamToDisplay_config.json

    # inputPath_ROI = "D:/Project/CVT/demo/Standalone_Player/1_DRCAM_KOR40BU4578_20190219_114431/roi_config.json"
    # roi_config.json

    # ret = load_jsonfile_preValue(inputPath_C2V, inputPath_D2V, inputPath_D2C)
    ret = load_jsonfile_preValue_extend(inputPath_C2V, inputPath_D2V)

    ret_json = load_jsonfile_HET(inputPath_HET)
    ret_ExtData = extract_availData_from_craft_algo(ret_json)

    print(ret_ExtData.index.values)
    tout = make_prototype_on_pandas(ret_ExtData.index.values)
    #todo : 데이터 업데이트 추가 필요

    tout = modify_frameId_craft_to_daimler(ret_ExtData, tout)
    tout = modify_HeadObject_from_craft_to_daimler(ret_ExtData, tout)

    tout = modify_EyeClosureHeight_craft_to_daimler(ret_ExtData, tout)
    tout = modify_frameCounterVirtual_craft_to_daimler(ret_ExtData, tout)
    tout = modify_HeadDetect_craft_to_daimler(ret_ExtData, tout)

    save_csvfile(tout, "./final_output.csv")


