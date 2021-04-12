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

def extract_availData_from_mra2_ROI(pJson):
    pass

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

def changeRotation_radian2unitvec(nR_rad):
    print("//////////", funcname(), "//////////")
    nR_unitvec = np.zeros((3, 1))
    print("input radian", nR_rad)
    # if(nR_rad[0])<0:
    #     nR_rad[0] += (math.pi/2)
    # if(nR_rad[1])<0:
    #     nR_rad[1] += (math.pi/2)
    # if(nR_rad[2])<0:
    #     nR_rad[2] += (math.pi/2)


    if (nR_rad[0] > (math.pi / 2)):
            nR_rad[0] -= math.pi
    elif (nR_rad[0] < (-math.pi / 2)):
            nR_rad[0] += math.pi
    if (nR_rad[1] > (math.pi / 2)):
            nR_rad[1] -= math.pi
    elif (nR_rad[1] < (-math.pi / 2)):
            nR_rad[1] += math.pi
    if (nR_rad[2] > (math.pi / 2)):
            nR_rad[2] -= math.pi
    elif (nR_rad[2] < (-math.pi / 2)):
            nR_rad[2] += math.pi


    nR_unitvec[0] = math.cos(nR_rad[0])
    nR_unitvec[1] = math.cos(nR_rad[1])
    nR_unitvec[2] = math.cos(nR_rad[2])

    dist = nR_unitvec[0]*nR_unitvec[0] + nR_unitvec[1]*nR_unitvec[1] + nR_unitvec[2]*nR_unitvec[2]
    # print('dist', dist)
    nR_unitvec[0] /= math.sqrt(dist)
    nR_unitvec[1] /= math.sqrt(dist)
    nR_unitvec[2] /= math.sqrt(dist)
    #
    # dist2 = nR_unitvec[0]*nR_unitvec[0] + nR_unitvec[1]*nR_unitvec[1] + nR_unitvec[2]*nR_unitvec[2]

    # if(int(dist2) == 1):
    #     print("is Unitvector OK")
    # nR_unitvec= (1,1,1)
    print('nR_unitvec', nR_unitvec)
    return nR_unitvec

def changeRotation_unitvec2radian(nR_unitvec):
    print("//////////", funcname(), "//////////")
    nR_rad = np.zeros((3, 1))
    #y = arctan(x)  #−π/2 < y < π/2
    #y = arccos(x)  #0 ≤ y ≤ π

    dist = nR_unitvec[0]*nR_unitvec[0] + nR_unitvec[1]*nR_unitvec[1] + nR_unitvec[2]*nR_unitvec[2]
    # print(dist, np.round(dist))
    if(np.round(dist) == 1):
        print("is Unitvector OK")

    nR_rad[0] = math.acos(nR_unitvec[0]/dist)
    nR_rad[1] = math.acos(nR_unitvec[1]/dist)
    nR_rad[2] = math.acos(nR_unitvec[2]/dist)

    # if (nR_rad[0] > (math.pi / 2)):
    #         nR_rad[0] -= math.pi
    # elif (nR_rad[0] < (-math.pi / 2)):
    #         nR_rad[0] += math.pi
    # if (nR_rad[1] > (math.pi / 2)):
    #         nR_rad[1] -= math.pi
    # elif (nR_rad[1] < (-math.pi / 2)):
    #         nR_rad[1] += math.pi
    # if (nR_rad[2] > (math.pi / 2)):
    #         nR_rad[2] -= math.pi
    # elif (nR_rad[2] < (-math.pi / 2)):
    #         nR_rad[2] += math.pi


    print('nR_rad', nR_rad)
    return nR_rad

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

def transformation_camera2vehicle(position3D_cam):
    temp_mat = np.zeros_like(position3D_cam)

    temp_mat[0] = position3D_cam[2]
    temp_mat[1] = position3D_cam[0]
    temp_mat[2] = position3D_cam[1]
    mat_result = np.matmul(compute_rotation_matrix(cam2veh_rot), temp_mat) + cam2veh_trans
    return mat_result

def transform_AtoB(nR, nT, pos3D):
    temp_mat = np.zeros_like(pos3D)

    # temp_mat[0] = pos3D[2]
    # temp_mat[1] = pos3D[0]
    # temp_mat[2] = pos3D[1]
    temp_mat[0] = pos3D[0]/1000
    temp_mat[1] = pos3D[1]/1000
    temp_mat[2] = pos3D[2]/1000
    mat_result = np.matmul(compute_rotation_matrix(nR), temp_mat) + nT

    return mat_result


def compute_rotation_matrix(angle):

    cosx = np.cos(angle[0])
    cosy = np.cos(angle[1])
    cosz = np.cos(angle[2])

    sinx = np.sin(angle[0])
    siny = np.sin(angle[1])
    sinz = np.sin(angle[2])

    matrix_r = np.zeros((3, 3))
    matrix_r[0][0] = cosz * cosy
    matrix_r[0][1] = (-sinz * cosx) + (cosz * siny * sinx)
    matrix_r[0][2] = (sinz * sinx) + (cosz * siny * cosx)

    matrix_r[1][0] = sinz * cosy
    matrix_r[1][1] = (cosz * cosx) + (sinz * siny * sinx)
    matrix_r[1][2] = (-cosz * sinx) + (sinz * siny * cosx)

    matrix_r[2][0] = -siny
    matrix_r[2][1] = cosy * sinx
    matrix_r[2][2] = cosy * cosx

    return matrix_r


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
            tvec2ang = changeRotation_unitvec2radian_check(extData.leftGazeVector3D.loc[tindex])

            tR, tT = changeAxis_opencv2daimler(tvec2ang * rad2Deg, extData.leftGazeStart3D.loc[tindex])
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
            tvec2ang = changeRotation_unitvec2radian_check(extData.rightGazeVector3D.loc[tindex])
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
            tvec2ang = changeRotation_unitvec2radian_check(extData.fusedGazeVector3D.loc[tindex])
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

def changeRotation_unitvec2radian_check(nR_unitvec):
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
    print('\n\n')
    alpha_yaw = math.atan(nR_unitvec[0] / nR_unitvec[2])
    beta_tilt = math.asin(nR_unitvec[1])
    print('yaw',math.atan(nR_unitvec[0] / nR_unitvec[2]), alpha_yaw*rad2Deg)
    print('tilt',math.asin(nR_unitvec[1]), beta_tilt*rad2Deg)
    print('r3', [math.sin(alpha_yaw)*math.cos(beta_tilt), math.sin(beta_tilt), math.cos(alpha_yaw)*math.cos(beta_tilt) ])

    #opencv  pitch, yaw, roll - return sequence
    #daimler roll, pitch, yaw
    return np.array([beta_tilt, alpha_yaw, 0])
    # return np.array([alpha_yaw, beta_tilt, 0])

def intersectionWithPlan(linePoint, lineDir, planOrth, planPoint):
    d = np.dot(np.subtract(linePoint, planPoint), planOrth) / (np.dot(lineDir, planOrth))
    intersectionPoint = np.subtract(np.multiply(d, lineDir), linePoint)
    return intersectionPoint

if __name__ == '__main__':
    intrinsic_matrix = np.array([[-1479.36, 0., 640.91], [0., -1479.39, 488.959], [0., 0., -1.]])
    distortion = np.array([-0.0897939, -0.405596, 0, 0, 0, 0, 0, 0])

    #head pos on vehicle coord
    pHead_on_veh = np.array([1504, -395, 874, 1000 ]) / 1000
    #head pos on disp coord
    pHead_on_disp = np.array([507.038195, -2.543902, 165.293075, 1000 ]) / 1000
    # param2 = np.array([507, -3, 165, 1000]) / 1000

    # disply coord to vehicle coord
    t_matrix = np.eye(4)
    s_trot = np.zeros((3, 1))
    s_trot[0] = deg2Rad * disp2veh_rot[0]
    s_trot[1] = deg2Rad * disp2veh_rot[1]
    s_trot[2] = deg2Rad * disp2veh_rot[2]
    t_matrix[0:3, 0:3] = eulerAnglesToRotationMatrix(s_trot)
    t_matrix[0:3, 3] = disp2veh_trans
    # print('t_matrix',t_matrix)
    t_matrix_inv = np.linalg.inv(t_matrix)
    # print('t_matrix_inv', t_matrix_inv)

    print('disp2veh_coord ret_pHead',np.matmul(t_matrix, pHead_on_disp )* 1000)
    t_matrix_inv = np.linalg.inv(t_matrix)
    print('veh2disp_coord ret_pHead', np.matmul(t_matrix_inv, pHead_on_veh )* 1000)
    print("//////////////////////////////\n\n")

    # head pose from disply coord to camera cood to vehicle cood
    t_matrix_1 = np.eye(4)
    s_trot = np.zeros((3, 1))
    s_trot[0] = deg2Rad * disp2cam_rot[0]
    s_trot[1] = deg2Rad * disp2cam_rot[1]
    s_trot[2] = deg2Rad * disp2cam_rot[2]
    t_matrix_1[0:3, 0:3] = eulerAnglesToRotationMatrix(s_trot)
    t_matrix_1[0:3, 3] = disp2cam_trans

    t_matrix_2 = np.eye(4)
    s_trot2 = np.zeros((3, 1))
    s_trot2[0] = deg2Rad * cam2veh_rot[0]
    s_trot2[1] = deg2Rad * cam2veh_rot[1]
    s_trot2[2] = deg2Rad * cam2veh_rot[2]
    t_matrix_2[0:3, 0:3] = eulerAnglesToRotationMatrix(s_trot2)
    t_matrix_2[0:3, 3] = cam2veh_trans


    print('disp2cam_coord ret_pHead',np.matmul(t_matrix_1, pHead_on_disp )* 1000)

    print('disp2cam2veh_coord ret_pHead', np.matmul(np.matmul(t_matrix_2, t_matrix_1), pHead_on_disp ) * 1000)
    t_matrix_out_inv = np.linalg.inv(np.matmul(t_matrix_2, t_matrix_1))
    print('veh2cam2disp_coord ret_pHead', np.matmul(t_matrix_out_inv, pHead_on_veh) * 1000)


    print("/////////////////////////\n\n")

    print("만약 camera에서 vehicle과 disp coord로 변환한다면\n")


    print("opencv axis to daimler axis로 변환한다면\n")

    # print('t_matrix_1',t_matrix_1,'\nt_matrix_2',t_matrix_2)
    # t_matrix_out = np.matrix(t_matrix_2 * 1000 * t_matrix_1)
    # print('t_matrix_out', t_matrix_out)
    # t_matrix_out_inv = np.linalg.inv(t_matrix_out)
    # print('t_matrix_out_inv', t_matrix_out_inv)
    #
    # print("\n\n")
    # print('ret_t_matrix', np.matmul(t_matrix_out, param2) * 1000)
    # t_matrix_out_inv = np.linalg.inv(t_matrix_out)
    # print('ret_t_matrix_inv', np.matmul(t_matrix_out_inv, param) * 1000)




    # euler = rotationMatrixToEulerAngles(t_matrix_inv[0:3, 0:3]) * radToDeg
    # tranx = t_matrix_inv[0][3]
    # trany = t_matrix_inv[1][3]
    # tranz = t_matrix_inv[2][3]

    # res = transform_AtoB(cam2veh_rot, cam2veh_trans, param)
    # print('res', res)
    #



    # 507.038195
    # - 2.543902
    # 165.293075
    param2 = np.array([507.038195, -2.543902, 165.293075])
    res2 = transform_AtoB(disp2veh_rot, disp2veh_trans, param2)
    print('res2', res2)

    param3 = np.array([1504, -395, 874])
    res3 = transform_AtoB(cam2veh_rot, cam2veh_trans, param3)
    print('res3', res3)




    posz3d = float(1) / 1000
    posx3d = float(0) / 1000
    posy3d = float(0) / 1000
    posx = posx3d / posz3d
    posy = posy3d / posz3d
    posz = posz3d / posz3d

    posx3d_veh_d = float(1504) / 1000
    posy3d_veh_d = float(-395) / 1000
    posz3d_veh_d = float(874) / 1000

    temp = np.array([posx3d_veh_d - cam2veh_trans[0],
                     posy3d_veh_d - cam2veh_trans[1],
                     posz3d_veh_d - cam2veh_trans[2]])
    result = np.matmul(np.linalg.inv(compute_rotation_matrix(cam2veh_rot)), temp)
    posx3d_veh = result[1]
    posy3d_veh = result[2]
    posz3d_veh = result[0]
    posx_veh = posx3d_veh / posz3d_veh
    posy_veh = posy3d_veh / posz3d_veh
    posz_veh = posz3d_veh / posz3d_veh


    roll_1 = float(0.4) * deg2Rad
    pitch_1 = float(-21.1) * deg2Rad
    yaw_1 = float(-5.9) * deg2Rad

    # pitch_1 = pitch_1 / 180 * np.pi
    # yaw_1 = yaw_1 / 180 * np.pi
    # roll_1 = roll_1 / 180 * np.pi

    result = np.matmul(np.linalg.inv(compute_rotation_matrix(cam2veh_rot)),
                       (compute_rotation_matrix([roll_1, pitch_1, yaw_1])))
    result2 = rotationMatrixToEulerAngles(result)

    roll_1 = result2[0]
    pitch_1 = result2[1]
    yaw_1 = result2[2]

    pitch_1 = -pitch_1
    yaw_1 = -yaw_1

    posx_veh, posy_veh = distort(posx_veh, posy_veh)
    c = np.matmul(intrinsic_matrix, np.array([[posx_veh], [posy_veh], [posz_veh]]))

    headx = c[0][0]
    heady = c[1][0]


# "f_version","f_frame_counter_left_camera","f_frame_counter_right_camera","f_frame_counter_virtual","f_master_counter_in","f_eye_model_generation","eye_left_partial_blockage","eye_left_blockage","MS_EyeLeftCameraLeft_old","MS_EyeLeftCameraRight_old","eye_right_partial_blockage","eye_right_blockage","MS_EyeRightCameraLeft_old","MS_EyeRightCameraRight_old","mouth_nose_partial_blockage","mouth_nose_blockage","MS_MouthNoseCameraLeft","MS_MouthNoseCameraRight","CAN_S_glasses_detected_old","CAN_S_face_detection","MS_EyeDetection","MS_S_Head_rot_X","MS_S_Head_rot_Y","MS_S_Head_rot_Z","HSVL_MS_CAN_S_Head_tracking_status","HSVL_MS_CAN_S_Head_tracking_mode","f_primary_face_landmark_camera","f_left_position_valid0","f_left_positionX0","f_left_positionY0","f_left_position_valid1","f_left_positionX1","f_left_positionY1","f_left_position_valid2","f_left_positionX2","f_left_positionY2","f_left_position_valid3","f_left_positionX3","f_left_positionY3","f_left_position_valid4","f_left_positionX4","f_left_positionY4","f_left_position_valid5","f_left_positionX5","f_left_positionY5","f_left_position_valid6","f_left_positionX6","f_left_positionY6","f_right_position_valid0","f_right_positionX0","f_right_positionY0","f_right_position_valid1","f_right_positionX1","f_right_positionY1","f_right_position_valid2","f_right_positionX2","f_right_positionY2","f_right_position_valid3","f_right_positionX3","f_right_positionY3","f_right_position_valid4","f_right_positionX4","f_right_positionY4","f_right_position_valid5","f_right_positionX5","f_right_positionY5","f_right_position_valid6","f_right_positionX6","f_right_positionY6","CAN_glint_detected","f_gaze_le_result_valid","MS_S_Gaze_LE_VA_rot_X","MS_S_Gaze_LE_VA_rot_Y","MS_S_Gaze_LE_VA_rot_Z","MS_S_Gaze_LE_Center_X","MS_S_Gaze_LE_Center_Y","MS_S_Gaze_LE_Center_Z","f_gaze_re_result_valid","MS_S_Gaze_RE_VA_rot_X","MS_S_Gaze_RE_VA_rot_Y","MS_S_Gaze_RE_VA_rot_Z","MS_S_Gaze_RE_Center_X","MS_S_Gaze_RE_Center_Y","MS_S_Gaze_RE_Center_Z","CAN_eye_closure_left","CAN_eye_closure_left_conf","CAN_eye_closure_right","CAN_eye_closure_right_conf","CAN_eye_closure","MS_eye_closed","CAN_long_eyeclosure","f_long_eyeclosure_counter","MS_eye_closed_AAS","MS_PERCLOS_AAS","MS_PERCLOS_AAS_conf","MS_PERCLOS_strict","MS_eye_closed_strict","CAN_eye_blink_conf","MS_Eye_blink_freq_conf","MS_Eye_blink_freq","CAN_eye_blink_t_closing","CAN_eye_blink_t_opening","CAN_eye_blink_duration","CAN_eye_blink_A_closing","CAN_eye_blink_A_opening","CAN_eye_blink_counter","CAN_S_Gaze_ROI","CAN_S_Gaze_ROI_X","CAN_S_Gaze_ROI_Y","f_roi_id","MS_S_Gaze_ROI_X_Raw","MS_S_Gaze_ROI_Y_Raw","HSVL_MS_S_Head_Pos_Veh_X","HSVL_MS_S_Head_Pos_Veh_Y","HSVL_MS_S_Head_Pos_Veh_Z","MS_S_Gaze_rot_X","MS_S_Gaze_rot_Y","MS_S_Gaze_rot_Z","HSVL_MS_CAN_S_Eye_dist","f_camera_left_measured_brightness","f_camera_left_target_brightness","f_camera_left_shutter_us","f_camera_left_column_gain","f_camera_left_digital_gain","f_camera_right_measured_brightness","f_camera_right_target_brightness","f_camera_right_shutter_us","f_camera_right_column_gain","f_camera_right_digital_gain","f_raw_result_age","HSVL_S_Head_dir_h","HSVL_S_Head_dir_v","HSVL_MS_S_Head_Pos_Disp_X","HSVL_MS_S_Head_Pos_Disp_Y","HSVL_MS_S_Head_Pos_Disp_Z","LGE_BD_frame_count","CAN_S_camera_close_blocked","CAN_S_glasses_detected","MS_camera_blockage_detection","MS_EyeLeftCameraLeft","MS_EyeLeftCameraRight","MS_EyeRightCameraLeft","MS_EyeRightCameraRight","MouthNoseCameraLeft","MouthNoseCameraRight","disp_left_cam_blocked_0","disp_left_cam_blocked_1","disp_left_cam_blocked_2","disp_left_cam_blocked_3","disp_right_cam_blocked_0","disp_right_cam_blocked_1","disp_right_cam_blocked_2","disp_right_cam_blocked_3","LGE_OOF_frame_count","bOutOfFocus","CAN_S_drcam_status","LGE_SWBA_frame_count","CAN_S_StWhl_adjust_occlusion","MS_HeadOcclusion","Absorber_Left_Center_nX","Absorber_Left_Center_nY","Absorber_Right_Center_nX","Absorber_Right_Center_nY","nAbsorber_Radius","Wheel_Left_Center_nX","Wheel_Left_Center_nY","Wheel_Right_Center_nX","Wheel_Right_Center_nY","nWheel_Radius","LGE_DB_frame_count","bIsHeadMoving","MS_Intended_head_movement","CAN_Driver_is_responsive","MS_nod_head_gesture","MS_shake_head_gesture","bIsHeadGestureResult","LGE_DI_frame_count","CAN_S_Driver_ID_Top_1","CAN_S_Driver_ID_Top_2","CAN_S_Driver_ID_Top_3","CAN_S_Driver_ID_Confidence_1","CAN_S_Driver_ID_Confidence_2","CAN_S_Driver_ID_Confidence_3","arrDrviers_01","arrDrviers_02","arrDrviers_03","arrDrviers_04","arrDrviers_05","arrDrviers_06","arrDrviers_07","arrDrviers_08","arrDrviers_09","arrDrviers_10","arrDrviers_11","arrDrviers_12","arrDrviers_13","status","numOfDriver","LGE_DOT_frame_count","CAN_S_Head_Pos_X","CAN_S_Head_Pos_Y","CAN_S_Head_Pos_Z","CAN_S_Head_Pos_type","bLeftIRLight","LGE_SP_frame_count","MS_spoofing_detected","CurrentStatus","OutConfidenceSpoof","OutConfidenceGenuine","f_raw_result_age","S_Head_dir_h","S_Head_dir_v","S_Head_Pos_Disp_x","S_Head_Pos_Disp_y","S_Head_Pos_Disp_z","f_left_position_valid7","f_left_positionX7","f_left_positionY7","f_right_position_valid7","f_right_positionX7","f_right_positionY7","FrameHistory","FrameDiff","FrameTimeStamp","f_head_pose_confidence","f_early_head_pose_confidence","LGE_DI_Ext_frame_count","CAN_S_EnrollAndDeleteStatus","CAN_S_HasStoredIDs","CAN_S_driverID_MsgCnt","ResultDataType","timeEnrollment","bFilteringFlag","f_le_iris_diameter","f_re_iris_diameter","HetAlgoGlintPosition.MS_S_Gaze_LE_Cornea_Center_X","HetAlgoGlintPosition.MS_S_Gaze_LE_Cornea_Center_Y","HetAlgoGlintPosition.MS_S_Gaze_LE_Cornea_Center_Z","HetAlgoGlintPosition.f_le_glint_position_idx_0_X","HetAlgoGlintPosition.f_le_glint_position_idx_0_Y","HetAlgoGlintPosition.f_le_glint_position_idx_1_X","HetAlgoGlintPosition.f_le_glint_position_idx_1_Y","HetAlgoGlintPosition.MS_S_Gaze_RE_Cornea_Center_X","HetAlgoGlintPosition.MS_S_Gaze_RE_Cornea_Center_Y","HetAlgoGlintPosition.MS_S_Gaze_RE_Cornea_Center_Z","HetAlgoGlintPosition.f_re_glint_position_idx_0_X","HetAlgoGlintPosition.f_re_glint_position_idx_0_Y","HetAlgoGlintPosition.f_re_glint_position_idx_1_X","HetAlgoGlintPosition.f_re_glint_position_idx_1_Y","HetAlgoEyelidPosition.f_le_lc_position_valid0","HetAlgoEyelidPosition.f_le_lc_positionX0","HetAlgoEyelidPosition.f_le_lc_positionY0","HetAlgoEyelidPosition.f_le_lc_position_valid1","HetAlgoEyelidPosition.f_le_lc_positionX1","HetAlgoEyelidPosition.f_le_lc_positionY1","HetAlgoEyelidPosition.f_le_lc_position_valid2","HetAlgoEyelidPosition.f_le_lc_positionX2","HetAlgoEyelidPosition.f_le_lc_positionY2","HetAlgoEyelidPosition.f_le_lc_position_valid3","HetAlgoEyelidPosition.f_le_lc_positionX3","HetAlgoEyelidPosition.f_le_lc_positionY3","HetAlgoEyelidPosition.f_le_lc_position_valid4","HetAlgoEyelidPosition.f_le_lc_positionX4","HetAlgoEyelidPosition.f_le_lc_positionY4","HetAlgoEyelidPosition.f_le_lc_position_valid5","HetAlgoEyelidPosition.f_le_lc_positionX5","HetAlgoEyelidPosition.f_le_lc_positionY5","HetAlgoEyelidPosition.f_le_lc_position_valid6","HetAlgoEyelidPosition.f_le_lc_positionX6","HetAlgoEyelidPosition.f_le_lc_positionY6","HetAlgoEyelidPosition.f_le_rc_position_valid0","HetAlgoEyelidPosition.f_le_rc_positionX0","HetAlgoEyelidPosition.f_le_rc_positionY0","HetAlgoEyelidPosition.f_le_rc_position_valid1","HetAlgoEyelidPosition.f_le_rc_positionX1","HetAlgoEyelidPosition.f_le_rc_positionY1","HetAlgoEyelidPosition.f_le_rc_position_valid2","HetAlgoEyelidPosition.f_le_rc_positionX2","HetAlgoEyelidPosition.f_le_rc_positionY2","HetAlgoEyelidPosition.f_le_rc_position_valid3","HetAlgoEyelidPosition.f_le_rc_positionX3","HetAlgoEyelidPosition.f_le_rc_positionY3","HetAlgoEyelidPosition.f_le_rc_position_valid4","HetAlgoEyelidPosition.f_le_rc_positionX4","HetAlgoEyelidPosition.f_le_rc_positionY4","HetAlgoEyelidPosition.f_le_rc_position_valid5","HetAlgoEyelidPosition.f_le_rc_positionX5","HetAlgoEyelidPosition.f_le_rc_positionY5","HetAlgoEyelidPosition.f_le_rc_position_valid6","HetAlgoEyelidPosition.f_le_rc_positionX6","HetAlgoEyelidPosition.f_le_rc_positionY6","HetAlgoEyelidPosition.f_re_lc_position_valid0","HetAlgoEyelidPosition.f_re_lc_positionX0","HetAlgoEyelidPosition.f_re_lc_positionY0","HetAlgoEyelidPosition.f_re_lc_position_valid1","HetAlgoEyelidPosition.f_re_lc_positionX1","HetAlgoEyelidPosition.f_re_lc_positionY1","HetAlgoEyelidPosition.f_re_lc_position_valid2","HetAlgoEyelidPosition.f_re_lc_positionX2","HetAlgoEyelidPosition.f_re_lc_positionY2","HetAlgoEyelidPosition.f_re_lc_position_valid3","HetAlgoEyelidPosition.f_re_lc_positionX3","HetAlgoEyelidPosition.f_re_lc_positionY3","HetAlgoEyelidPosition.f_re_lc_position_valid4","HetAlgoEyelidPosition.f_re_lc_positionX4","HetAlgoEyelidPosition.f_re_lc_positionY4","HetAlgoEyelidPosition.f_re_lc_position_valid5","HetAlgoEyelidPosition.f_re_lc_positionX5","HetAlgoEyelidPosition.f_re_lc_positionY5","HetAlgoEyelidPosition.f_re_lc_position_valid6","HetAlgoEyelidPosition.f_re_lc_positionX6","HetAlgoEyelidPosition.f_re_lc_positionY6","HetAlgoEyelidPosition.f_re_rc_position_valid0","HetAlgoEyelidPosition.f_re_rc_positionX0","HetAlgoEyelidPosition.f_re_rc_positionY0","HetAlgoEyelidPosition.f_re_rc_position_valid1","HetAlgoEyelidPosition.f_re_rc_positionX1","HetAlgoEyelidPosition.f_re_rc_positionY1","HetAlgoEyelidPosition.f_re_rc_position_valid2","HetAlgoEyelidPosition.f_re_rc_positionX2","HetAlgoEyelidPosition.f_re_rc_positionY2","HetAlgoEyelidPosition.f_re_rc_position_valid3","HetAlgoEyelidPosition.f_re_rc_positionX3","HetAlgoEyelidPosition.f_re_rc_positionY3","HetAlgoEyelidPosition.f_re_rc_position_valid4","HetAlgoEyelidPosition.f_re_rc_positionX4","HetAlgoEyelidPosition.f_re_rc_positionY4","HetAlgoEyelidPosition.f_re_rc_position_valid5","HetAlgoEyelidPosition.f_re_rc_positionX5","HetAlgoEyelidPosition.f_re_rc_positionY5","HetAlgoEyelidPosition.f_re_rc_position_valid6","HetAlgoEyelidPosition.f_re_rc_positionX6","HetAlgoEyelidPosition.f_re_rc_positionY6","Het2DLandmark.f_left_position_valid0","Het2DLandmark.f_left_positionX0","Het2DLandmark.f_left_positionY0","Het2DLandmark.f_left_position_valid1","Het2DLandmark.f_left_positionX1","Het2DLandmark.f_left_positionY1","Het2DLandmark.f_left_position_valid2","Het2DLandmark.f_left_positionX2","Het2DLandmark.f_left_positionY2","Het2DLandmark.f_left_position_valid3","Het2DLandmark.f_left_positionX3","Het2DLandmark.f_left_positionY3","Het2DLandmark.f_left_position_valid4","Het2DLandmark.f_left_positionX4","Het2DLandmark.f_left_positionY4","Het2DLandmark.f_left_position_valid5","Het2DLandmark.f_left_positionX5","Het2DLandmark.f_left_positionY5","Het2DLandmark.f_left_position_valid6","Het2DLandmark.f_left_positionX6","Het2DLandmark.f_left_positionY6","Het2DLandmark.f_left_position_valid7","Het2DLandmark.f_left_positionX7","Het2DLandmark.f_left_positionY7","Het2DLandmark.f_right_position_valid0","Het2DLandmark.f_right_positionX0","Het2DLandmark.f_right_positionY0","Het2DLandmark.f_right_position_valid1","Het2DLandmark.f_right_positionX1","Het2DLandmark.f_right_positionY1","Het2DLandmark.f_right_position_valid2","Het2DLandmark.f_right_positionX2","Het2DLandmark.f_right_positionY2","Het2DLandmark.f_right_position_valid3","Het2DLandmark.f_right_positionX3","Het2DLandmark.f_right_positionY3","Het2DLandmark.f_right_position_valid4","Het2DLandmark.f_right_positionX4","Het2DLandmark.f_right_positionY4","Het2DLandmark.f_right_position_valid5","Het2DLandmark.f_right_positionX5","Het2DLandmark.f_right_positionY5","Het2DLandmark.f_right_position_valid6","Het2DLandmark.f_right_positionX6","Het2DLandmark.f_right_positionY6","Het2DLandmark.f_right_position_valid7","Het2DLandmark.f_right_positionX7","Het2DLandmark.f_right_positionY7","Het2DLandmark.f_stereo_face_plane_nose_residual_mm","f_headpose_tracking_mode","S_Head_tracking_status_from_early","Number_of_eye_blinks_haf","timestamp_result_receive","eyeclose_height_left","eyeclose_height_right"

    inputPath_HET = "D:/Project/CVT/demo/1_DRCAM_KOR40BU4578_20190219_114431_0002/HetData_1614867983883741729.json"
    # inputPath_HET = "D:/Project/CVT/demo/result101/HetData_01.json"
    # inputPath_HET = "D:/Project/CVT/성능비교/HetData_test_small.json"

    inputPath_C2V = "D:/Project/CVT/demo/Standalone_Player/1_DRCAM_KOR40BU4578_20190219_114431/pose_config.json"
    # inputPath_C2V = "D:/Project/CVT/성능비교/DRCAM_KOR40BU4578_20190214_113824_0021_2/Result0001_20210205/pose_config.json"
    # pose_config.json

    inputPath_D2V = "D:/Project/CVT/demo/Standalone_Player/1_DRCAM_KOR40BU4578_20190219_114431/display_config.json"
    # inputPath_D2V = "D:/Project/CVT/성능비교/DRCAM_KOR40BU4578_20190214_113824_0021_2/Result0001_20210205/display_config.json"
    # display_config.json

    inputPath_D2C = ""
    # "D:/Project/CVT/성능비교/DRCAM_KOR40BU4578_20190214_113824_0021_2/Result0001_20210205/CamToDisplay_config.json"
    # CamToDisplay_config.json

    inputPath_ROI = "D:/Project/CVT/demo/Standalone_Player/1_DRCAM_KOR40BU4578_20190219_114431/roi_config.json"
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
    # tout = modify_HeadRotation_from_craft_to_daimler(ret_ExtData, tout)
    # tout = modify_HeadDirection_craft_to_daimler(ret_ExtData, tout)

    tout = modify_EyeClosureHeight_craft_to_daimler(ret_ExtData, tout)
    tout = modify_frameCounterVirtual_craft_to_daimler(ret_ExtData, tout)
    tout = modify_HeadDetect_craft_to_daimler(ret_ExtData, tout)

    # print(1/0)

    tout = modify_EyeGaze_craft_to_daimler(ret_ExtData, tout)
    save_csvfile(tout, "./final_output.csv")

    print(1 / 0)

    nR_unitvec = np.zeros((3, 1))
    nR_deg = np.zeros((3, 1))

    # nR_unitvec[0] = 0.093045800924301147
    # nR_unitvec[1] = -0.26501166820526123
    # nR_unitvec[2] = -0.95974433422088623
    nR_unitvec[0] = 0.048387713730335236
    nR_unitvec[1] = -0.30887135863304138
    nR_unitvec[2] = -0.94987112283706665

    nR_rad = changeRotation_unitvec2radian(nR_unitvec)
    print("degree", nR_rad*rad2Deg)

    ret = changeRotation_radian2unitvec(nR_rad)

    nR_deg[0] = -0.11313585 *rad2Deg
    nR_deg[1] = -0.04440416 *rad2Deg
    nR_deg[2] = 0.02910747 *rad2Deg

    print('\n\nnR_deg', nR_deg)
    ret = changeRotation_radian2unitvec(nR_deg*deg2Rad)
    ret2 = changeRotation_unitvec2radian(ret)


    nr1 = np.array([0.8017 , -0.2086, 0.5602])
    nr2 = np.array([0.0067 ,  0.9439, 0.3392])
    nr3 = np.array([-0.1937,  0.2726, 1.0756])

    nr1_2_cross = np.array([-0.5995299,  -0.2681833,   0.75812225])

    changeRotation_unitvec2radian_check(nr1_2_cross)
    # changeRotation_radian2unitvec_check

    print(np.cross(nr1, nr2))
    alpha_yaw = math.atan(nr1_2_cross[0] / nr1_2_cross[2])
    beta_tilt = math.acos(nr1_2_cross[1])
    print("test_tilt", math.acos(nr1_2_cross[1]),math.asin(nr1_2_cross[2]))
    print('yaw',math.atan(nr1_2_cross[0] / nr1_2_cross[2]), alpha_yaw*rad2Deg)
    print('tilt',math.acos(nr1_2_cross[1]), beta_tilt*rad2Deg)
    print('r3', [math.sin(alpha_yaw)*math.sin(beta_tilt), math.cos(beta_tilt), math.cos(alpha_yaw)*math.sin(beta_tilt) ])
    # print("degree", ret2 * rad2Deg)
    print('\n\n')
    alpha_yaw = math.atan(nr1_2_cross[0] / nr1_2_cross[2])
    beta_tilt = math.asin(nr1_2_cross[1])
    print('yaw',math.atan(nr1_2_cross[0] / nr1_2_cross[2]), alpha_yaw*rad2Deg)
    print('tilt',math.asin(nr1_2_cross[1]), beta_tilt*rad2Deg)
    print('r3', [math.sin(alpha_yaw)*math.cos(beta_tilt), math.sin(beta_tilt), math.cos(alpha_yaw)*math.cos(beta_tilt) ])



    # nr1 = np.array([0.4430 , -0.1153, 0.3096])
    # nr2 = np.array([0.0037 ,  0.5216, 0.18753392])
    # nr3 = np.array([-0.1071,  0.1506, 0.5944])
    # print(nr1/np.linalg.norm(nr1))
    # print(nr2/np.linalg.norm(nr1))
    # print(nr3/np.linalg.norm(nr1))

    #Pupil point (L+R)
        # "x": 763.2879638671875,
        # "y": 258.6845703125
        # "x": 937.3201904296875,
        # "y": 265.359130859375

    # "leftGazeDegree":
    #     "x": -2.0081093311309814,
    #     "y": -17.709419250488281
    # "leftGazeRadian":
    #     "x": -0.035048119723796844,
    #     "y": -0.30908766388893127
    # "leftGazeStart3D":
    #     "x": 0.10329416394233704,
    #     "y": -0.073243081569671631,
    #     "z": 0.50226587057113647
    # "leftGazeVector":
    #     "x": -0.035040944814682007,
    #     "y": -0.30400258302688599,
    #     "z": -0.95202547311782837
    # "rightGazeDegree":
    #     "x": 7.3035531044006348,
    #     "y": -18.343164443969727
    # "rightGazeRadian":
    #     "x": 0.12747105956077576,
    #     "y": -0.32014861702919006
    # "rightGazeStart3D":
    #     "x": 0.043515890836715698,
    #     "y": -0.076108850538730621,
    #     "z": 0.50938701629638672
    # "rightGazeVector":
    #     "x": 0.12712612748146057,
    #     "y": -0.31215381622314453,
    #     "z": -0.94148653745651245

    # "frameId": 45291,
    # "fusedGazeDegree":
    #     "x": 2.7734947204589844,
    #     "y": -18.0130615234375
    # "fusedGazeRadian":
    #     "x": 0.048406615853309631,
    #     "y": -0.31438723206520081
    # "fusedGazeStart3D":
    #     "x": 0.072999998927116394,
    #     "y": -0.075000002980232239,
    #     "z": 0.50599998235702515
    # "fusedGazeVector":
    #     "x": 0.048387713730335236,
    #     "y": -0.30887135863304138,
    #     "z": -0.94987112283706665
    # "headOri":
    #     "x": -0.11313585937023163,
    #     "y": -0.044404163956642151,
    #     "z": 0.029107470065355301
    # "headPos3D":
    #     "x": 0.072999998927116394,
    #     "y": -0.075000002980232239,
    #     "z": 0.50599998235702515
    headPos3D_meter = np.array([0.072999,-0.075000,0.505999])
    headOri_radian = np.array([-0.113135, -0.0444, 0.02910])
    fusedGazeVector = np.array([0.048387, -0.3088, -0.94987])

    pupil_l = np.array([763.287963, 258.684570])
    pupil_r = np.array([937.320190, 265.359130])

    # "x": 763.2879638671875,
    # "y": 258.6845703125
    # "x": 937.3201904296875,
    # "y": 265.359130859375


    # origin = [tvec[0][0], tvec[1][0], tvec[2][0]]
    # headDir = np.dot(rot2, np.dot(rt, [0, 0, 1]))
    # camPlaneOrthVector = [0, 0, 1]
    # pointOnPlan = [0, 0, 0]

    print("\n\n check//////////", headOri_radian*rad2Deg)


    origin = (headPos3D_meter )
    pitch_yaw_roll = changeRotation_unitvec2radian_check(fusedGazeVector)
    rt = eulerAnglesToRotationMatrix(headOri_radian)
    # rot2 = np.eye(3)
    rot2 = eulerAnglesToRotationMatrix(pitch_yaw_roll)
    headDir = np.dot(rot2, np.dot(rt, [0, 0, 1]))
    # headDir = np.dot(rt, [0, 0, 1])
    camPlaneOrthVector = [0, 0, 1]
    pointOnPlan = [0, 0, 0]

    print('origin', origin, '\nheadDir', headDir)
    print('\ngaze_pitch_yaw_roll_deg', pitch_yaw_roll*rad2Deg)
    print('\ngaze_pitch_yaw_roll',pitch_yaw_roll, '\nrt',rt, '\nrot2', rot2)
    tview_point = intersectionWithPlan(origin, headDir, camPlaneOrthVector, pointOnPlan)
    print('tview_point', tview_point)

    intrinsic_matrix3 = np.array([[1479.36, 0., 640.91], [0., 1479.39, 488.959], [0., 0., 1.]])

    # result3 = np.matmul(intrinsic_matrix, np.array([[posx_veh], [posy_veh], [posz_veh]]))
    result3 = np.matmul(intrinsic_matrix3, np.array(tview_point))
    print("K * tview_point ",result3)
    # tview_point[-2.64.586200e+02
    # 1.63.778058e+02
    # 5.68434189e-14]
    # tview_point[-1.25.261755e+02
    # 3.05.435605e+02 - 5.68434189e-14]


    rx = math.cos(headOri_radian[0]) * math.sin(headOri_radian[1]) * 200   #scale 200
    ry = math.sin(headOri_radian[0]) * 200
    calcCx = (headPos3D_meter[0] + (fusedGazeVector[0] * 300)) - rx  #scale 300
    calcCy = (headPos3D_meter[1] + (fusedGazeVector[1] * 300)) + ry  #scale 300
    print("\n\nrx", rx, 'ry', ry)
    print("\ncalcCx", calcCx, "calcCy", calcCy)
    #
    # t_matrix_1 = np.eye(4)
    # s_trot = np.zeros((3, 1))
    # s_trot[0] = deg2Rad * disp2cam_rot[0]
    # s_trot[1] = deg2Rad * disp2cam_rot[1]
    # s_trot[2] = deg2Rad * disp2cam_rot[2]
    # t_matrix_1[0:3, 0:3] = eulerAnglesToRotationMatrix(s_trot)
    # t_matrix_1[0:3, 3] = disp2cam_trans

    # size = 60.0
    # zAxis[] = {size * static_cast < float_t > (gaze3D.x),
    #            size * static_cast < float_t > (gaze3D.y),
    #            size * static_cast < float_t > (-gaze3D.z)};
    # translation[] = {static_cast < float_t > (startPosition.x),
    #                  static_cast < float_t > (startPosition.y), 0.0F};
    #
    # zVect = cv::Mat(3, 1, CV_32F, zAxis);
    # translationVect = cv::Mat(3, 1, CV_32F, translation);
    #
    # zVect = zVect + translationVect;
    # endPosition(
    #     static_cast < int32_t > (zVect.at < float_t > (0, 0)),
    #     static_cast < int32_t > (zVect.at < float_t > (1, 0)));
    print("\n\n")
    print('lpupil', pupil_l)
    zAxis = np.array([fusedGazeVector[0], fusedGazeVector[1], fusedGazeVector[2]])
    print('zAxis', zAxis)
    translationVect = zAxis  + np.array([763.2879638671875, 258.6845703125, 0])
    print("translationVect", translationVect, '\n' ,(translationVect - np.array([763.2879638671875, 258.6845703125, 0])))
    translationVect = zAxis* 60 + np.array([ 763.2879638671875, 258.6845703125, 0])
    print("translationVect", translationVect)
    translationVect = zAxis* 120 + np.array([ 763.2879638671875, 258.6845703125, 0])
    print("translationVect", translationVect)
    translationVect = zAxis * 180 + np.array([763.2879638671875, 258.6845703125, 0])
    print("translationVect", translationVect)
    translationVect = zAxis * headPos3D_meter[2]*1000 + np.array([763.2879638671875, 258.6845703125, 0])
    print("translationVect", translationVect)
    translationVect /= translationVect[2]
    print("translationVect", translationVect)
    print("\n\nend")

    ret_json_roi = load_jsonfile_ROI(inputPath_ROI)
    extract_availData_from_mra2_ROI(ret_json_roi)
    # count frame 370
    # roi ID, roi x, roi y
    # 6	20	15
    # "_comment": "Display HeadUnit",
    # "id": 6,
    # "head_only_id": 6,
    # "priority": 3,
    # "obj_params":
    # {
    #     "top_left": [1.04338, -0.143, 0.636058],
    #     "top_right": [1.04338, 0.137, 0.636058],
    #     "bottom_left": [1.207, -0.143, 0.434]
    #rot x, y, z 7.3	-29  	-34.1
    #veh 1495	-346	836
    #lpupil 1472	-369	835	/   0	-26.6	-27.8
    #rpupil 1508	-316	843	/   0	-26.6	-27.8

    # rr, tt = changeAxis_daimler2opencv(np.array([7.3,-29,-34.1])*deg2Rad, np.array([1495,-346,836]))

    headPos3D_meter = np.array([1495,-346,836])
    headOri_radian = np.array([7.3,-29,-34.1])*deg2Rad

    beta_tilt = -26.6 * deg2Rad
    alpha_yaw = -27.8 * deg2Rad
    fusedGazeVector = [math.sin(alpha_yaw) * math.cos(beta_tilt), math.sin(beta_tilt), math.cos(alpha_yaw) * math.cos(beta_tilt)]
    print('fusedGazeVector', fusedGazeVector)

    # alpha_yaw = math.atan(nr1_2_cross[0] / nr1_2_cross[2])
    # beta_tilt = math.asin(nr1_2_cross[1])
    # print('yaw',math.atan(nr1_2_cross[0] / nr1_2_cross[2]), alpha_yaw*rad2Deg)
    # print('tilt',math.asin(nr1_2_cross[1]), beta_tilt*rad2Deg)
    # print('r3', )

    # pupil_l = np.array([763.287963, 258.684570])
    # pupil_r = np.array([937.320190, 265.359130])

    origin = (headPos3D_meter)
    pitch_yaw_roll = changeRotation_unitvec2radian_check(fusedGazeVector)
    roll_pitch_yaw = np.array([pitch_yaw_roll[2], pitch_yaw_roll[0], pitch_yaw_roll[1]])
    print('pitch_yaw_roll',pitch_yaw_roll*rad2Deg)
    print('roll_pitch_yaw', roll_pitch_yaw * rad2Deg)

    rt = eulerAnglesToRotationMatrix(headOri_radian)
    # rot2 = np.eye(3)
    rot2 = eulerAnglesToRotationMatrix(roll_pitch_yaw)
    headDir = np.dot(rot2, np.dot(rt, [1, 0, 0]))
    # headDir = np.dot(rt, [0, 0, 1])
    camPlaneOrthVector = [0, 0, 1]
    pointOnPlan = [0, 0, 0]

    tview_point = intersectionWithPlan(origin, headDir, camPlaneOrthVector, pointOnPlan)
    print('tview_point', tview_point)

    zAxis = np.array([fusedGazeVector[0], fusedGazeVector[1], fusedGazeVector[2]])
    print('zAxis', zAxis)

    # p0 = top_left = np.array([1043.38, -143, 636.058])
    # p1 = top_right = np.array([1043.38, 137, 636.058])
    # p2 = bottom_left = np.array([1207, -143, 434])
    # bottom_right = np.array([1207, -143, 434]) +  top_right - top_left
    #
    # print("bottom_right",bottom_right)
    # lpupil_3d = np.array([1472, -369, 835])
    # rpupil_3d = np.array([1508,	-316, 843])
    #
    # print('test', (top_left - lpupil_3d)/zAxis)
    # translationVect = zAxis*251 +   (top_left - lpupil_3d)
    # print('translationVect', translationVect)
    # translationVect2 = zAxis*-251 +   lpupil_3d
    # print('translationVect2', translationVect2)
    #
    # print("1", top_left - lpupil_3d)
    # print("2", top_right - lpupil_3d)
    # print("3", bottom_left - lpupil_3d)
    # print("4", bottom_right - lpupil_3d)
    # print("gaze_scale", zAxis * 1000)
    #
    # n =  np.cross((p2 - p0) , (p1 - p0))
    #
    # n2 =  (p2 - p0)
    # n3 =  (p1 - p2)
    # test = n2 * n3 / np.linalg.norm(n2 * n3)
    #
    # # np.linalg.norm
    # print('n', n)
    # print('n2', n2, n3)
    # print('test',test)
    # print(np.cross((1,0,1), (-2,2,1)), np.dot((1,0,1), (-2,2,1)))
    #
    # pOrigin = np.array([1472, -369, 835])
    # vDir = np.array([-0.41702159046519227, -0.4477590878387697, 0.7909518268160673])
    # v_result = plane_p0 - line_origin
    # f_result = np.dot(v_result, plane_n)
    # f_t = f_result / np.dot(line_vDir, plane_n)
    # f_s = line_origin + line_vDir_scale(f_t)


    print("\n\n\n test/////////////////////")
    # 1번 샘플 - count frame 45282
    # roi ID, roi x, roi y
    # 3     10    80
    # "_comment": "모름",
    # "id": 3,
    # "obj_params":
    # {
    #     "top_left": [1.316, -0.127, 0.985],
    #     "top_right": [1.316, 0.127, 0.985],
    #     "bottom_left": [1.316, -0.127, 0.905]
    #Headrot_veh x, y, z   1.5	-9.3	1.5
    #Headpos_veh 1501	-422	857

    #lpupil 1501	-453	859	/   0	-9.8	0.2
    #rpupil 1498	-392	861	/   0	-9.8	0.2

    p0 = top_left = np.array([1316, -127, 985])
    p1 = top_right = np.array([1316, 127, 985])
    p2 = bottom_left = np.array([1316, -127, 905])
    p3 = bottom_right = bottom_left +  top_right - top_left

    print("bottom_right",bottom_right)
    lpupil_3d = np.array([1501,	-453,	859])
    rpupil_3d = np.array([1498,	-392,	861])

    print('test', (top_left - lpupil_3d)/zAxis)
    translationVect = zAxis*251 +   (top_left - lpupil_3d)
    print('translationVect', translationVect)
    translationVect2 = zAxis*-251 +   lpupil_3d
    print('translationVect2', translationVect2)

    print("1", top_left - lpupil_3d)
    print("2", top_right - lpupil_3d)
    print("3", bottom_left - lpupil_3d)
    print("4", bottom_right - lpupil_3d)
    print("gaze_scale", zAxis * 1000)

    n =  np.cross((p2 - p0) , (p1 - p0))
    n2 =  np.cross((p2 - p1) , (p1 - p0))
    plane_n = np.cross((p2 - p1), (p3 - p0))

    n2 =  (p2 - p0)
    n3 =  (p1 - p2)
    test = n2 * n3 / np.linalg.norm(n2 * n3)

    # np.linalg.norm
    print('n', n)
    print('n2', n2, n3)
    print('plain',plane_n)
    print('test',test)
    print(np.cross((1,0,1), (-2,2,1)), np.dot((1,0,1), (-2,2,1)))

    pOrigin = np.array([1501,	-453,	859])
    vDir = np.array([-0.41702159046519227, -0.4477590878387697, 0.7909518268160673])

    pp =  pOrigin + 0.5* vDir

    # np.dot(plain_n, pp-

    # v_result = plane_p0 - line_origin
    # f_result = np.dot(v_result, plane_n)
    # f_t = f_result / np.dot(line_vDir, plane_n)
    # f_s = line_origin + line_vDir_scale(f_t)

