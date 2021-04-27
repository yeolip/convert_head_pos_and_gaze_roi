

# from __future__ import print_function
import numpy as np
import math
import sys

C_PRINT_ENABLE = 1

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


# https://gamedev.stackexchange.com/questions/172147/convert-3d-direction-vectors-to-yaw-pitch-roll-angles
def changeRotation_unitvec2radian(typeIn, nR_unitvec, typeOut ):
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

def changeRotation_unitvec2radian2(typeIn, nR_unitvec, typeOut ):
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

def distance_xyz(a,b):
    temp = a - b
    dist = np.sqrt(temp[0] * temp[0] + temp[1] * temp[1] + temp[2] * temp[2])
    return dist

if __name__=="__main__":
	print("Utiltity_of_calculation_by_lip...")