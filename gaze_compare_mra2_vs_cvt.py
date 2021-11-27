import math
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

'''
compare CVT roi position VS MRA2 roi position

CVT algo roi position 
4 roi [  0.47032739 -29.18582592  -2.52451618](0    -19.04887314    -2.331581878)
    "fusedGazeDegree" : 
        "x" : 5.4291658401489258,
        "y" : -27.637931823730469
    "fusedGazeRadian" : 
        "x" : 0.094756826758384705,
        "y" : -0.48237290978431702
    "fusedGazeStart3D" : 
        "x" : 0.0030000000260770321,
        "y" : -0.016000000759959221,
        "z" : 0.47600001096725464
    "fusedGazeVector" : 
        "x" : 0.094615086913108826,
        "y" : -0.46180105209350586,
        "z" : -0.88192164897918701
    
1 roi 0.59, -15.9, -3.33 (0    -5.779668416    -3.22318371)
    "fusedGazeDegree" : 
        "x" : 0.77771860361099243,
        "y" : -22.511287689208984
    "fusedGazeRadian" : 
        "x" : 0.013573751784861088,
        "y" : -0.39289608597755432
    "fusedGazeStart3D" : 
        "x" : 0.018999999389052391,
        "y" : -0.026000000536441803,
        "z" : 0.63700002431869507
    "fusedGazeVector" : 
        "x" : 0.01357333455234766,
        "y" : -0.38282975554466248,
        "z" : -0.9237181544303894
12 roi [-0.33248421 -6.52532404  1.88400322](0    3.61524451    1.875527425)
    "fusedGazeDegree" : 
        "x" : 2.9285943508148193,
        "y" : -15.259483337402344
    "fusedGazeRadian" : 
        "x" : 0.051113616675138474,
        "y" : -0.26632824540138245
    "fusedGazeStart3D" : 
        "x" : 0.013000000268220901,
        "y" : -0.010999999940395355,
        "z" : 0.41899999976158142
    "fusedGazeVector" : 
        "x" : 0.051091361790895462,
        "y" : -0.26284691691398621,
        "z" : -0.96348285675048828

7 roi [ -6.62270381 -15.70251973  40.42873963](0    -7.892898984    39.07008191)
    "fusedGazeDegree" : 
        "x" : 42.898780822753906,
        "y" : 0.68388813734054565
    "fusedGazeRadian" : 
        "x" : 0.74872493743896484,
        "y" : 0.011936100199818611
    "fusedGazeStart3D" : 
        "x" : 0.075999997556209564,
        "y" : -0.018999999389052391,
        "z" : 0.67000001668930054
    "fusedGazeVector" : 
        "x" : 0.68070524930953979,
        "y" : 0.00874367356300354,
        "z" : -0.73250389099121094
    
8 roi [  8.85865979 -20.87692497 -57.5193252 ](0    -15.21255113    -54.76695843)
    "fusedGazeDegree" : 
        "x" : -50.446460723876953,
        "y" : -22.368207931518555
    "fusedGazeRadian" : 
        "x" : -0.88045686483383179,
        "y" : -0.39039888978004456
    "fusedGazeStart3D" : 
        "x" : -0.054000001400709152,
        "y" : -0.0080000003799796104,
        "z" : 0.66699999570846558
    "fusedGazeVector" : 
        "x" : -0.77102988958358765,
        "y" : -0.24233794212341309,
        "z" : -0.58888304233551025

SM algo roi position

4 roi
MS_S_Gaze_RE_VA_rot_X    MS_S_Gaze_RE_VA_rot_Y    MS_S_Gaze_RE_VA_rot_Z
0    -1.1    -0.9

1 roi   
MS_S_Gaze_RE_VA_rot_X    MS_S_Gaze_RE_VA_rot_Y    MS_S_Gaze_RE_VA_rot_Z
0    -14.1    1.4

12 roi
MS_S_Gaze_RE_VA_rot_X    MS_S_Gaze_RE_VA_rot_Y    MS_S_Gaze_RE_VA_rot_Z
0    -30.7    6.1

7 roi
MS_S_Gaze_RE_VA_rot_X    MS_S_Gaze_RE_VA_rot_Y    MS_S_Gaze_RE_VA_rot_Z
0    -10.8    43.1

8 roi
MS_S_Gaze_RE_VA_rot_X    MS_S_Gaze_RE_VA_rot_Y    MS_S_Gaze_RE_VA_rot_Z
0    -4.3    -55.6
'''


if __name__ == '__main__':
    rt_2 = np.dot(eulerAnglesToRotationMatrix(np.array([0, math.pi, 0])), np.array([1, 1, 1])).round(5)
    rt = eulerAnglesToRotationMatrix(np.array([0,0,0]) * rt_2)
    print('rt_2', rt_2)
    print('rt', rt)
    #rt = eulerAnglesToRotationMatrix(np.array([1, 1, 1]) ) #* rt_2
    #print('rt', rt)
    #print('euler_deg',     rotationMatrixToEulerAngles(rt)*rad2Deg)
    # tvec2ang = changeRotation_unitvec2radian('PYR', np.array(np.array([-1/math.sqrt(3),-1/math.sqrt(3),-1/math.sqrt(3)])), 'PYR')
    # print(tvec2ang * rad2Deg)
    #
    # tvec2ang = changeRotation_unitvec2radian('PYR', np.array(np.array([-0.82998448610305786,-0.089330889284610748,-0.55058485269546509])), 'PYR')
    # print(tvec2ang * rad2Deg)

    tvec2ang = changeRotation_unitvec2radian('PYR', np.array(np.array([0.094615, -0.4618, -0.8819 ])), 'PYR')
    # tvec2ang = changeRotation_unitvec2radian('PYR', np.array(np.array([0.094615, -0.4618, 0.8819 ])), 'PYR')
    # tvec2ang = changeRotation_unitvec2radian('PYR', np.array(np.array([-0.094615, -0.4618, 0.8819])), 'PYR')
    # "fusedGazeDegree" :
    # {
    #     "x" : 5.4291658401489258,
    #     "y" : -27.637931823730469
    # },
    print('\nfront roi-test_rad',tvec2ang)
    print('\nfront roi-test',tvec2ang * rad2Deg)
    # print(1/0)

    rt_2 = np.dot(eulerAnglesToRotationMatrix(np.array([0, math.pi, 0])), tvec2ang).round(5)
    print('rotation yaw_180',rt_2 * rad2Deg)
    aaa= tvec2ang.copy()
    aaa[0] = -aaa[0]
    aaa[1] = aaa[1]
    aaa[2] = -aaa[2]
    print('33matrix_0\n',eulerAnglesToRotationMatrix(tvec2ang))
    print('33matrix_1\n',-eulerAnglesToRotationMatrix(tvec2ang))
    print('33matrix_2\n', eulerAnglesToRotationMatrix(aaa))
    print('33matrix_3\n', -eulerAnglesToRotationMatrix(aaa))
    print('deg_euler ang_0\n',rotationMatrixToEulerAngles(eulerAnglesToRotationMatrix(tvec2ang))*rad2Deg)
    print('deg_euler ang_1\n',rotationMatrixToEulerAngles(-eulerAnglesToRotationMatrix(tvec2ang))*rad2Deg)
    print('deg_euler ang_2\n', rotationMatrixToEulerAngles(eulerAnglesToRotationMatrix(aaa))*rad2Deg)
    print('deg_euler ang_3\n', rotationMatrixToEulerAngles(-eulerAnglesToRotationMatrix(aaa))*rad2Deg)


    # print(1/0)

    tvec2ang = changeRotation_unitvec2radian('PYR', np.array(np.array([0.05109,  -0.26284,  -0.9634])), 'PYR')
    # "fusedGazeDegree" :
    # {
    #     "x" : 2.9285943508148193,
    #     "y" : -15.259483337402344
    # },
    print('\ndown roi-tvang_rad', tvec2ang)
    print('\ndown roi-tvang', tvec2ang * rad2Deg)

    tvec2ang = changeRotation_unitvec2radian('PYR', np.array(np.array([0.68070, 0.00874, -0.7325 ])), 'PYR')
    # "fusedGazeDegree" :
    # {
    #     "x" : 42.898780822753906,
    #     "y" : 0.68388813734054565
    # },
    print('\nleft roi-test_rad',tvec2ang)
    print('\nleft roi-test',tvec2ang * rad2Deg)

    tvec2ang = changeRotation_unitvec2radian('PYR', np.array(np.array([-0.771029,  -0.24233,  -0.5888])), 'PYR')
    # "fusedGazeDegree" :
    # {
    #     "x" : -50.446460723876953,
    #     "y" : -22.368207931518555
    # },
    print('\nright roi-tvang_rad', tvec2ang)
    print('\nright roi-tvang', tvec2ang * rad2Deg)

    print(1/0)

    # aaa = changeRotation_pitchyaw2unitvec('RPY',np.array(np.array([0,-0.16084566712379456,-0.9790799617767334])) ,'RPY')
    aaa = changeRotation_pitchyaw2unitvec('PYR',np.array(np.array([-0.0894541, -0.9850797, 0])) ,'PYR')
    print('\naaa1', aaa)
    aaa = changeRotation_pitchyaw2unitvec('RPY',np.array(np.array([0, -0.33206932, -0.04065555])) ,'RPY')
    print('\naaa2', aaa)

    aaa = changeRotation_pitchyaw2unitvec('RPY',np.array(np.array([0, -0.97907, -0.16084566])) ,'RPY')
    print('\naaa3', aaa)

    # tvec2ang = changeRotation_unitvec2radian('PYR', np.array(np.array([0/math.sqrt(1),0/math.sqrt(1),1/math.sqrt(1)])), 'PYR')
    # print(tvec2ang* rad2Deg)
    print(1/0)


