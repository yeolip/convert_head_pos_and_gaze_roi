

# from __future__ import print_function
import numpy as np
import math
import sys

# import util_calc as ut
from util_calc import *


C_PRINT_ENABLE = 1

# deg2Rad = math.pi/180
# rad2Deg = 180/math.pi

class match_intersection_roi(object):
	def __del__(self):
		print("*************delete match_intersection_roi class***********\n")

	def __init__(self):
		self.debugflag = C_PRINT_ENABLE
		print("*************initial match_intersection_roi class***********\n")
		pass

	def line_plane_collision(self, planeNormal, planePoint, rayDirection, rayPoint, epsilon=1e-6):
		ndotu = planeNormal.dot(rayDirection)
		# print('ndotu',ndotu)
		if abs(ndotu) < epsilon:
			return np.array([0,0,0])
			# raise RuntimeError("no intersection or line is within plane")

		w = rayPoint - planePoint
		si = -planeNormal.dot(w) / ndotu
		Psi = w + si * rayDirection + planePoint
		return Psi

	def is_same_direction(self, planePoint_from_ray, rayDirection, rayPoint):
		ret = np.dot(rayDirection, planePoint_from_ray - rayPoint)
		print('ret', ret, ret >= 0)
		return ret >= 0

	def is_sameside_on_line(self, a, b, p, q):
		# print('\na',a, 'b',b,'p',p,'q',q)

		c1 = np.cross(b - a, p - a)
		c2 = np.cross(b - a, q - a)
		# print('c1',c1,'c2',c2, 'c1*c2 >= 0',c1 * c2 , np.dot(c1,c2), np.dot(c1,c2) >= 0)
		# print('')
		return np.dot(c1, c2) >= 0

	def normal_vector_from_plane(self, p0, p1, p2):
		return np.cross((p2 - p0), (p1 - p2)) / np.linalg.norm(np.cross((p2 - p0), (p1 - p2)))

	def is_inside_triangle(self, a, b, c, p):
		# print(is_sameside_on_line(a, b, c, p))
		# print(is_sameside_on_line(b, c, a, p))
		# print(is_sameside_on_line(c, a, b, p))
		# print(all([is_sameside_on_line(a, b, c, p),
		# 	 is_sameside_on_line(b, c, a, p),
		# 	 is_sameside_on_line(c, a, b, p)]))

		xs = (a, b, c) * 2
		# print(xs)
		# print(xs[i:i+3] for i in range(3))
		# print(all(is_sameside(*xs[i:i+3], p) for i in range(3)))
		# return 'NULL'
		return all(self.is_sameside_on_line(*xs[i:i+3], p) for i in range(3))

	def is_inside_plane(self, a, b, c, d, p):
		# print(self.is_sameside_on_line(a, b, c, p))
		# print(self.is_sameside_on_line(a, d, b, p))
		# print(self.is_sameside_on_line(c, b, a, p))
		# print(self.is_sameside_on_line(c, d, a, p))

		# xs = (a, b, c, d) * 2
		# print(xs)
		# # print(xs[i:i+3] for i in range(3))
		# # print(all(self.is_sameside_on_line(*xs[i:i+3], p) for i in range(3)))
		# # return 'NULL'
		return all([self.is_sameside_on_line(a, b, c, p),self.is_sameside_on_line(a, d, b, p),
					self.is_sameside_on_line(c, b, a, p),self.is_sameside_on_line(c, d, a, p)])

	def is_same_on_plane(self, p0, p1, p2, pnt, epsilon=3):
		tnormal_basic_plane =  self.normal_vector_from_plane(p0,p1,p2)
		tnormal_test_plane = self.normal_vector_from_plane(p0,p1,pnt)
		# print("  tnormal_basic_plane",tnormal_basic_plane,"\n  tnormal_test_plane",tnormal_test_plane)
		print(" ",np.round(tnormal_basic_plane,epsilon) == np.round(tnormal_test_plane,epsilon))
		print(" ",np.round(np.dot(tnormal_basic_plane, tnormal_test_plane),epsilon))

		# print('N of plane',normal_vector_from_plane(p0,p1,p2))
		# print('N of plane',normal_vector_from_plane(sa,sb,sq6))
		# print("///square result=",is_inside_plane(sa, sb, sc, sd, sq6))
		return all(np.round(tnormal_basic_plane,epsilon) == np.round(tnormal_test_plane,epsilon))

	def check_available_point_on_plane(self, p0, p1, p3, p2, tpnt):
		ret_inside = self.is_inside_plane(p0, p1, p3, p2, tpnt)
		ret_on_plane = self.is_same_on_plane(p0, p1, p3, tpnt)
		print(" ret inside=",ret_inside,", ret_on_plane", ret_on_plane)
		print("final",all([ret_inside, ret_on_plane]))
		return all([ret_inside, ret_on_plane])

	def line_point_min_dist(self, p, a, b):
		# normalized tangent vector
		d = np.divide(b - a, np.linalg.norm(b - a))

		# signed parallel distance components
		s = np.dot(a - p, d)
		t = np.dot(p - b, d)

		# clamped parallel distance
		h = np.maximum.reduce([s, t, 0])

		# perpendicular distance component
		c = np.cross(p - a, d)

		return np.hypot(h, np.linalg.norm(c))

if __name__=="__main__":
	# #Define plane
	# planeNormal = np.array([0, 0, 1])
	# planePoint = np.array([0, 0, 5]) #Any point on the plane
	#
	# #Define ray
	# rayDirection = np.array([0, -1, -1])
	# rayPoint = np.array([0, 0, 10]) #Any point along the ray

	#Define plane
	planeNormal = np.array([-0.51503999,	0,	0.85716615])
	planePoint = np.array([937.15,  -350.,     802.538]) #Any point on the plane
	# planePoint = np.array([1224.3, - 800.,	975.076]) #Any point on the plane

	# at[383.70970791 - 456.45847575
	# 469.99584065]

	#Define ray
	# rayDirection = np.array([0.94451975, 0.02913004, 0.32716034])
	rayDirection = np.array([0.94451975, 0.02913004, 0.32716034])
	rayPoint = np.array([1501,	-422,	857]) #Any point along the ray

	tObj = match_intersection_roi()

	Psi = tObj.line_plane_collision(planeNormal, planePoint, rayDirection, rayPoint)
	print ("intersection at", Psi)


	# print(*map(lambda x: np.array(*x),	[(0, 0),	(10, 10),	(6, 2), 	(2, 2)]))
	# isInside(*map(lambda x: Vector(*x),	[(0, 0),	(10, 10),	(6, 2), 	(2, 2)]))
	print('\n\n')
	a = np.array(( 0, 0, 0))
	b = np.array((10,10, 0))
	c = np.array(( 6, 2, 0))
	q = np.array(( 5, 3, 0))	 # true
	q1 = np.array(( 8, 0, 0)) 	 # false
	q2 = np.array((11,11, 0))    # false
	q3 = np.array(( 1,10, 0))	 # false
	q4 = np.array((-10,10,0))    # false
	q5 = np.array(( 8, 6, 0))    # true

	# print(is_sameside_on_line(a, b, c, q3))
	print("///triangle result=",tObj.is_inside_triangle(a, b, c, q5))

	sa = np.array(( 0, 0, 0))
	sb = np.array((10, 0, 0))
	sc = np.array((10,10, 0))
	sd = np.array(( 0,10, 0))
	sq1 = np.array(( 1, 1, 0)) 	  # true
	sq2 = np.array((11,11, 0))    # false
	sq3 = np.array(( 1,10, 0))	  # ??겹침 true
	sq4 = np.array((-10,10,0))    # false
	sq5 = np.array(( 5,-1, 0))    # false
	sq6 = np.array(( 8, 4, 0))    # true

	print("///square result=",tObj.is_inside_plane(sa, sb, sc, sd, sq6))
	# print(1/0)

	print("\n\ndouble check.......")
	# p0 = top_left = np.array([1316, -127, 985])
	# p1 = top_right = np.array([1316, 127, 985])
	# p2 = bottom_left = np.array([1316, -127, 905])
	p0 = top_left = np.array([1224.3, -800, 975.076])
	p1 = top_right = np.array([1224.3, 100, 975.076])
	p2 = bottom_left = np.array([650, -800, 630])
	p3 = bottom_right = bottom_left + top_right - top_left

	tpnt = np.array([383.70970819, -456.45847574,	469.99584075])	#point of square from vector -Head Eye gaze [0.94451975 0.02913004 0.32716034]
																	# -> user input unitvec2radian [  0.,         -19.09650997,   1.76650551]
	tpnt2 = np.array([937.15, -350., 802.538])		#center of square
	tpnt3 = np.array([1065.46305, - 435.432, 706.13982])	#virtual point(from user)
	tpnt4 = np.array([1221.56363,    43.727, 973.43181])	#virtual point(from user)-Head Eye gaze(0.50306617 -0.83844362 -0.20961)]
															#-> user input unitvec2radian [-0.          12.09949823 - 59.03825349]
	tpnt5 = np.array([1002.18644251,	-594.77776676,  841.61603489])  ##virtual point(from user2 swap(y,z))-Head Eye gaze [0.94451975, 0.32716034, 0.02913004 ]

	print("///square result=", tObj.is_inside_plane(p0, p1, p3, p2, tpnt))

	print(np.dot(planeNormal, planePoint))
	print(np.dot(planeNormal, p0))
	print(np.dot(planeNormal, p1))
	print(np.dot(planeNormal, p2))
	print(np.dot(planeNormal, p3))
	print(np.dot(planeNormal, tpnt))
	print(np.dot(planeNormal, tpnt2))
	print(np.dot(planeNormal, tpnt2))

	print(np.cross((p3 - p1), (p2 - p3)) / np.linalg.norm(np.cross((p3 - p1), (p2 - p3))))
	print(np.cross((p3 - p1), (p2 - p1)) / np.linalg.norm(np.cross((p3 - p1), (p2 - p1))))
	print(np.cross((p3 - p1), (p0 - p3)) / np.linalg.norm(np.cross((p3 - p1), (p0 - p3))))
	print(np.cross((p3 - p1), (tpnt2 - p3)) / np.linalg.norm(np.cross((p3 - p1), (tpnt2 - p3))))

	print(np.cross((p3 - p1), (p2 - p3)) )
	print(np.cross((p3 - p1), (p2 - p1)) )
	print(np.cross((p3 - p1), (p0 - p3)) )
	print(np.cross((p3 - p1), (tpnt2 - p3)) )

	print(np.cross((sb - sa), (sd - sa)) )
	print(np.dot((sb - sa), (sc - sa)))


	print('N of plane',tObj.normal_vector_from_plane(sa,sb,sc))
	print('N of plane',tObj.normal_vector_from_plane(sa,sb,sq6))
	print(tObj.is_same_on_plane(sa, sb, sc, sq6))
	print(tObj.is_same_on_plane(sa, sb, sc, np.array((8, 4, 1))))
	print(tObj.is_same_on_plane(p0, p1, p2, tpnt2))
	print(tObj.is_same_on_plane(p0, p1, p2, tpnt))

	tObj.check_available_point_on_plane(p0, p1, p3, p2, tpnt)
	tObj.check_available_point_on_plane(p0, p1, p3, p2, tpnt2)
	tObj.check_available_point_on_plane(sa, sb, sc, sd, np.array((8, 4, 1)))
	tObj.check_available_point_on_plane(sa, sb, sc, sd, sq6)
	tObj.check_available_point_on_plane(p0, p1, p3, p2, tpnt3)
	tObj.check_available_point_on_plane(p0, p1, p3, p2, tpnt4)
	tObj.check_available_point_on_plane(p0, p1, p3, p2, tpnt5)

	theta = 180 * deg2Rad
	print(theta)
	R_y = np.array([[math.cos(theta), 0, math.sin(theta)],
					[0, 1, 0],
					[-math.sin(theta), 0, math.cos(theta)]])
	R_z = np.array([[math.cos(theta), -math.sin(theta), 0],
					[math.sin(theta), math.cos(theta), 0],
					[0, 0, 1]])
	vvv = np.array([1,1,1])
	R = np.dot(R_y, vvv)
	R2 = np.dot(R_z, vvv)

	print('R',R)
	print('R2',R2)

	aaaaa = tObj.line_plane_collision(planeNormal, planePoint, np.array([0.94464068, 0.02371891, 0.32724823]), rayPoint)
	bbbbb = tObj.line_plane_collision(planeNormal, planePoint, np.array([0.99979601, -0.01828377,  0.00858109]), rayPoint)
	ccccc = tObj.line_plane_collision(planeNormal, planePoint, np.array([0.99938273, 0.03406657, 0.008581099]), rayPoint)
	ddddd = tObj.line_plane_collision(planeNormal, planePoint, np.array([-0.32626789, -0.0315359,   0.94475116]), rayPoint)

	print ("intersection at", aaaaa)
	print ("intersection at", bbbbb)
	print ("intersection at", ccccc)
	print ("intersection at", ddddd)

	# check_available_point_on_plane(p0, p1, p3, p2, aaaaa)
	# check_available_point_on_plane(p0, p1, p3, p2, bbbbb)
	tObj.check_available_point_on_plane(p0, p1, p3, p2, ccccc)
	# check_available_point_on_plane(p0, p1, p3, p2, ddddd)
	# print(1/0)

	# headOri_deg[1.5, - 9.3,1.5]
	# lpupil_deg[0., - 9.8 , 0.2]
	# print())
	print(np.dot(eulerAnglesToRotationMatrix(np.array([0,0,math.pi])), np.array([1,1,1])).round(5))