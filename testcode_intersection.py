
#Line-plane intersection
#By Tim Sheerman-Chase
#This software may be reused under the CC0 license

#Based on http://geomalgorithms.com/a05-_intersect-1.html
from __future__ import print_function
import numpy as np

def line_plane_collision(planeNormal, planePoint, rayDirection, rayPoint, epsilon=1e-6):

	ndotu = planeNormal.dot(rayDirection)
	if abs(ndotu) < epsilon:
		raise RuntimeError("no intersection or line is within plane")

	w = rayPoint - planePoint
	si = -planeNormal.dot(w) / ndotu
	Psi = w + si * rayDirection + planePoint
	return Psi

# /**
#  * Determines the point of intersection between a plane defined by a point and a normal vector and a line defined by a point and a direction vector.
#  *
#  * @param planePoint    A point on the plane.
#  * @param planeNormal   The normal vector of the plane.
#  * @param linePoint     A point on the line.
#  * @param lineDirection The direction vector of the line.
#  * @return The point of intersection between the line and the plane, null if the line is parallel to the plane.
#  */
def lineIntersection(planePoint, planeNormal, linePoint, lineDirection):
	line_unitvec = lineDirection
	# line_unitvec = lineDirection / np.linalg.norm(lineDirection)
	# if (planeNormal.dot(np.linalg.norm(lineDirection)) == 0):
	# 	return "NULL"
	#
	# t = (planeNormal.dot(planePoint) - planeNormal.dot(linePoint)) / planeNormal.dot(np.linalg.norm(lineDirection))
	# print('t', t)
	# return linePoint + np.linalg.norm(lineDirection) * t
	print(np.linalg.norm(lineDirection))
	print(np.dot(planeNormal, lineDirection/np.linalg.norm(lineDirection)))
	if (planeNormal.dot(line_unitvec)  == 0):
		return "NULL"

	t = (planeNormal.dot(planePoint) - planeNormal.dot(linePoint)) / planeNormal.dot(line_unitvec)
	print('t', t)
	return linePoint + (line_unitvec * t)

def lineIntersection2(planePoint, planeNormal, linePoint, lineDirection):
	# //ax × bx + ay × by
	if (planeNormal[0] * lineDirection[0] + planeNormal[1] * lineDirection[1] == 0):
		return "NULL"

	# // Ref for dot product calculation: https://www.mathsisfun.com/algebra/vectors-dot-product.html
	dot2 = (planeNormal[0] * planePoint[0] + planeNormal[1] * planePoint[1])
	dot3 = (planeNormal[0] * linePoint[0] + planeNormal[1] * linePoint[1])
	dot4 = (planeNormal[0] * lineDirection[0] + planeNormal[1] * lineDirection[1])

	t = (dot2 - dot3) / dot4
	print('t',t)
	return linePoint + lineDirection * t

def is_sameside_on_line(a, b, p, q):
	print('\na',a, 'b',b,'p',p,'q',q)
	# x = (b - a) * (p - a)
	# y = (b - a) * (q - a)
	# print('x',x,'y',y, 'x @ y >= 0',x @ y, x @ y >= 0)

	c1 = np.cross(b - a, p - a)
	c2 = np.cross(b - a, q - a)
	print('c1',c1,'c2',c2, 'c1*c2 >= 0',c1 * c2 , np.dot(c1,c2), np.dot(c1,c2) >= 0)
	print('')
	return np.dot(c1, c2) >= 0

def normal_vector_from_plane(p0, p1, p2):
	return np.cross((p2 - p0), (p1 - p2)) / np.linalg.norm(np.cross((p2 - p0), (p1 - p2)))

def is_inside_triangle(a, b, c, p):
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
	return all(is_sameside_on_line(*xs[i:i+3], p) for i in range(3))

def is_inside_plane(a, b, c, d, p):
	# print(is_sameside_on_line(a, b, c, p))
	# print(is_sameside_on_line(a, d, b, p))
	# print(is_sameside_on_line(c, b, a, p))
	# print(is_sameside_on_line(c, d, a, p))

	# xs = (a, b, c, d) * 2
	# print(xs)
	# # print(xs[i:i+3] for i in range(3))
	# # print(all(is_sameside_on_line(*xs[i:i+3], p) for i in range(3)))
	# # return 'NULL'
	return all([is_sameside_on_line(a, b, c, p),is_sameside_on_line(a, d, b, p),
				is_sameside_on_line(c, b, a, p),is_sameside_on_line(c, d, a, p)])

def is_same_on_plane(p0, p1, p2, pnt, epsilon=3):
	tnormal_basic_plane =  normal_vector_from_plane(p0,p1,p2)
	tnormal_test_plane = normal_vector_from_plane(p0,p1,pnt)
	print("  tnormal_basic_plane",tnormal_basic_plane,"\n  tnormal_test_plane",tnormal_test_plane)
	print(" ",np.round(tnormal_basic_plane,epsilon) == np.round(tnormal_test_plane,epsilon))
	# print(" ",all(np.round(tnormal_basic_plane,epsilon) == np.round(tnormal_test_plane,epsilon)))
	# print(np.dot(tnormal_basic_plane,tnormal_test_plane))
	print(" ",np.round(np.dot(tnormal_basic_plane, tnormal_test_plane),epsilon))

	# print('N of plane',normal_vector_from_plane(p0,p1,p2))
	# print('N of plane',normal_vector_from_plane(sa,sb,sq6))
	# print("///square result=",is_inside_plane(sa, sb, sc, sd, sq6))
	return all(np.round(tnormal_basic_plane,epsilon) == np.round(tnormal_test_plane,epsilon))

def check_available_point_on_plane(p0, p1, p3, p2, tpnt):
	ret_inside = is_inside_plane(p0, p1, p3, p2, tpnt)
	ret_on_plane = is_same_on_plane(p0, p1, p3, tpnt)
	print(" ret inside=",ret_inside,", ret_on_plane", ret_on_plane)
	print("final",all([ret_inside, ret_on_plane]))
	return all([ret_inside, ret_on_plane])

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
	rayDirection = np.array([0.94451975, 0.02913004, 0.32716034])
	rayPoint = np.array([1501,	-422,	857]) #Any point along the ray

	Psi = line_plane_collision(planeNormal, planePoint, rayDirection, rayPoint)
	print ("intersection at", Psi)

	result = lineIntersection(planeNormal, planePoint, rayDirection, rayPoint)
	print ("lineIntersection1 at", result)

	result = lineIntersection2(planeNormal, planePoint, rayDirection, rayPoint)
	print ("lineIntersection2 at", result)

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
	print("///triangle result=",is_inside_triangle(a, b, c, q5))

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

	print("///square result=",is_inside_plane(sa, sb, sc, sd, sq6))
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

	print("///square result=", is_inside_plane(p0, p1, p3, p2, tpnt))

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


	print('N of plane',normal_vector_from_plane(sa,sb,sc))
	print('N of plane',normal_vector_from_plane(sa,sb,sq6))
	print(is_same_on_plane(sa, sb, sc, sq6))
	print(is_same_on_plane(sa, sb, sc, np.array((8, 4, 1))))
	print(is_same_on_plane(p0, p1, p2, tpnt2))
	print(is_same_on_plane(p0, p1, p2, tpnt))

	check_available_point_on_plane(p0, p1, p3, p2, tpnt)
	check_available_point_on_plane(p0, p1, p3, p2, tpnt2)
	check_available_point_on_plane(sa, sb, sc, sd, np.array((8, 4, 1)))
	check_available_point_on_plane(sa, sb, sc, sd, sq6)
	check_available_point_on_plane(p0, p1, p3, p2, tpnt3)
	check_available_point_on_plane(p0, p1, p3, p2, tpnt4)
	check_available_point_on_plane(p0, p1, p3, p2, tpnt5)


