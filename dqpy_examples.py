from dqpy import DQ
import numpy as np
print('>>>a = DQ()	# Creates a zero dual quaternion (dq)')
a = DQ()
print(a)
print('>>>b = DQ(1)	# Creates a new unit value dq')
b = DQ(1)
print(b)
print('>>>c = DQ([1, 2, 3, 4])	# Creates a new dq with primary part only')
c = DQ([1, 2, 3, 4])
print(c)
print('>>>d = DQ([1, 2, 3, 4, 5, 6, 7, 8])	# Creates a new dq with primary and secondary parts')
d = DQ([1, 2, 3, 4, 5, 6, 7, 8])
print(d)
print('>>>e = c*d+b 	# Performs multiplication and addition on dqs')
e = c*d+b
print(e)
print('>>>e.re(), e.im()	# Retrieves the real and imaginary parts of the dq respectively')
e.re(), e.im()
print('>>>e.p(), e.d()	# Retrieves the primary and dual parts of the dq respectively')
e.p(), e.d()
print('>>>e.im().d(2)	# Retries the j coefficient of the imaginary part of the dual part')
e.im().d(2)
print('>>>DQ.i(), DQ.j(), DQ.k()	# Returns the i, j, k imaginary units respectively')
DQ.i(), DQ.j(), DQ.k()
print('>>>DQ.e() 	# Returns the dual unit constant')
DQ.e()
print('>>>r1 = DQ([np.cos(np.pi/8), np.sin(np.pi/8), 0, 0]) # Creates a rotation pi/4 at x axis')
r1 = DQ([np.cos(np.pi/8), np.sin(np.pi/8), 0, 0])
print(r1)
print('>>>r2 = DQ([np.cos(np.pi/6), 0, np.sin(np.pi/6), 0]) # Creates a rotation pi/3 at y axis')
r2 = DQ([np.cos(np.pi/6), 0, np.sin(np.pi/6), 0])
print(r2)
print('>>>r3 = DQ.k()*np.sin(np.pi/4) + np.cos(np.pi/4) # Creates a rotation pi/2 at z axis')
r3 = DQ.k()*np.sin(np.pi/4) + np.cos(np.pi/4)
print(r3)
print('>>>r = r1*r2*r3 	# Returns the rotation xyz performed by the previous angles')
r = r1*r2*r3
print(r)
print('>>>r.norm() # Confirms that r is unit dual quaternion')
r.norm()
print('>>>t = DQ.e()*[0, 1, -2, 3] + 1 # Creates a translation 2, -4, 6 in x, y, z axis respectively')
t = DQ.e()*[0, 1, -2, 3] + 1
print(t)
print('>>>p = t*r # Creates a pose with translation t and rotation r')
p = t*r
print(p)
print('p.translation() # Verify the translation')
print(p.translation())
print('np.rad2deg(p.rotation_angle()), p.rotation_axis() # Verify the rotation angle and rotation axis')
print('theta = %.4f deg' % (np.rad2deg(p.rotation_angle())))
print(p.rotation_axis())
print('>>>p.p() == r # Verify if the rotation is equal to r')
p.p() == r
print('>>>p.t() == t # Verify if the translation is equal to t')
p.t() == t
print('>>>p.is_unit() # Verify that p is unit dual quaternion')
p.is_unit()
print('>>>x =  DQ(np.random.randn(8,))	# generate a random dual quaternion')
x =  DQ(np.random.randn(8,))
print('>>>x =  x.normalize()	# normalizes x, that is x now is unit dual quaternion')
x =  x.normalize()
print('>>>x.plot()	# plot x pose (position and orientation)')
x.plot()