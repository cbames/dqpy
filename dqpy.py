import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import warnings

class DQ(object):
    ''' Dual Quaternion class'''

    __threshold = 1e-12
    '''Absolute values below the threshold are considered zero.'''

    def __init__(self, values=0.):
        if not hasattr(values, '__iter__'):
            values = [values]
        values = np.asarray(values)
        self.q = np.r_[values, np.zeros(8 - values.size)]

    def toRound(self, value):
        if type(value) is bool:
            __threshold = value
    
#    Selection and transformation methods
    def p(self, idx=-1):
        '''
        dq.P returns the primary part of the dual quaternion.
        dq.P(index), returns the coefficient corresponding to index.
        '''
        if idx == -1:
            return DQ(self.q[:4])
        else:
            if idx in range(4):
                return self.q[idx]

    def d(self, idx=-1):
        '''
        dq.D returns the dual part of the dual quaternion.
        dq.D(index), returns the coefficient corresponding to index.
        '''
        if idx == -1:
            return DQ(self.q[4:])
        else:
            if idx in range(4):
                return self.q[4+idx]

    def re(self):
        '''Return the real part of the dual quaternion.'''
        return DQ(np.r_[self.q[0], 0, 0, 0, self.q[4]])

    def im(self):
        '''Return the imaginary part of the dual quaternion.'''
        return DQ(np.r_[0, self.q[1:4], 0, self.q[5:8]])

    @staticmethod
    def e():
        '''Dual unit'''
        return DQ([0, 0, 0, 0, 1])

    @staticmethod
    def i():
        '''Imaginary i'''
        return DQ([0, 1, 0, 0])

    @staticmethod
    def j():
        '''Imaginary j'''
        return DQ([0, 0, 1, 0])

    @staticmethod
    def k():
        '''Imaginary k'''
        return DQ([0, 0, 0, 1])
    
    @staticmethod
    def c4():
        ''''matrix C4 that satisfies the relation vec4(x') = C4 * vec4(x).'''
        return np.diag([1, -1, -1, -1])

    @staticmethod
    def c8():
        ''''return C8 matrix that satisfies vec8(x') = C8 * vec8(x).'''
        return np.diag([1, -1, -1, -1]*2)

    def hamiplus4(self):
        '''return the positive Hamilton operator (4x4) of the quaternion.'''
        return np.array([[self.q[0], -self.q[1], -self.q[2], -self.q[4]],\
            [self.q[1], self.q[0], -self.q[3], self.q[2]],\
            [self.q[2], self.q[3], self.q[0], -self.q[1]],
            [self.q[3], -self.q[2], self.q[1], self.q[0]]])

    def hamiminus4(self):
        '''return the negative Hamilton operator (4x4) of the quaternion.'''
        return np.array([[self.q[0], -self.q[1], -self.q[2], -self.q[4]],\
            [self.q[1], self.q[0], self.q[3], -self.q[2]],\
            [self.q[2], -self.q[3], self.q[0], self.q[1]],
            [self.q[3], self.q[2], -self.q[1], self.q[0]]])

    def hamiplus8(self):
        '''return the positive Hamilton operator (8x8) of the dual quaternion.'''
        return np.array([[self.p().hamiplus4(), np.zeros([4, 4])], \
                [self.d().hamiplus4(), self.p().hamiplus4()]])

    def haminus8(self):
        '''return the negative Hamilton operator (8x8) of the dual quaternion.'''
        return np.array([[self.p().hamiminus4(), np.zeros([4, 4])], \
                [self.d().hamiminus4(), self.p().hamiminus4()]])
    
    def vec4(self):
        '''return the vector with the four dual quaternion coefficients.'''
        return self.q[:4]

    def vec8(self):
        '''return the vector with the eight dual quaternion coefficients.'''
        return self.q[:]

#    Math methods
    def acos(self):
        '''
        returns the acos of scalar dual quaternion x. 
        If x is not scalar, the function returns an error.
        '''
        if self.im() != 0. or self.d() != 0.:
            raise ValueError('The dual quaternion is not a scalar.')
        return np.arccos(self.q[0])

    def rotation_angle(self):
        '''returns the rotation angle of dq or an error otherwise.'''
        if not self.is_unit():
            raise ValueError('The dual quaternion does not have unit norm.')
        return self.p().re().acos()*2

    def rotation_axis(self):
        '''returns the rotation axis (nx*i + ny*j + nz*k) of \
        the unit dual quaternion.'''
        if not self.is_unit():
            raise ValueError('The dual quaternion does not have unit norm.')
        phi = np.arccos(self.q[0])
        if phi == 0:
            # Convention = DQ.k. It could be any rotation axis.
            return DQ([0, 0, 0, 1])
        else:
            return self.p().im() * np.sin(phi)**-1

    def translation(self):
        '''returns the translation of the unit dual quaternion.'''
        if not self.is_unit():
            raise ValueError('The dual quaternion does not have unit norm.')
        return self.d() * self.p().conj() * 2

    def t(self):
        '''returns the translation part of dual quaternion.'''
        # More specifically, if x = r+DQ.E*(1/2)*p*r,
        # T(x) returns 1+DQ.E*(0.5)*p
        return self * self.p().conj()

    def conj(self):
        '''returns the conjugate of dq.'''
        return DQ([self.q[0], -self.q[1], -self.q[2], -self.q[3],\
                   self.q[4], -self.q[5], -self.q[6], -self.q[7]])

    def __add__(self, other):
        if other.__class__ != DQ:
            other = DQ(other)
        return DQ(self.q + other.q)

    def __sub__(self, other):
        if other.__class__ != DQ:
            other = DQ(other)
        return DQ(self.q - other.q)

    @staticmethod
    def __quaternion_mul(q_a, q_b):
        '''returns quaterion multiplication.'''
        mat_a = np.array([[q_a[0], -q_a[1], -q_a[2], -q_a[3]], \
                [q_a[1], q_a[0], -q_a[3], q_a[2]], \
                [q_a[2], q_a[3], q_a[0], -q_a[1]], \
                [q_a[3], -q_a[2], q_a[1], q_a[0]]])
        return np.dot(mat_a, q_b)

    def __mul__(self, other):
        '''returns the standard dual quaternion multiplication.'''
        if other.__class__ != DQ:
            other = DQ(other)
        non_dual = DQ.__quaternion_mul(self.q[:4], other.q[:4])
        dual = DQ.__quaternion_mul(self.q[:4], other.q[4:]) + \
            DQ.__quaternion_mul(self.q[4:], other.q[:4])
        return DQ(np.r_[non_dual, dual])

    def inv(self):
        '''returns the inverse of the dual quaternion.'''
        dq_c = self.conj()
        dq_q = self * dq_c
        inv = DQ([1/dq_q.q[0], 0, 0, 0, -dq_q.q[4]/(dq_q.q[0]**2)])
        return dq_c*inv

    def log(self):
        '''returns the logarithm of the dual quaternion.'''
        if not self.is_unit():
            raise ValueError('The log function is currently defined \
                only for unit dual quaternions.')
        prim = self.rotation_axis() * (self.rotation_angle() * .5)
        dual = self.translation() * .5
        return prim + dual * DQ.e()
        
    def exp(self):
        '''returns the exponential of the pure dual quaternion'''
        if self.re() != 0:
            raise ValueError('The exponential operation is defined only \
                for pure dual quaternions.')
        phi = np.linalg.norm(self.p().q[:])
        if phi != 0:
            prim = self.p() * (np.sin(phi)/phi) + np.cos(phi)
        else:
            prim = DQ([1])
        return prim + self.d() * prim * DQ.e()

    def __pow__(self, num):
        '''
        returns the dual quaternion corresponding to the operation
        exp(m*log(dq)), where dq is a dual quaternion and m is a real number.
        For the moment, this operation is defined only for unit dual quaternions.
        '''
        if not isinstance(num, (int, float)):
            raise ValueError('The second parameter must be a double.')
        return (self.log() * num).exp()

    def norm(self):
        '''returns the dual scalar corresponding to the norm of dq.'''
        dq_a = self.conj()*self
    # Taking the square root (compatible with the definition of quaternion norm)
    # This is performed based on the Taylor expansion.
        if dq_a.p() == 0:
            return DQ([0])
        else:
            dq_a.q[0] = np.sqrt(dq_a.q[0])
            dq_a.q[4] = dq_a.q[4] / (2*dq_a.q[0])
            for i in range(8):
                if np.abs(dq_a.q[i]) < DQ.__threshold:
                    dq_a.q[i] = 0
            return dq_a

    def normalize(self):
        '''normalizes the dual quaternion.'''
        return self * self.norm().inv()

    def cross(self, other):
        '''returns the cross product between two dual quaternions.'''
        return (self*other - other*self) * .5

    def dot(self, other):
        '''returns the dot product between two dual quaternions.'''
        return  self.p(0)*other.p(0) + self.p(1)*other.p(1) + self.p(2)*other.p(2) + self.p(3)*other.p(3) + self.d(0)*other.d(0) +self.d(1)*other.d(1) +self.d(2)*other.d(2) +self.d(3)*other.d(3)

    def pinv(self):
        '''returns the inverse of dq under the decompositional.'''
        conj = self.conj()
        tmp = self.t() * conj.t()
        return tmp.conj() * conj

    def dec(self, other):
        '''returns decompositional multiplication between dual quaternions.'''
        return self.t() * other.t() * self.p() * other.p()

#    Comparison and other methods
    def __pos__(self):
        return self

    def __neg__(self):
        return DQ(np.r_[self.q * -1])

    def __invert__(self):
        '''behaves as conjugate'''
        return self.conj()

    def is_unit(self):
        '''returns True if dq is a unit norm dual quaternion, False otherwise.'''
        if self.norm() == 1:
            return True
        else:
            return False

    def __eq__(self, other):
        if other.__class__ != DQ:
            other = DQ(other)
        return np.all(np.abs(self.q - other.q) < DQ.__threshold)

    def __ne__(self, other):
        return not DQ.__eq__(self, other)

    def __str__(self):
        und = ['', 'i', 'j', 'k']
        sig = ['']*4
        str1 = []
        for i in range(4):
            if self.q[i] >= 0:
                sig[i] = '+'
            if self.q[i] != 0:
                str1.append('%s%s%s' % (sig[i], self.q[i], und[i]))
        str1 = ' '.join(str1)
        if not str1:
            str1 = '0'
        if np.any(self.q[4:] != 0):
            sig = ['']*4
            str2 = ['E*(']
            for i in range(4):
                if self.q[4+i] >= 0 and i > 0:
                    sig[i] = '+'
                if self.q[4+i] != 0:
                    str2.append('%s%s%s' % (sig[i], self.q[4+i], und[i]))
            str2.append(')'); str2 = ' '.join(str2)
            if str1[0] == '+':
                str1 = str1[1:]
            if str2[0] == '+':
                str2 = str2[1:]
            if str1 != '0':
                str1 = '( ' + str1 + ' ) + ' + str2
            else:
                str1 = str2
        return str1

    def __repr__(self):
        return str(self)

    def round(self):
        '''Round absolute values below the threshold to zero.'''
        for i in range(8):
            if np.abs(self.q[i]) < DQ.__threshold:
                self.q[i] = 0.;
        return DQ(self.q)

    def plot(self, scale = 1.):
        '''
        DQ.plot(dq, OPTIONS) plots the dual quaternion dq.
        Ex.: plot(dq, scale=5) will plot dq with the axis scaled by a factor of 5
        '''
        if self.is_unit():

            # create unit vectors and rotate them by the quaternion part of dq.
            t1 = DQ.e() * [0, scale*.5, 0, 0] + 1
            t2 = DQ.e() * [0, 0, scale*.5, 0] + 1
            t3 = DQ.e() * [0, 0, 0, scale*.5] + 1

            # vector rotation
            xvec = self.p() * t1 * self.p().conj()
            yvec = self.p() * t2 * self.p().conj()
            zvec = self.p() * t3 * self.p().conj()

            # collecting points
            xx, xy, xz = np.array([0, xvec.translation().q[1]]) + self.translation().q[1], \
                np.array([0, xvec.translation().q[2]]) + self.translation().q[2], \
                np.array([0, xvec.translation().q[3]]) + self.translation().q[3]
            yx, yy, yz = np.array([0, yvec.translation().q[1]]) + self.translation().q[1], \
                np.array([0, yvec.translation().q[2]]) + self.translation().q[2], \
                np.array([0, yvec.translation().q[3]]) + self.translation().q[3]
            zx, zy, zz = np.array([0, zvec.translation().q[1]]) + self.translation().q[1], \
                np.array([0, zvec.translation().q[2]]) + self.translation().q[2], \
                np.array([0, zvec.translation().q[3]]) + self.translation().q[3]

            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.plot(xx, xy, xz, 'r')
            ax.plot(yx, yy, yz, 'g')
            ax.plot(zx, zy, zz, 'b')
            ax.relim()
            #ax.set_aspect('equal')
            #ax.set_aspect(1./ax.get_data_ratio())
            ax.autoscale_view(True,True,True)
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
            ax.set_zlabel('z (m)')
            plt.show(block=False)
        else:
            warnings.warn('Only unit dual quaternions can be plotted!')
