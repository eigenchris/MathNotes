import numpy as np
import random
from cmath import isclose, sqrt, exp
from math import cos, sin, cosh, sinh, pi


# Check double-sided SU(2) rotations
print("Checking sigma matrices...")

def getTwoByTwoRep(w):
    t,x,y,z = w
    t,x,y,z = t[0],x[0],y[0],z[0]
    return np.array([
        [t+z, x-y*1j],
        [x+y*1j, t-z]
    ])

T = np.array([[1],[0],[0],[0]])
X = np.array([[0],[1],[0],[0]])
Y = np.array([[0],[0],[1],[0]])
Z = np.array([[0],[0],[0],[1]])

I2 = np.array([
    [1,0],
    [0,1]
])
st = I2
sx = np.array([
    [0,1],
    [1,0]
])
sy = np.array([
    [0,-1j],
    [1j,0]
])
sz = np.array([
    [1,0],
    [0,-1]
])

assert np.array_equal(sx.dot(sx), I2)
assert np.array_equal(sy.dot(sy), I2)
assert np.array_equal(sz.dot(sz), I2)
assert np.array_equal(sx.dot(sy), -sy.dot(sx))
assert np.array_equal(sy.dot(sz), -sz.dot(sy))
assert np.array_equal(sz.dot(sx), -sx.dot(sz))

assert np.array_equal(getTwoByTwoRep(T), st)
assert np.array_equal(getTwoByTwoRep(X), sx)
assert np.array_equal(getTwoByTwoRep(Y), sy)
assert np.array_equal(getTwoByTwoRep(Z), sz)


# Check vector lengths
print("Checking lengths vs determinants...")

w = np.random.random((4,1))
v = np.random.random((4,1))
v[0] = 0

V = getTwoByTwoRep(v)
W = getTwoByTwoRep(w)

vx, vy, vz = v[1][0], v[2][0], v[3][0]
l2 = vx*vx + vy*vy + vz*vz
assert isclose(l2, -np.linalg.det(V))

wt, wx, wy, wz = w[0][0], w[1][0], w[2][0], w[3][0]
s2 = wt*wt - wx*wx - wy*wy - wz*wz
assert isclose(s2, np.linalg.det(W))


# Check double-sided transformations
print("Checking double-sided transformations...")

I4 = np.array([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,0],
    [0,0,0,1]
])

def rotXY_vec(th):
    return np.array([
        [1,0,0,0],
        [0,cos(th),-sin(th),0],
        [0,sin(th),cos(th),0],
        [0,0,0,1]
    ])

def rotYZ_vec(th):
    return np.array([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,cos(th),-sin(th)],
        [0,0,sin(th),cos(th)]
    ])

def rotZX_vec(th):
    return np.array([
        [1,0,0,0],
        [0,cos(th),0,sin(th)],
        [0,0,1,0],
        [0,-sin(th),0,cos(th)]
    ])

def boostTX_vec(phi):
    return np.array([
        [cosh(phi),-sinh(phi),0,0],
        [-sinh(phi),cosh(phi),0,0],
        [0,0,1,0],
        [0,0,0,1]
    ])

def boostTY_vec(phi):
    return np.array([
        [cosh(phi),0,-sinh(phi),0],
        [0,1,0,0],
        [-sinh(phi),0,cosh(phi),0],
        [0,0,0,1]
    ])

def boostTZ_vec(phi):
    return np.array([
        [cosh(phi),0,0,-sinh(phi)],
        [0,1,0,0],
        [0,0,1,0],
        [-sinh(phi),0,0,cosh(phi)]
    ])

def rotXY_spin(th):
    return cos(th/2)*I2 - sin(th/2)*(sx.dot(sy))

def rotYZ_spin(th):
    return cos(th/2)*I2 - sin(th/2)*(sy.dot(sz))

def rotZX_spin(th):
    return cos(th/2)*I2 - sin(th/2)*(sz.dot(sx))

def boostTX_spin(phi):
    return cosh(phi/2)*I2 - sinh(phi/2)*(st.dot(sx))

def boostTY_spin(phi):
    return cosh(phi/2)*I2 - sinh(phi/2)*(st.dot(sy))

def boostTZ_spin(phi):
    return cosh(phi/2)*I2 - sinh(phi/2)*(st.dot(sz))

assert np.isclose(rotXY_vec(pi/2).dot(X), Y).all()
assert np.isclose(rotYZ_vec(pi/2).dot(Y), Z).all()
assert np.isclose(rotZX_vec(pi/2).dot(Z), X).all()

boostedX = boostTX_vec(1).dot(T)
boostedY = boostTY_vec(1).dot(T)
boostedZ = boostTZ_vec(1).dot(T)
# bosting ref frame in +x direction causes a vector's x component to become negative
assert abs(boostedX[0][0]) > abs(boostedX[1][0]) and boostedX[1][0] < 0 
assert abs(boostedY[0][0]) > abs(boostedY[2][0]) and boostedY[2][0] < 0
assert abs(boostedZ[0][0]) > abs(boostedZ[3][0]) and boostedZ[3][0] < 0


UXY = rotXY_spin(pi/2)
UYZ = rotYZ_spin(pi/2)
UZX = rotZX_spin(pi/2)
assert np.isclose(UXY.dot(sx).dot(np.conjugate(UXY.T)), sy).all()
assert np.isclose(UYZ.dot(sy).dot(np.conjugate(UYZ.T)), sz).all()
assert np.isclose(UZX.dot(sz).dot(np.conjugate(UZX.T)), sx).all()

assert np.isclose(UXY.dot(W).dot(np.conjugate(UXY.T)), getTwoByTwoRep(rotXY_vec(pi/2).dot(w))).all()
assert np.isclose(UYZ.dot(W).dot(np.conjugate(UYZ.T)), getTwoByTwoRep(rotYZ_vec(pi/2).dot(w))).all()
assert np.isclose(UZX.dot(W).dot(np.conjugate(UZX.T)), getTwoByTwoRep(rotZX_vec(pi/2).dot(w))).all()

LTX = boostTX_spin(1)
LTY = boostTY_spin(1)
LTZ = boostTZ_spin(1)
assert np.isclose(LTX.dot(W).dot(np.conjugate(LTX.T)), getTwoByTwoRep(boostTX_vec(1).dot(w))).all()
assert np.isclose(LTY.dot(W).dot(np.conjugate(LTY.T)), getTwoByTwoRep(boostTY_vec(1).dot(w))).all()
assert np.isclose(LTZ.dot(W).dot(np.conjugate(LTZ.T)), getTwoByTwoRep(boostTZ_vec(1).dot(w))).all()

# Rotations are Unitary; Boosts are Hermitian.
assert np.isclose(UXY.T.conj(), np.linalg.inv(UXY)).all()
assert np.isclose(UYZ.T.conj(), np.linalg.inv(UYZ)).all()
assert np.isclose(UZX.T.conj(), np.linalg.inv(UZX)).all()
assert np.isclose(LTX.T.conj(), LTX).all()
assert np.isclose(LTY.T.conj(), LTY).all()
assert np.isclose(LTZ.T.conj(), LTZ).all()

# Check factoring into Spinors
print("Checking factoring into spinors...")
def factorPauliVector(v):
    _,x,y,z = v
    x,y,z = x[0],y[0],z[0]
    xi1 = sqrt( x-y*1j)
    xi2 = sqrt(-x-y*1j)
    mult = sqrt(-x*x-y*y)
    spinor = np.array([
        [xi1],
        [xi2],
    ])
    dual_spinor = np.array([
        [-xi2, xi1],
    ])
    return spinor, dual_spinor


def factorWeylVector(w):
    t,x,y,z = w
    t,x,y,z = t[0],x[0],y[0],z[0]
    A = abs(sqrt(t+z))
    absB = abs(sqrt(t-z))
    phase_diff = (x+y*1j)/sqrt(x*x+y*y)
    B = absB*phase_diff
    spinor = np.array([
        [A],
        [B],
    ])
    # A is always real, so conjugate does nothing
    dual_spinor = np.array([
        [A,B.conj()],
    ])
    return spinor, dual_spinor

# REMEMBER!!! YOU NEED A MATRIX WITH DETERMINANT ZERO TO FACTOR!!!
vv = np.random.random((4,1))
vv = np.array([
    [0],
    [vv[1][0]],
    [vv[2][0]],
    [1j*sqrt(vv[1][0]*vv[1][0] + vv[2][0]*vv[2][0])]
])
# vv = np.array([
#     [0],
#     [3],
#     [4],
#     [5j],
# ])
ww = np.random.random((4,1))
ww[0] = abs(sqrt(ww[1]*ww[1] + ww[2]*ww[2] + ww[3]*ww[3]))

VV = getTwoByTwoRep(vv)
WW = getTwoByTwoRep(ww)

# comparisons against zero will fail, so do a +1 and compare to +1
assert isclose(abs(np.linalg.det(VV)) + 1, 1)
assert isclose(abs(np.linalg.det(WW)) + 1, 1)

pauli_spinor, pauli_dual = factorPauliVector(vv)
assert np.isclose(pauli_spinor.dot(pauli_dual), VV).all()

xi1 = pauli_spinor[0][0]
xi2 = pauli_spinor[1][0]
assert np.isclose(0.5*(xi1*xi1 - xi2*xi2), vv[1][0])
assert np.isclose(0.5j*(xi1*xi1 + xi2*xi2), vv[2][0])
assert np.isclose(-xi1*xi2, vv[3][0])

weyl_spinor,  weyl_dual  = factorWeylVector(ww)
assert np.isclose(weyl_spinor.dot(weyl_dual), WW).all()

psi1 = weyl_spinor[0][0]
psi2 = weyl_spinor[1][0]
assert np.isclose( 0.5 *(psi1*psi1.conj() + psi2*psi2.conj()), ww[0][0])
assert np.isclose( 0.5 *(psi1*psi1.conj() - psi2*psi2.conj()), ww[3][0])
assert np.isclose( 0.5 *(psi2*psi1.conj() + psi1*psi2.conj()), ww[1][0])
assert np.isclose(-0.5j*(psi2*psi1.conj() - psi1*psi2.conj()), ww[2][0])

# check that overall phases don't change the weyl vector
phase = exp(1j*pi/7)
assert np.isclose(weyl_spinor.dot(weyl_dual), (weyl_spinor*phase).dot(phase.conjugate()*weyl_dual)).all()


# Check inner products
print("Checking inner products... TODO")

# Check the 4 reps
print("Checking left/right representations... TODO")




print("ALL DONE!!")