import scipy as sp
import numpy as np
from scipy.optimize import fmin
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def myfunc(x):
    return 2*x[0]**4-6*x[0]**3+8*x[1]**3+ 7*x[1]**2+7


x0 = np.arange(1,3.5,.01)
y0= np.arange(1,3.5,.01)

#xy2 = [(i,j) for i in np.linspace(1,3.5,.01) for j in np.linspace(1,3.5,.01)]
x,y=np.meshgrid(x0,y0) #np.array([x0.flatten(),y0.flatten()]).T
#print (xy2)

fval = 2*x**4-6*x**3+8*y**3+ 7*y**2+7
min= fmin(myfunc,[2,2])
fmin = 2*min[0]**4-6*min[0]**3+8*min[1]**3+ 7*min[1]**2+7

print ("Minimum of the function is at : ", min, "Minimum value of function is ",fmin)#"(",min.x[0],min.y[0],min.fun,")" )
#print ("Fun", func)
#plt.figure()
#plt.plot(xy2,myfunc(xy),"r",label ="Function")
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(x, y, fval)
ax.plot(min[0],min[1],fmin)
plt.show()




"""b = np.arange(0.2, 3.2, 0.2)
d = np.arange(0.1, 1.0, 0.1)

B, D = np.meshgrid(b, d)
nu = 2*B**4-6*B**3+8*D**3+ 7*D**2+7#np.sqrt( 1 + (2*D*B)**2 ) / np.sqrt( (1-B**2)**2 + (2*D*B)**2)

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(B, D, nu)
plt.xlabel('b')
plt.ylabel('d')
plt.show()"""



#plt.show()

"""plt.plot(min.x[0],min.fun,"bo",label = "Minimum of Function")
plt.title("Function minimization using Scipy")
plt.xlabel("x"+r"$\rightarrow$")
plt.ylabel("f(x)"+r"$\rightarrow$")
plt.legend()

plt.show()"""
