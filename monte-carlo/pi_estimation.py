# imports needed
import matplotlib.pyplot as plt
import numpy as np
import random

# number of trails
k =20000

l = 0
x = []
y = []
m = []
x1 =[]
x2 = []
y1 =[]
y2=[]

random.seed()

# main processing of the Monte Carlo simulation
for i in range(1, k+1):

  # generating a random uniform point in the square space of one unit area
  xx = random.uniform(0,1)
  yy = random.uniform(0,1)

  # checking if the generated point falls inside (or on) the quarter
  if ((xx**2)+(yy**2))**0.5<=1:
    l += 1
    x1.append(xx)
    y1.append(yy)
  else:
    x2.append(xx)
    y2.append(yy)

  # calculaitng the estimated pi
  y.append(4*l/i)
  x.append(i)
  m.append(3.14159265359)

# plotting
a = np.arange(3,3.3,0.0001)
ptl = plt.figure()
pt = ptl.add_axes([0,0,1,1])
pt.plot(x,y,label='Estimated value of pi')
pt.plot(x,m,label='Value of pi')
plt.xlabel('Trials')
plt.ylabel('Estimated value of pi')
plt.legend()
fig,ax = plt.subplots()
ax.hist(y,bins =a)
plt.show()
ptl = plt.figure()
pt = ptl.add_axes([0,0,1,1])
pt.plot(x1,y1,'.',color='green')
pt.plot(x2,y2,'.',color='blue')
ptl.set_figheight(5)
ptl.set_figwidth(5)
plt.show()

# value of pi that is estimated after the run of the Monte Carlo simulation
print(f'Value of pi calculated is : {4*l/k}')
