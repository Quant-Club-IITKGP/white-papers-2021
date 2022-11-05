# imports needed
import matplotlib.pyplot as plt
import random

# inputs
n = 2500
base = 3 
Van_der_Corput_sequence = []
t = 1
x = []
random_ = []

# function to make one number in the sequence
def Van_der_Corput(c,base):
  convert_1 = []
  i = float(c)
  if i<0:
    t = '-'
    i*=-1
  elif i == 0:
    return 0
  else:
    t = '+'
  while (i):
    d = i%base
    convert_1.append(d)
    i//=base
  i=0
  h=1
  for k in convert_1:
    i = i + k/float(base**h)
    h+=1
  if t=='-':
    i*=-1
  return i

# making of the sequence
for i in range(n):
  Van_der_Corput_sequence.append(Van_der_Corput(i,base))
  x.append(t)
  t+=1
for i in range(n):
  random_.append(random.uniform(0,1))
  
# plots
ptl = plt.figure()
pt = ptl.add_axes([0,0,1,1])
pt.plot(x,random_,'.',color='blue',label='Uniform Random points')
pt.plot(x,Van_der_Corput_sequence,'.', color='cyan', label='Van der Corput sequence points')
ptl.legend()
plt.show()
