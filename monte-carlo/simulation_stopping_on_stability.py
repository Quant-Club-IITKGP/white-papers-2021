# imports needed
import matplotlib.pyplot as plt
import math
import random

# inputs
error = float(input("Enter the error limit in the decimals: "))
value = 272.745090535
starting = 0.2
ending = 7.7

l = 0
i = 0
x = []
y = []
final = []
m = []
streak = 0
err = []
domm = ending - starting

# formula
def formula (a):
  b = ((math.sin(a))/a + a**3.2 + math.exp(-5.4*a))**0.73
  return b

# main processing of the monte carlo
while True:
  i = i + 1
  xx = random.uniform(starting, ending)
  yy = formula(xx)
  l = (l*(i-1) + yy*domm)/(i)
  x.append(i)
  y.append(l)
  s = 0
  if (i>20):
    for j in range(i-22, i):
      s = s + (y[j]-y[j-1])**2
    s = s/20
    s = s**0.5
    if s<error:
      streak = streak + 1
    else:
      streak = 0
  err.append(s)
  if streak==20:
    break

for j in range(i):
  final.append(l)
  m.append(value)

# plots
print()
ptl = plt.figure()
pt = ptl.add_axes([0,0,1,1])
pt.plot(x,y,label='Estimated Value')
pt.plot(x,m, label='Actual value')
pt.plot(x,final, label='Estimated final value')
pt.legend()
plt.xlabel('Trials')
plt.ylabel('Estimated value of the integral')
plt.show()
print()
# prints
print(f'Estimated Value = {l}')
print(f'Actual value= {value}')
if l>value:
  s = l - value
else:
  s = value - l
print(f'The error% in calculation = {(1 -(l/value))*100}')
print(f'The number of iterations needed = {i}')
