# imports needed
import random
import matplotlib.pyplot as plt

# number of trails in each run
o = [100, 10000, 1000000]

random.seed()

# each iteration of this loop runs one Monte Carlo simulation
for k in o:
  l = 0
  x = []
  y = []
  m = []

  # main processing of the Monte Carlo simulation
  for i in range(1,k+1):

    # getting a random value uniformly in the range [0,1]
    s = random.uniform(0,1)

    # assigning the value as head for being greater than 0.5
    # otherwise it is made a tail
    if s>=0.5:
      t = 1
    else:
      t = 0
    
    # calculating the cumulative average after each trail
    l = (t + l*(i-1))/i
    x.append(i)
    y.append(l)
    m.append(0.5)
  
  # plotting
  ptl = plt.figure()
  pt = ptl.add_axes([0,0,1,1])
  pt.plot(x,y,label='Average of the trails')
  pt.plot(x,m,label='Expected')
  plt.xlabel('Trials')
  plt.ylabel('Probability of getting a head')
  pt.set_ylim(0,1)
  pt.legend()
  plt.show()
