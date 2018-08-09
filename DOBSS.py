import numpy as np
import picos as pic
import cvxopt as cvx
import random

typenum = random.randint(1,4)

defnum = random.randint(2,5)
attnum = random.randint(2,5)
DefenderRF = [[[random.randint(-5,6) for p in range(attnum)] for q in range(defnum)] for r in range(typenum)]
AttackerRF = [[[random.randint(-5,6) for p in range(attnum)] for q in range(defnum)] for r in range(typenum)]

Prob = [random.randint(1,10) for p in range(typenum)]
s = sum(Prob)
for i in range(0,len(Prob)):
    Prob[i]=Prob[i]/s

def DOBSS(AttackerRF,DefenderRF,Prob):
    prob = pic.Problem()
    l=len(DefenderRF[0])
    m=len(DefenderRF[0][0])
    tnum=len(DefenderRF)
    x = prob.add_variable('x',l,lower=0,upper=1)
    na = prob.add_variable('na',(tnum,m),'binary')
    v = prob.add_variable('v',tnum)
    rfa = pic.new_param('rfa',AttackerRF)
    rfd = pic.new_param('rfd',DefenderRF)
    pr = pic.new_param('pr',Prob)

    prob.add_constraint(pic.sum([x[i] for i in range(l)],'i','[l]')==1)
    for j in range(0,tnum):
        prob.add_constraint(pic.sum([na[j,i] for i in range(m)],'i','[m]')==1)

    for j in range(0,tnum):
        prob.add_list_of_constraints([v[j]-(x.T*rfa[j])[i]>0 for i in range(m)],'i','[m]')
        prob.add_list_of_constraints([v[j]-(x.T*rfa[j])[i]<1000000*(1-na[j,i]) for i in range(m)],'i','[m]')

    obj=0
    for j in range(0,tnum):
        obj=obj+pr[j]*x.T*rfd[j]*(na[j,:]).T

    #obj = pic.sum([pr[p]*x.T*rfd[p]*(na[p,:]).T for p in range(tnum)],'p','[tnum]') 
    prob.set_objective('max',obj)
    prob.solve(verbose=0)
    return np.array(x),np.array(obj)

d0,rew0 = DOBSS(AttackerRF,DefenderRF,Prob)
print("Final Solution:\n")
print("Stackelberg Strategy:\n",d0)
print("Expected reward: ",rew0)


