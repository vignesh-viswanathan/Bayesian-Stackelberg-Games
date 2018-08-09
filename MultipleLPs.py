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

def generatePureStrategyMLP(AttackerRF):
    purestrategylist = []
    typecounter = [0]*len(AttackerRF)
    limit = len(AttackerRF[0][0])
    while(typecounter[0]!=limit):
        
        purestrategy = []
        for i in range(0,len(AttackerRF)):
            strategy = [0]*limit
            strategy[typecounter[i]]=1
            purestrategy.append(strategy)
        purestrategylist.append(purestrategy)
        typecounter[len(AttackerRF)-1]+=1
        for i in range(-len(AttackerRF)+1,0):
            ind = -i
            if(typecounter[ind]==limit):
                typecounter[ind]=0
                typecounter[ind-1]+=1
    return purestrategylist

def MultipleLPSolver(AttackerRF,DefenderRF,Prob):
    PureStrategyList = generatePureStrategyMLP(AttackerRF)
    rewardmax = -10000000
    deltamax = []

    for i in range(0,len(PureStrategyList)):
        try:
            prob = pic.Problem()
            m = len(DefenderRF[0][0])
            l = len(DefenderRF[0])
            t = len(DefenderRF)
            delta  = prob.add_variable('delta',l,lower=0,upper=1)
            pr = pic.new_param('pr',Prob)
            rfa = pic.new_param('rfa',AttackerRF)
            rfd = pic.new_param('rfd',DefenderRF)
            sigma = pic.new_param('sigma',PureStrategyList[i])
            purestrategylist = pic.new_param('purestrategylist',PureStrategyList)
            prob.add_constraint(pic.sum([delta[j] for j in range(l)],'j','[l]')==1)
            attrew=pic.sum([pr[p]*delta.T*rfa[p]*(sigma[p,:]).T for p in range(t)],'p','[t]')
            for k in range(0,len(PureStrategyList)):
                prob.add_constraint(attrew>pic.sum([pr[p]*delta.T*rfa[p]*(purestrategylist[k][p,:]).T for p in range(t)],'p','[t]'))

            obj = pic.sum([pr[p]*delta.T*rfd[p]*(sigma[p,:]).T for p in range(t)],'p','[t]')
            prob.set_objective('max',obj)
            #print(prob)
            prob.solve(verbose=0)
            print(obj)
            print(rewardmax)
            if((obj.value[0])>rewardmax):
                rewardmax = (obj.value[0])
                deltamax = np.array(delta)
            else:
                print("Kaboom")
        except:
            print("boohoo")
            
    return deltamax,rewardmax

d0,rew0 = MultipleLPSolver(AttackerRF,DefenderRF,Prob)
print("Final Solution:\n")
print("Stackelberg Strategy:\n",d0)
print("Expected reward: ",rew0)
