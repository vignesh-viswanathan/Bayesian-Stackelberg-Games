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

def generatePureStrategy(AttackerRF):
    purestrategylist = []
    typecounter = [0]
    limit = len(AttackerRF[0])
    while(typecounter[0]!=limit):        
               
        strategy = [0]*limit
        strategy[typecounter[0]]=1
        
        purestrategylist.append(strategy)
        typecounter[0]+=1
    return purestrategylist

def solveRestrictedGame(DefenderRF,AttackerRF):
    PureStrategyList = generatePureStrategy(AttackerRF)
    Delete = []
    Bound = [10000000]*len(PureStrategyList)
    rewardmax = -10000000
    deltamax = []
    count=0

    for i in range(0,len(PureStrategyList)):
        try:
            prob = pic.Problem()
            m = len(DefenderRF[0])
            l = len(DefenderRF)
            t=1
            delta  = prob.add_variable('delta',l,lower=0,upper=1)
            rfa = pic.new_param('rfa',AttackerRF)
            rfd = pic.new_param('rfd',DefenderRF)
            sigma = pic.new_param('sigma',PureStrategyList[i])
            purestrategylist = pic.new_param('purestrategylist',PureStrategyList)
            prob.add_constraint(pic.sum([delta[j] for j in range(l)],'j','[l]')==1)
            attrew=delta.T*rfa*(sigma)
            #print(purestrategylist)
            for k in range(0,len(PureStrategyList)):
                prob.add_constraint(attrew>delta.T*rfa*(purestrategylist[k].T))

            obj = delta.T*rfd*(sigma) 
            prob.set_objective('max',obj)
            prob.solve(verbose=0)
            print(obj)
            if(np.array(obj)>rewardmax):
                rewardmax = np.array(obj)
                deltamax = np.array(delta)
            Bound[count]=np.array(obj)
            count+=1
        except:
            Delete.append(PureStrategyList[i])
            print("boohoo")
    for i in range(0,len(Delete)):
        PureStrategyList.remove(Delete[i])
    return PureStrategyList,Bound[0:len(PureStrategyList)],deltamax

def generateCompleteList(PureStrategy, Bound):
    completelist =[]
    boundlist =[]
    typecounter = [0]*len(PureStrategy)
    while (typecounter[0]<len(PureStrategy[0])):
        purestrategy = []
        boundval = 0
        for i in range(0,len(PureStrategy)):
            strategy = PureStrategy[i][typecounter[i]]
            boundval+= Prob[i]*Bound[i][typecounter[i]]
            purestrategy.append(strategy)
        completelist.append(purestrategy)
        boundlist.append(boundval)
        typecounter[len(PureStrategy)-1]+=1
        for i in range(-len(PureStrategy)+1,0):
            ind = -i
            if(typecounter[ind]==len(PureStrategy[ind])):
                typecounter[ind]=0
                typecounter[ind-1]+=1
    return completelist,boundlist    

def solveCompleteGame(DefenderRF,AttackerRF,PureStrategyList,BoundList,Prob):
    rewardmax = -10000000
    deltamax = []

    for i in range(0,len(PureStrategyList)):
        
        #print(BoundList[i])
        if(np.array(BoundList[i].value)<np.array(rewardmax)):
            continue
        try:
            
            prob = pic.Problem()
            m = len(DefenderRF[0][0])
            l = len(DefenderRF[0])
            t = len(DefenderRF)
            delta  = prob.add_variable('delta',l,lower=0,upper=1)
            rfa = pic.new_param('rfa',AttackerRF)
            rfd = pic.new_param('rfd',DefenderRF)
            probs = pic.new_param('prob',Prob)
            sigma = pic.new_param('sigma',PureStrategyList[i])
            purestrategylist = pic.new_param('purestrategylist',PureStrategyList)
            prob.add_constraint(pic.sum([delta[j] for j in range(l)],'j','[l]')==1)
            attrew=pic.sum([probs[p]*delta.T*rfa[p]*(sigma[p,:]).T for p in range(t)],'p','[t]')
            for k in range(0,len(PureStrategyList)):
                prob.add_constraint(attrew>pic.sum([probs[p]*delta.T*rfa[p]*(purestrategylist[k][p,:]).T for p in range(t)],'p','[t]'))
            obj=0
            for j in range(0,t):
                obj=obj+probs[j]*delta.T*rfd[j]*(sigma[j,:]).T
            #obj = pic.sum([probs[p]*delta.T*rfd[p]*(sigma[p,:]).T for p in range(t)],'p','[t]')
            prob.set_objective('max',obj)
            #print(prob)
            prob.solve(verbose=0)
            print(obj)
            if((obj.value[0])>rewardmax):
                rewardmax = (obj.value[0])
                deltamax = np.array(delta)
        except:
            print("boohoo")
    return deltamax,rewardmax

def HBGS(AttackerRF,DefenderRF,Prob):
    PureStrategy = []
    BoundList = []
    for i in range(0,len(AttackerRF)):
        ps, b, d = solveRestrictedGame(DefenderRF[i],AttackerRF[i])
        PureStrategy.append(ps)
        BoundList.append(b)
    
    cl,bl = generateCompleteList(PureStrategy,BoundList)
    print("Starting complete game")
    d,rew =solveCompleteGame(DefenderRF,AttackerRF,cl,bl,Prob)
    return d,rew

d0,rew0 = HBGS(AttackerRF,DefenderRF,Prob)
print("Final Solution:\n")
print("Stackelberg Strategy:\n",d0)
print("Expected reward: ",rew0)