from collections import defaultdict
from typing import Tuple
from operator import itemgetter
import random
import cmath
import logging
# Importing important imports
import math
import sys
import traceback
from collections import namedtuple
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from typing import List
import operator
from random import randint
from secrets import SystemRandom
import numpy as np
"""
testing please ignore agent by dllu
http://www.rpscontest.com/entry/342001
"""
TESTING_PLEASE_IGNORE = """
from collections import defaultdict
import operator
import random


if input == "":
    score  = {'RR': 0, 'PP': 0, 'SS': 0, \
              'PR': 1, 'RS': 1, 'SP': 1, \
              'RP': -1, 'SR': -1, 'PS': -1,}
    cscore = {'RR': 'r', 'PP': 'r', 'SS': 'r', \
              'PR': 'b', 'RS': 'b', 'SP': 'b', \
              'RP': 'c', 'SR': 'c', 'PS': 'c',}
    beat = {'P': 'S', 'S': 'R', 'R': 'P'}
    cede = {'P': 'R', 'S': 'P', 'R': 'S'}
    rps = ['R', 'P', 'S']
    wlt = {1:0,-1:1,0:2}
    
    def counter_prob(probs):
        weighted_list = []
        for h in rps:
            weighted = 0
            for p in probs.keys():
                points = score[h+p]
                prob = probs[p]
                weighted += points * prob
            weighted_list.append((h, weighted))

        return max(weighted_list, key=operator.itemgetter(1))[0]

    played_probs = defaultdict(lambda: 1)
    dna_probs = [defaultdict(lambda: defaultdict(lambda: 1)) for i in range(18)]
    
    wlt_probs = [defaultdict(lambda: 1) for i in range(9)]

    answers = [{'c': 1, 'b': 1, 'r': 1} for i in range(12)]
    
    patterndict = [defaultdict(str) for i in range(6)]
    
    consec_strat_usage = [[0]*6,[0]*6,[0]*6] #consecutive strategy usage
    consec_strat_candy = [[],   [],   []   ] #consecutive strategy candidates

    output = random.choice(rps)
    histories = ["","",""]
    dna = ["" for i in range(12)]

    sc = 0
    strats = [[] for i in range(3)] 
else:
    prev_sc = sc

    sc = score[output + input]
    for j in range(3):
        prev_strats = strats[j][:]
        for i, c in enumerate(consec_strat_candy[j]):
            if c == input:
                consec_strat_usage[j][i] += 1
            else:
                consec_strat_usage[j][i] = 0
        m = max(consec_strat_usage[j])
        strats[j] = [i for i, c in enumerate(consec_strat_candy[j]) if consec_strat_usage[j][i] == m]
        
        for s1 in prev_strats:
            for s2 in strats[j]:
                wlt_probs[j*3+wlt[prev_sc]][chr(s1)+chr(s2)] += 1
                
        if dna[2*j+0] and dna[2*j+1]:
            answers[2*j+0][cscore[input+dna[2*j+0]]] += 1
            answers[2*j+1][cscore[input+dna[2*j+1]]] += 1	
        if dna[2*j+6] and dna[2*j+7]:
            answers[2*j+6][cscore[input+dna[2*j+6]]] += 1
            answers[2*j+7][cscore[input+dna[2*j+7]]] += 1

        for length in range(min(10, len(histories[j])), 0, -2):
            pattern = patterndict[2*j][histories[j][-length:]]
            if pattern:
                for length2 in range(min(10, len(pattern)), 0, -2):
                    patterndict[2*j+1][pattern[-length2:]] += output + input
            patterndict[2*j][histories[j][-length:]] += output + input
    played_probs[input] += 1
    dna_probs[0][dna[0]][input] +=1
    dna_probs[1][dna[1]][input] +=1
    dna_probs[2][dna[1]+dna[0]][input] +=1
    dna_probs[9][dna[6]][input] +=1
    dna_probs[10][dna[6]][input] +=1
    dna_probs[11][dna[7]+dna[6]][input] +=1

    histories[0] += output + input
    histories[1] += input
    histories[2] += output
    
    dna = ["" for i in range(12)]
    for j in range(3):
        for length in range(min(10, len(histories[j])), 0, -2):
            pattern = patterndict[2*j][histories[j][-length:]]
            if pattern != "":
                dna[2*j+1] = pattern[-2]
                dna[2*j+0]  = pattern[-1]
                for length2 in range(min(10, len(pattern)), 0, -2):
                    pattern2 = patterndict[2*j+1][pattern[-length2:]]
                    if pattern2 != "":
                        dna[2*j+7] = pattern2[-2]
                        dna[2*j+6] = pattern2[-1]
                        break
                break

    probs = {}
    for hand in rps:
        probs[hand] = played_probs[hand]
        
    for j in range(3):
        if dna[j*2] and dna[j*2+1]:
            for hand in rps:
                probs[hand] *= dna_probs[j*3+0][dna[j*2+0]][hand] * \
                               dna_probs[j*3+1][dna[j*2+1]][hand] * \
                               dna_probs[j*3+2][dna[j*2+1]+dna[j*2+0]][hand]
                probs[hand] *= answers[j*2+0][cscore[hand+dna[j*2+0]]] * \
                               answers[j*2+1][cscore[hand+dna[j*2+1]]]
            consec_strat_candy[j] = [dna[j*2+0], beat[dna[j*2+0]], cede[dna[j*2+0]],\
                                     dna[j*2+1], beat[dna[j*2+1]], cede[dna[j*2+1]]]
            strats_for_hand = {'R': [], 'P': [], 'S': []}
            for i, c in enumerate(consec_strat_candy[j]):
                strats_for_hand[c].append(i)
            pr = wlt_probs[wlt[sc]+3*j]
            for hand in rps:
                for s1 in strats[j]:
                    for s2 in strats_for_hand[hand]:
                        probs[hand] *= pr[chr(s1)+chr(s2)]
        else:
            consec_strat_candy[j] = []
    for j in range(3):
        if dna[j*2+6] and dna[j*2+7]:
            for hand in rps:
                probs[hand] *= dna_probs[j*3+9][dna[j*2+6]][hand] * \
                               dna_probs[j*3+10][dna[j*2+7]][hand] * \
                               dna_probs[j*3+11][dna[j*2+7]+dna[j*2+6]][hand]
                probs[hand] *= answers[j*2+6][cscore[hand+dna[j*2+6]]] * \
                               answers[j*2+7][cscore[hand+dna[j*2+7]]]

    output = counter_prob(probs)
"""

"""
centrifugal bumblepuppy 4 bot by dllu
http://www.rpscontest.com/entry/161004
"""
CENTRIFUGAL_BUMBLEPUPPY_4 = """
#                         WoofWoofWoof
#                     Woof            Woof
#                Woof                      Woof
#              Woof                          Woof
#             Woof  Centrifugal Bumble-puppy  Woof
#              Woof                          Woof
#                Woof                      Woof
#                     Woof            Woof
#                         WoofWoofWoof

import random

number_of_predictors = 60 #yes, this really has 60 predictors.
number_of_metapredictors = 4 #actually, I lied! This has 240 predictors.


if not input:
    limits = [50,20,6]
    beat={'R':'P','P':'S','S':'R'}
    urmoves=""
    mymoves=""
    DNAmoves=""
    outputs=[random.choice("RPS")]*number_of_metapredictors
    predictorscore1=[3]*number_of_predictors
    predictorscore2=[3]*number_of_predictors
    predictorscore3=[3]*number_of_predictors
    predictorscore4=[3]*number_of_predictors
    nuclease={'RP':'a','PS':'b','SR':'c','PR':'d','SP':'e','RS':'f','RR':'g','PP':'h','SS':'i'}
    length=0
    predictors=[random.choice("RPS")]*number_of_predictors
    metapredictors=[random.choice("RPS")]*number_of_metapredictors
    metapredictorscore=[3]*number_of_metapredictors
else:

    for i in range(number_of_predictors):
        #metapredictor 1
        predictorscore1[i]*=0.8
        predictorscore1[i]+=(input==predictors[i])*3
        predictorscore1[i]-=(input==beat[beat[predictors[i]]])*3
        #metapredictor 2: beat metapredictor 1 (probably contains a bug)
        predictorscore2[i]*=0.8
        predictorscore2[i]+=(output==predictors[i])*3
        predictorscore2[i]-=(output==beat[beat[predictors[i]]])*3
        #metapredictor 3
        predictorscore3[i]+=(input==predictors[i])*3
        if input==beat[beat[predictors[i]]]:
            predictorscore3[i]=0
        #metapredictor 4: beat metapredictor 3 (probably contains a bug)
        predictorscore4[i]+=(output==predictors[i])*3
        if output==beat[beat[predictors[i]]]:
            predictorscore4[i]=0
            
    for i in range(number_of_metapredictors):
        metapredictorscore[i]*=0.96
        metapredictorscore[i]+=(input==metapredictors[i])*3
        metapredictorscore[i]-=(input==beat[beat[metapredictors[i]]])*3
        
    
    #Predictors 1-18: History matching
    urmoves+=input		
    mymoves+=output
    DNAmoves+=nuclease[input+output]
    length+=1
    
    for z in range(3):
        limit = min([length,limits[z]])
        j=limit
        while j>=1 and not DNAmoves[length-j:length] in DNAmoves[0:length-1]:
            j-=1
        if j>=1:
            i = DNAmoves.rfind(DNAmoves[length-j:length],0,length-1) 
            predictors[0+6*z] = urmoves[j+i] 
            predictors[1+6*z] = beat[mymoves[j+i]] 
        j=limit			
        while j>=1 and not urmoves[length-j:length] in urmoves[0:length-1]:
            j-=1
        if j>=1:
            i = urmoves.rfind(urmoves[length-j:length],0,length-1) 
            predictors[2+6*z] = urmoves[j+i] 
            predictors[3+6*z] = beat[mymoves[j+i]] 
        j=limit
        while j>=1 and not mymoves[length-j:length] in mymoves[0:length-1]:
            j-=1
        if j>=1:
            i = mymoves.rfind(mymoves[length-j:length],0,length-1) 
            predictors[4+6*z] = urmoves[j+i] 
            predictors[5+6*z] = beat[mymoves[j+i]]
    #Predictor 19,20: RNA Polymerase		
    L=len(mymoves)
    i=DNAmoves.rfind(DNAmoves[L-j:L-1],0,L-2)
    while i==-1:
        j-=1
        i=DNAmoves.rfind(DNAmoves[L-j:L-1],0,L-2)
        if j<2:
            break
    if i==-1 or j+i>=L:
        predictors[18]=predictors[19]=random.choice("RPS")
    else:
        predictors[18]=beat[mymoves[j+i]]
        predictors[19]=urmoves[j+i]

    #Predictors 21-60: rotations of Predictors 1:20
    for i in range(20,60):
        predictors[i]=beat[beat[predictors[i-20]]] #Trying to second guess me?
    
    metapredictors[0]=predictors[predictorscore1.index(max(predictorscore1))]
    metapredictors[1]=beat[predictors[predictorscore2.index(max(predictorscore2))]]
    metapredictors[2]=predictors[predictorscore3.index(max(predictorscore3))]
    metapredictors[3]=beat[predictors[predictorscore4.index(max(predictorscore4))]]
    
    #compare predictors
output = beat[metapredictors[metapredictorscore.index(max(metapredictorscore))]]
if max(metapredictorscore)<0:
    output = beat[random.choice(urmoves)]
"""

"""
IO2_fightinguuu bot by sdfsdf
http://www.rpscontest.com/entry/885001
"""
IO2_FIGHTINGUUU = """
#Iocaine powder based AI

import random

# 2 different lengths of history, 3 kinds of history, both, mine, yours
# 3 different limit length of reverse learning
# 6 kinds of strategy based on Iocaine Powder
num_predictor = 27

if input=="":
    len_rfind = [20]
    limit = [10,20,60]
    beat = { "R":"P" , "P":"S", "S":"R"}
    not_lose = { "R":"PPR" , "P":"SSP" , "S":"RRS" } #50-50 chance
    my_his   =""
    your_his =""
    both_his =""
    list_predictor = [""]*num_predictor
    length = 0
    temp1 = { "PP":"1" , "PR":"2" , "PS":"3",
              "RP":"4" , "RR":"5", "RS":"6",
              "SP":"7" , "SR":"8", "SS":"9"}
    temp2 = { "1":"PP","2":"PR","3":"PS",
                "4":"RP","5":"RR","6":"RS",
                "7":"SP","8":"SR","9":"SS"} 
    who_win = { "PP": 0, "PR":1 , "PS":-1,
                "RP": -1,"RR":0, "RS":1,
                "SP": 1, "SR":-1, "SS":0}
    score_predictor = [0]*num_predictor
    output = random.choice("RPS")
    predictors = [output]*num_predictor
else:
    #update predictors
    #\"\"\"
    if len(list_predictor[0])<5:
        front =0
    else:
        front =1
    for i in range (num_predictor):
        if predictors[i]==input:
            result ="1"
        else:
            result ="0"
        list_predictor[i] = list_predictor[i][front:5]+result #only 5 rounds before
    #history matching 1-6
    my_his += output
    your_his += input
    both_his += temp1[input+output]
    length +=1
    for i in range(1):
        len_size = min(length,len_rfind[i])
        j=len_size
        #both_his
        while j>=1 and not both_his[length-j:length] in both_his[0:length-1]:
            j-=1
        if j>=1:
            k = both_his.rfind(both_his[length-j:length],0,length-1)
            predictors[0+6*i] = your_his[j+k]
            predictors[1+6*i] = beat[my_his[j+k]]
        else:
            predictors[0+6*i] = random.choice("RPS")
            predictors[1+6*i] = random.choice("RPS")
        j=len_size
        #your_his
        while j>=1 and not your_his[length-j:length] in your_his[0:length-1]:
            j-=1
        if j>=1:
            k = your_his.rfind(your_his[length-j:length],0,length-1)
            predictors[2+6*i] = your_his[j+k]
            predictors[3+6*i] = beat[my_his[j+k]]
        else:
            predictors[2+6*i] = random.choice("RPS")
            predictors[3+6*i] = random.choice("RPS")
        j=len_size
        #my_his
        while j>=1 and not my_his[length-j:length] in my_his[0:length-1]:
            j-=1
        if j>=1:
            k = my_his.rfind(my_his[length-j:length],0,length-1)
            predictors[4+6*i] = your_his[j+k]
            predictors[5+6*i] = beat[my_his[j+k]]
        else:
            predictors[4+6*i] = random.choice("RPS")
            predictors[5+6*i] = random.choice("RPS")

    for i in range(3):
        temp =""
        search = temp1[(output+input)] #last round
        for start in range(2, min(limit[i],length) ):
            if search == both_his[length-start]:
                temp+=both_his[length-start+1]
        if(temp==""):
            predictors[6+i] = random.choice("RPS")
        else:
            collectR = {"P":0,"R":0,"S":0} #take win/lose from opponent into account
            for sdf in temp:
                next_move = temp2[sdf]
                if(who_win[next_move]==-1):
                    collectR[temp2[sdf][1]]+=3
                elif(who_win[next_move]==0):
                    collectR[temp2[sdf][1]]+=1
                elif(who_win[next_move]==1):
                    collectR[beat[temp2[sdf][0]]]+=1
            max1 = -1
            p1 =""
            for key in collectR:
                if(collectR[key]>max1):
                    max1 = collectR[key]
                    p1 += key
            predictors[6+i] = random.choice(p1)

    #rotate 9-27:
    for i in range(9,27):
        predictors[i] = beat[beat[predictors[i-9]]]

    #choose a predictor
    len_his = len(list_predictor[0])
    for i in range(num_predictor):
        sum = 0
        for j in range(len_his):
            if list_predictor[i][j]=="1":
                sum+=(j+1)*(j+1)
            else:
                sum-=(j+1)*(j+1)
        score_predictor[i] = sum
    max_score = max(score_predictor)
    #min_score = min(score_predictor)
    #c_temp = {"R":0,"P":0,"S":0}
    #for i in range (num_predictor):
        #if score_predictor[i]==max_score:
        #    c_temp[predictors[i]] +=1
        #if score_predictor[i]==min_score:
        #    c_temp[predictors[i]] -=1
    if max_score>0:
        predict = predictors[score_predictor.index(max_score)]
    else:
        predict = random.choice(your_his)
    output = random.choice(not_lose[predict])
"""

"""
dllu1 bot by dllu
http://www.rpscontest.com/entry/498002
"""
DLLU1 = """
# see also www.dllu.net/rps
# remember, rpsrunner.py is extremely useful for offline testing, 
# here's a screenshot: http://i.imgur.com/DcO9M.png
import random
numPre = 30
numMeta = 6
if not input:
    limit = 8
    beat={'R':'P','P':'S','S':'R'}
    moves=['','','','']
    pScore=[[5]*numPre,[5]*numPre,[5]*numPre,[5]*numPre,[5]*numPre,[5]*numPre]
    centrifuge={'RP':0,'PS':1,'SR':2,'PR':3,'SP':4,'RS':5,'RR':6,'PP':7,'SS':8}
    centripete={'R':0,'P':1,'S':2}
    soma = [0,0,0,0,0,0,0,0,0];
    rps = [1,1,1];
    a="RPS"
    best = [0,0,0];
    length=0
    p=[random.choice("RPS")]*numPre
    m=[random.choice("RPS")]*numMeta
    mScore=[5,2,5,2,4,2]
else:
    for i in range(numPre):
        pp = p[i]
        bpp = beat[pp]
        bbpp = beat[bpp]
        pScore[0][i]=0.9*pScore[0][i]+((input==pp)-(input==bbpp))*3
        pScore[1][i]=0.9*pScore[1][i]+((output==pp)-(output==bbpp))*3
        pScore[2][i]=0.87*pScore[2][i]+(input==pp)*3.3-(input==bpp)*1.2-(input==bbpp)*2.3
        pScore[3][i]=0.87*pScore[3][i]+(output==pp)*3.3-(output==bpp)*1.2-(output==bbpp)*2.3
        pScore[4][i]=(pScore[4][i]+(input==pp)*3)*(1-(input==bbpp))
        pScore[5][i]=(pScore[5][i]+(output==pp)*3)*(1-(output==bbpp))
    for i in range(numMeta):
        mScore[i]=0.96*(mScore[i]+(input==m[i])-(input==beat[beat[m[i]]]))
    soma[centrifuge[input+output]] +=1;
    rps[centripete[input]] +=1;
    moves[0]+=str(centrifuge[input+output])
    moves[1]+=input
    moves[2]+=output
    length+=1
    for y in range(3):
        j=min([length,limit])
        while j>=1 and not moves[y][length-j:length] in moves[y][0:length-1]:
            j-=1
        i = moves[y].rfind(moves[y][length-j:length],0,length-1)
        p[0+2*y] = moves[1][j+i] 
        p[1+2*y] = beat[moves[2][j+i]]
    j=min([length,limit])
    while j>=2 and not moves[0][length-j:length-1] in moves[0][0:length-2]:
        j-=1
    i = moves[0].rfind(moves[0][length-j:length-1],0,length-2)
    if j+i>=length:
        p[6] = p[7] = random.choice("RPS")
    else:
        p[6] = moves[1][j+i] 
        p[7] = beat[moves[2][j+i]]
        
    best[0] = soma[centrifuge[output+'R']]*rps[0]/rps[centripete[output]]
    best[1] = soma[centrifuge[output+'P']]*rps[1]/rps[centripete[output]]
    best[2] = soma[centrifuge[output+'S']]*rps[2]/rps[centripete[output]]
    p[8] = p[9] = a[best.index(max(best))]
    
    for i in range(10,numPre):
        p[i]=beat[beat[p[i-10]]]
        
    for i in range(0,numMeta,2):
        m[i]=       p[pScore[i  ].index(max(pScore[i  ]))]
        m[i+1]=beat[p[pScore[i+1].index(max(pScore[i+1]))]]
output = beat[m[mScore.index(max(mScore))]]
if max(mScore)<0.07 or random.randint(3,40)>length:
    output=beat[random.choice("RPS")]
"""

"""
RPS_Meta_Fix bot by TeleZ
http://www.rpscontest.com/entry/5649874456412160
"""
RPS_META_FIX = """
import random

RNA={'RR':'1','RP':'2','RS':'3','PR':'4','PP':'5','PS':'6','SR':'7','SP':'8','SS':'9'}
mix={'RR':'R','RP':'R','RS':'S','PR':'R','PP':'P','PS':'P','SR':'S','SP':'P','SS':'S'}
rot={'R':'P','P':'S','S':'R'}

if not input:
   DNA=[""]*3
   prin=[random.choice("RPS")]*18
   meta=[random.choice("RPS")]*6
   skor1=[[0]*18,[0]*18,[0]*18,[0]*18,[0]*18,[0]*18]
   skor2=[0]*6
else:
   for j in range(18):
       for i in range(4):
           skor1[i][j]*=0.8
       for i in range(4,6):
           skor1[i][j]*=0.5
       for i in range(0,6,2):
           skor1[i][j]-=(input==rot[rot[prin[j]]])
           skor1[i+1][j]-=(output==rot[rot[prin[j]]])
       for i in range(2,6,2):
           skor1[i][j]+=(input==prin[j])
           skor1[i+1][j]+=(output==prin[j])
       skor1[0][j]+=1.3*(input==prin[j])-0.3*(input==rot[prin[j]])
       skor1[1][j]+=1.3*(output==prin[j])-0.3*(output==rot[prin[j]])
   for i in range(6):
       skor2[i]=0.9*skor2[i]+(input==meta[i])-(input==rot[rot[meta[i]]])
   DNA[0]+=input
   DNA[1]+=output
   DNA[2]+=RNA[input+output]
   for i in range(3):
       j=min(21,len(DNA[2]))
       k=-1
       while j>1 and k<0:
             j-=1
             k=DNA[i].rfind(DNA[i][-j:],0,-1)
       prin[2*i]=DNA[0][j+k]
       prin[2*i+1]=rot[DNA[1][j+k]]
       k=DNA[i].rfind(DNA[i][-j:],0,j+k-1)
       prin[2*i]=mix[prin[2*i]+DNA[0][j+k]]
       prin[2*i+1]=mix[prin[2*i+1]+rot[DNA[1][j+k]]]
   for i in range(6,18):
       prin[i]=rot[prin[i-6]]
   for i in range(0,6,2):
       meta[i]=prin[skor1[i].index(max(skor1[i]))]
       meta[i+1]=rot[prin[skor1[i+1].index(max(skor1[i+1]))]]
output=rot[meta[skor2.index(max(skor2))]]
"""

"""
Are you a lucker? bot by sdfsdf
http://www.rpscontest.com/entry/892001
"""
ARE_YOU_A_LUCKER = """
#This one is just for the proof that luck plays an important role in the leaderboard
#only 200 matches but there are more than 650 ai score > 5000 
import random

num_predictors =27
num_meta= 18

if input =="":
    len_rfind = [20]
    limit = [10,20,60]
    beat = { "P":"S" , "R":"P" , "S":"R" }
    not_lose = { "R":"PR", "P":"SP", "S":"RS" } 
    your_his =""
    my_his = ""
    both_his=""
    both_his2=""
    length =0
    score1=[3]*num_predictors
    score2=[3]*num_predictors
    score3=[3]*num_predictors
    score4=[3]*num_predictors
    score5=[3]*num_predictors
    score6=[3]*num_predictors
    metascore=[3]*num_meta
    temp1 = { "PP":"1","PR":"2","PS":"3",
              "RP":"4","RR":"5","RS":"6",
              "SP":"7","SR":"8","SS":"9"}
    temp2 = { "1":"PP","2":"PR","3":"PS",
                "4":"RP","5":"RR","6":"RS",
                "7":"SP","8":"SR","9":"SS"} 
    who_win = { "PP": 0, "PR":1 , "PS":-1,
                "RP": -1,"RR":0, "RS":1,
                "SP": 1, "SR":-1, "SS":0}
    index = { "P":0, "R":1, "S":2 }
    chance =[0]*num_predictors
    chance2 =[0]*num_predictors
    output = random.choice("RPS")
    predictors = [output]*num_predictors
    metapredictors = [output]*num_meta
else:
    #calculate score
    for i in range(num_predictors):
        #meta 1
        score1[i]*=0.8
        if input==predictors[i]:
            score1[i]+=3
        else:
            score1[i]-=3
        #meta 2
        if input==predictors[i]:
            score2[i]+=3
        else:
            score2[i]=0
        #meta 3
        score3[i]*=0.8
        if output==predictors[i]:
            score3[i]+=3
        else:
           score3[i]-=3
        #meta 4
        if output==predictors[i]:
            score4[i]+=3
        else:
            score4[i]=0
        #meta 5
        score5[i]*=0.8
        if input==predictors[i]:
            score5[i]+=3
        else:
            if chance[i]==1:
                chance[i]=0
                score5[i]-=3
            else:
                chance[i]=1
                score5[i]=0
        #meta 6
        score6[i]*=0.8
        if output==predictors[i]:
            score6[i]+=3
        else:
            if chance2[i]==1:
                chance2[i]=0
                score6[i]-=3
            else:
                chance2[i]=1
                score6[i]=0
    #calculate metascore
    for i in range(num_meta):
        metascore[i]*=0.9
        if input==metapredictors[i]:
            metascore[i]+=3
        else:
            metascore[i]=0
    #Predictors
    #if length>1:
    #    output=beat[predict]
    your_his+=input
    my_his+=output
    both_his+=temp1[(input+output)]
    both_his2+=temp1[(output+input)]
    length+=1

    #history matching
    for i in range(1):
        len_size = min(length,len_rfind[i])
        j=len_size
        #both_his
        while j>=1 and not both_his[length-j:length] in both_his[0:length-1]:
            j-=1
        if j>=1:
            k = both_his.rfind(both_his[length-j:length],0,length-1)
            predictors[0+6*i] = your_his[j+k]
            predictors[1+6*i] = beat[my_his[j+k]]
        else:
            predictors[0+6*i] = random.choice("RPS")
            predictors[1+6*i] = random.choice("RPS")
        j=len_size
        #your_his
        while j>=1 and not your_his[length-j:length] in your_his[0:length-1]:
            j-=1
        if j>=1:
            k = your_his.rfind(your_his[length-j:length],0,length-1)
            predictors[2+6*i] = your_his[j+k]
            predictors[3+6*i] = beat[my_his[j+k]]
        else:
            predictors[2+6*i] = random.choice("RPS")
            predictors[3+6*i] = random.choice("RPS")
        j=len_size
        #my_his
        while j>=1 and not my_his[length-j:length] in my_his[0:length-1]:
            j-=1
        if j>=1:
            k = my_his.rfind(my_his[length-j:length],0,length-1)
            predictors[4+6*i] = your_his[j+k]
            predictors[5+6*i] = beat[my_his[j+k]]
        else:
            predictors[4+6*i] = random.choice("RPS")
            predictors[5+6*i] = random.choice("RPS")
    
    #Reverse
    for i in range(3):
        temp =""
        search = temp1[(output+input)] #last round
        for start in range(2, min(limit[i],length) ):
            if search == both_his2[length-start]:
                temp+=both_his2[length-start+1]
        if(temp==""):
            predictors[6+i] = random.choice("RPS")
        else:
            collectR = {"P":0,"R":0,"S":0} #take win/lose from opponent into account
            for sdf in temp:
                next_move = temp2[sdf]
                if(who_win[next_move]==-1):
                    collectR[temp2[sdf][1]]+=3
                elif(who_win[next_move]==0):
                    collectR[temp2[sdf][1]]+=1
                elif(who_win[next_move]==1):
                    collectR[beat[temp2[sdf][0]]]+=1
            max1 = -1
            p1 =""
            for key in collectR:
                if(collectR[key]>max1):
                    max1 = collectR[key]
                    p1 += key
            predictors[6+i] = random.choice(p1)
    
    for i in range(9,27):
        predictors[i]=beat[beat[predictors[i-9]]]
    #find prediction for each meta
    metapredictors[0]=predictors[score1.index(max(score1))]
    metapredictors[1]=predictors[score2.index(max(score2))]
    metapredictors[2]=beat[predictors[score3.index(max(score3))]]
    metapredictors[3]=beat[predictors[score4.index(max(score4))]]
    metapredictors[4]=predictors[score5.index(max(score5))]
    metapredictors[5]=beat[predictors[score6.index(max(score6))]]
    for i in range(6,18):
        metapredictors[i] = beat[metapredictors[i-6]]
    
    predict = metapredictors[metascore.index(max(metascore))]
    output = beat[predict]
    #output = random.choice(not_lose[predict])
"""

"""
bayes14 bot by pyfex
http://www.rpscontest.com/entry/202003
"""
BAYES_14 = """
# See http://overview.cc/RockPaperScissors for more information about rock, paper, scissors
# Extension to bayes13: Use also the csc function for singleopp and singlemy

from collections import defaultdict
import operator
import random

if input == "":
  score = {'RR': 0, 'PP': 0, 'SS': 0, 'PR': 1, 'RS': 1, 'SP': 1,'RP': -1, 'SR': -1, 'PS': -1,}
  cscore = {'RR': 'r', 'PP': 'r', 'SS': 'r', 'PR': 'b', 'RS': 'b', 'SP': 'b','RP': 'c', 'SR': 'c', 'PS': 'c',}
  beat = {'P': 'S', 'S': 'R', 'R': 'P'}
  cede = {'P': 'R', 'S': 'P', 'R': 'S'}
  rps = ['R', 'P', 'S']
  
  def counter_prob(probs):
    weighted_list = []
    for h in ['R', 'P', 'S']:
      weighted = 0
      for p in probs.keys():
        points = score[h+p]
        prob = probs[p]
        weighted += points * prob
      weighted_list.append((h, weighted))

    return max(weighted_list, key=operator.itemgetter(1))[0]

  played_probs = defaultdict(lambda: 1)
  opp_probs = defaultdict(lambda: defaultdict(lambda: 1))
  my_probs = defaultdict(lambda: defaultdict(lambda: 1))
  both_probs = defaultdict(lambda: defaultdict(lambda: 1))

  singleopp_opp_probs = defaultdict(lambda: defaultdict(lambda: 1))
  singleopp_my_probs = defaultdict(lambda: defaultdict(lambda: 1))
  singleopp_both_probs = defaultdict(lambda: defaultdict(lambda: 1))

  singlemy_opp_probs = defaultdict(lambda: defaultdict(lambda: 1))
  singlemy_my_probs = defaultdict(lambda: defaultdict(lambda: 1))
  singlemy_both_probs = defaultdict(lambda: defaultdict(lambda: 1))

  opp2_probs = defaultdict(lambda: defaultdict(lambda: 1))
  my2_probs = defaultdict(lambda: defaultdict(lambda: 1))
  both2_probs = defaultdict(lambda: defaultdict(lambda: 1))

  singleopp_opp2_probs = defaultdict(lambda: defaultdict(lambda: 1))
  singleopp_my2_probs = defaultdict(lambda: defaultdict(lambda: 1))
  singleopp_both2_probs = defaultdict(lambda: defaultdict(lambda: 1))

  singlemy_opp2_probs = defaultdict(lambda: defaultdict(lambda: 1))
  singlemy_my2_probs = defaultdict(lambda: defaultdict(lambda: 1))
  singlemy_both2_probs = defaultdict(lambda: defaultdict(lambda: 1))

  win_probs = defaultdict(lambda: 1)
  lose_probs = defaultdict(lambda: 1)
  tie_probs = defaultdict(lambda: 1)

  singleopp_win_probs = defaultdict(lambda: 1)
  singleopp_lose_probs = defaultdict(lambda: 1)
  singleopp_tie_probs = defaultdict(lambda: 1)

  singlemy_win_probs = defaultdict(lambda: 1)
  singlemy_lose_probs = defaultdict(lambda: 1)
  singlemy_tie_probs = defaultdict(lambda: 1)

  opp_answers = {'c': 1, 'b': 1, 'r': 1}
  my_answers = {'c': 1, 'b': 1, 'r': 1}

  opp2_answers = {'c': 1, 'b': 1, 'r': 1}
  my2_answers = {'c': 1, 'b': 1, 'r': 1}  

  singleopp_opp_answers = {'c': 1, 'b': 1, 'r': 1}
  singleopp_my_answers = {'c': 1, 'b': 1, 'r': 1}

  singleopp_opp2_answers = {'c': 1, 'b': 1, 'r': 1}
  singleopp_my2_answers = {'c': 1, 'b': 1, 'r': 1}  

  singlemy_opp_answers = {'c': 1, 'b': 1, 'r': 1}
  singlemy_my_answers = {'c': 1, 'b': 1, 'r': 1}

  singlemy_opp2_answers = {'c': 1, 'b': 1, 'r': 1}
  singlemy_my2_answers = {'c': 1, 'b': 1, 'r': 1}

  patterndict = defaultdict(str)
  patterndict2 = defaultdict(str)
  opppatterndict = defaultdict(str)
  opppatterndict2 = defaultdict(str)
  mypatterndict = defaultdict(str)
  mypatterndict2 = defaultdict(str)

  csu = [0] * 6 # consecutive strategy usage
  csc = []  # consecutive strategy candidates
  singleopp_csu = [0] * 6 # consecutive strategy usage
  singleopp_csc = []  # consecutive strategy candidates
  singlemy_csu = [0] * 6 # consecutive strategy usage
  singlemy_csc = []  # consecutive strategy candidates

  output = random.choice(["R", "P", "S"])
  hist = "" 
  myhist = "" 
  opphist = "" 

  my = opp = my2 = opp2 =  ""
  singleopp_my = singleopp_opp = singleopp_my2 = singleopp_opp2 = ""
  singlemy_my = singlemy_opp = singlemy_my2 = singlemy_opp2 = ""

  sc = 0
  opp_strats = []
  singleopp_oppstrats = []
  singlemy_oppstrats = []
else:
  previous_opp_strats = opp_strats[:]
  previous_singleopp_oppstrats = singleopp_oppstrats[:]
  previous_singlemy_oppstrats = singlemy_oppstrats[:]
  previous_sc = sc

  sc = score[output + input]
  for i, c in enumerate(csc):
    if c == input:
      csu[i] += 1
    else:
      csu[i] = 0

  for i, c in enumerate(singleopp_csc):
    if c == input:
      singleopp_csu[i] += 1
    else:
      singleopp_csu[i] = 0

  for i, c in enumerate(singlemy_csc):
    if c == input:
      singlemy_csu[i] += 1
    else:
      singlemy_csu[i] = 0

  m = max(csu)
  opp_strats = [i for i, c in enumerate(csc) if csu[i] == m]

  m = max(singleopp_csu)
  singleopp_oppstrats = [i for i, c in enumerate(singleopp_csc) if singleopp_csu[i] == m]

  m = max(csu)
  singlemy_oppstrats = [i for i, c in enumerate(singlemy_csc) if singlemy_csu[i] == m]
  
  if previous_sc == 1:
    for s1 in previous_opp_strats:
      for s2 in opp_strats:
        win_probs[chr(s1)+chr(s2)] += 1

    for s1 in previous_singleopp_oppstrats:
      for s2 in singleopp_oppstrats:
        singleopp_win_probs[chr(s1)+chr(s2)] += 1

    for s1 in previous_singlemy_oppstrats:
      for s2 in singlemy_oppstrats:
        singlemy_win_probs[chr(s1)+chr(s2)] += 1
  
  if previous_sc == 0:
    for s1 in previous_opp_strats:
      for s2 in opp_strats:
        tie_probs[chr(s1)+chr(s2)] += 1

    for s1 in previous_singleopp_oppstrats:
      for s2 in singleopp_oppstrats:
        singleopp_tie_probs[chr(s1)+chr(s2)] += 1

    for s1 in previous_singlemy_oppstrats:
      for s2 in singlemy_oppstrats:
        singlemy_tie_probs[chr(s1)+chr(s2)] += 1

  if previous_sc == -1:
    for s1 in previous_opp_strats:
      for s2 in opp_strats:
        lose_probs[chr(s1)+chr(s2)] += 1
    for s1 in previous_singleopp_oppstrats:
      for s2 in singleopp_oppstrats:
        singleopp_lose_probs[chr(s1)+chr(s2)] += 1
    for s1 in previous_singlemy_oppstrats:
      for s2 in singlemy_oppstrats:
        singlemy_lose_probs[chr(s1)+chr(s2)] += 1

  if my and opp:
    opp_answers[cscore[input+opp]] += 1
    my_answers[cscore[input+my]] += 1
  if my2 and opp2:
    opp2_answers[cscore[input+opp2]] += 1
    my2_answers[cscore[input+my2]] += 1

  if singleopp_my and singleopp_opp:
    singleopp_opp_answers[cscore[input+singleopp_opp]] += 1
    singleopp_my_answers[cscore[input+singleopp_my]] += 1
  if singleopp_my2 and singleopp_opp2:
    singleopp_opp2_answers[cscore[input+singleopp_opp2]] += 1
    singleopp_my2_answers[cscore[input+singleopp_my2]] += 1

  if singlemy_my and singlemy_opp:
    singlemy_opp_answers[cscore[input+singlemy_opp]] += 1
    singlemy_my_answers[cscore[input+singlemy_my]] += 1
  if singlemy_my2 and singlemy_opp2:
    singlemy_opp2_answers[cscore[input+singlemy_opp2]] += 1
    singlemy_my2_answers[cscore[input+singlemy_my2]] += 1

  for length in range(min(10, len(hist)), 0, -2):
    pattern = patterndict[hist[-length:]]
    if pattern:
      for length2 in range(min(10, len(pattern)), 0, -2):
        patterndict2[pattern[-length2:]] += output + input
    patterndict[hist[-length:]] += output + input

  # singleopp
  for length in range(min(5, len(opphist)), 0, -1):
    pattern = opppatterndict[opphist[-length:]]
    if pattern:
      for length2 in range(min(10, len(pattern)), 0, -2):
        opppatterndict2[pattern[-length2:]] += output + input
    opppatterndict[opphist[-length:]] += output + input

  # singlemy
  for length in range(min(5, len(myhist)), 0, -1):
    pattern = mypatterndict[myhist[-length:]]
    if pattern:
      for length2 in range(min(10, len(pattern)), 0, -2):
        mypatterndict2[pattern[-length2:]] += output + input
    mypatterndict[myhist[-length:]] += output + input

  played_probs[input] += 1
  opp_probs[opp][input] += 1
  my_probs[my][input] += 1
  both_probs[my+opp][input] += 1

  opp2_probs[opp2][input] += 1
  my2_probs[my2][input] += 1
  both2_probs[my2+opp2][input] += 1

  hist += output + input
  myhist += output
  opphist += input

  my = opp = my2 = opp2 = ""
  singleopp_my = singleopp_opp = singleopp_my2 = singleopp_opp2 = ""
  singlemy_my = singlemy_opp = singlemy_my2 = singlemy_opp2 = ""

  for length in range(min(10, len(hist)), 0, -2):
    pattern = patterndict[hist[-length:]]
    if pattern != "":
      my = pattern[-2]
      opp = pattern[-1]
      for length2 in range(min(10, len(pattern)), 0, -2):
        pattern2 = patterndict2[pattern[-length2:]]
        if pattern2 != "":
          my2 = pattern2[-2]
          opp2 = pattern2[-1]
          break
      break

  # singleopp
  for length in range(min(5, len(opphist)), 0, -1):
    pattern = opppatterndict[opphist[-length:]]
    if pattern != "":
      singleopp_my = pattern[-2]
      singleopp_opp = pattern[-1]
      for length2 in range(min(10, len(pattern)), 0, -2):
        pattern2 = opppatterndict2[pattern[-length2:]]
        if pattern2 != "":
          singleopp_my2 = pattern2[-2]
          singleopp_opp2 = pattern2[-1]
          break
      break

  # singlemy
  for length in range(min(5, len(myhist)), 0, -1):
    pattern = mypatterndict[myhist[-length:]]
    if pattern != "":
      singlemy_my = pattern[-2]
      singlemy_opp = pattern[-1]
      for length2 in range(min(10, len(pattern)), 0, -2):
        pattern2 = mypatterndict2[pattern[-length2:]]
        if pattern2 != "":
          singlemy_my2 = pattern2[-2]
          singlemy_opp2 = pattern2[-1]
          break
      break

  probs = {}
  for hand in rps:
    probs[hand] = played_probs[hand]
        
  if my and opp:
    for hand in rps:
      probs[hand] *= opp_probs[opp][hand] * my_probs[my][hand] * both_probs[my+opp][hand]
      probs[hand] *= opp_answers[cscore[hand+opp]] * my_answers[cscore[hand+my]]


    csc = [opp, beat[opp], cede[opp], my, cede[my], beat[my]]
  
    strats_for_hand = {'R': [], 'P': [], 'S': []}
    for i, c in enumerate(csc):
      strats_for_hand[c].append(i)

    if sc == 1:
      pr = win_probs
    if sc == 0:
      pr = tie_probs
    if sc == -1:
      pr = lose_probs

    for hand in rps:
      for s1 in opp_strats:
        for s2 in strats_for_hand[hand]:
          probs[hand] *= pr[chr(s1)+chr(s2)]
  else:
    csc = []

  if singleopp_my and singleopp_opp:
    for hand in rps:
      probs[hand] *= singleopp_opp_probs[singleopp_opp][hand] * \
                     singleopp_my_probs[singleopp_my][hand] * \
                     singleopp_both_probs[singleopp_my+singleopp_opp][hand]
      probs[hand] *= singleopp_opp_answers[cscore[hand+singleopp_opp]] * singleopp_my_answers[cscore[hand+singleopp_my]]

    singleopp_csc = [singleopp_opp, beat[singleopp_opp], cede[singleopp_opp], singleopp_my, cede[singleopp_my], beat[singleopp_my]]
  
    strats_for_hand = {'R': [], 'P': [], 'S': []}
    for i, c in enumerate(singleopp_csc):
      strats_for_hand[c].append(i)

    if sc == 1:
      pr = singleopp_win_probs
    if sc == 0:
      pr = singleopp_tie_probs
    if sc == -1:
      pr = singleopp_lose_probs

    for hand in rps:
      for s1 in singleopp_oppstrats:
        for s2 in strats_for_hand[hand]:
          probs[hand] *= pr[chr(s1)+chr(s2)]
  else:
    singleopp_csc = []

  if singlemy_my and singlemy_opp:
    for hand in rps:
      probs[hand] *= singlemy_opp_probs[singlemy_opp][hand] * \
                     singlemy_my_probs[singlemy_my][hand] * \
                     singlemy_both_probs[singlemy_my+singlemy_opp][hand]
      probs[hand] *= singlemy_opp_answers[cscore[hand+singlemy_opp]] * singlemy_my_answers[cscore[hand+singlemy_my]]

    singlemy_csc = [singlemy_opp, beat[singlemy_opp], cede[singlemy_opp], singlemy_my, cede[singlemy_my], beat[singlemy_my]]
  
    strats_for_hand = {'R': [], 'P': [], 'S': []}
    for i, c in enumerate(singlemy_csc):
      strats_for_hand[c].append(i)

    if sc == 1:
      pr = singlemy_win_probs
    if sc == 0:
      pr = singlemy_tie_probs
    if sc == -1:
      pr = singlemy_lose_probs

    for hand in rps:
      for s1 in singlemy_oppstrats:
        for s2 in strats_for_hand[hand]:
          probs[hand] *= pr[chr(s1)+chr(s2)]
  else:
    singlemy_csc = []
                
  if my2 and opp2:
    for hand in rps:
      probs[hand] *= opp2_probs[opp2][hand] * my2_probs[my2][hand] * both2_probs[my2+opp2][hand]
      probs[hand] *= opp2_answers[cscore[hand+opp2]] * my2_answers[cscore[hand+my2]]

  if singleopp_my2 and singleopp_opp2:
    for hand in rps:
      probs[hand] *= singleopp_opp2_probs[singleopp_opp2][hand] *\
                     singleopp_my2_probs[singleopp_my2][hand] *\
                     singleopp_both2_probs[singleopp_my2+singleopp_opp2][hand]
      probs[hand] *= singleopp_opp2_answers[cscore[hand+singleopp_opp2]] * \
                     singleopp_my2_answers[cscore[hand+singleopp_my2]]

  if singlemy_my2 and singlemy_opp2:
    for hand in rps:
      probs[hand] *= singlemy_opp2_probs[singlemy_opp2][hand] *\
                     singlemy_my2_probs[singlemy_my2][hand] *\
                     singlemy_both2_probs[singlemy_my2+singlemy_opp2][hand]
      probs[hand] *= singlemy_opp2_answers[cscore[hand+singlemy_opp2]] * \
                     singlemy_my2_answers[cscore[hand+singlemy_my2]]

  output = counter_prob(probs)
"""


SIGNS = 3
EQUAL_PROBS = np.array([1, 1, 1], dtype=np.float) / 3
EQUAL_PROBS.flags.writeable = False
SEQUENCES = {
    'de_bruijn': [2, 2, 2, 2, 2, 1, 2, 2, 0, 2, 2, 2, 2, 2, 1, 2, 2, 0, 2, 1, 1, 2, 1, 0, 2, 1, 2, 2, 1, 1, 2, 1, 0, 2, 0, 0, 2, 0, 2, 2, 0, 1, 2, 0, 0, 2, 2, 2, 2, 2, 1, 2, 2, 0, 2, 1, 1, 2, 1, 0, 2, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 2, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 2, 1, 0, 1, 1, 0, 0, 1, 2, 2, 1, 2, 1, 1, 2, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 2, 2, 0, 2, 1, 0, 2, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 2, 2, 2, 2, 2, 1, 2, 2, 0, 2, 1, 1, 2, 1, 0, 2, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 1, 2, 2, 0, 2, 2, 2, 2, 2, 1, 2, 2, 0, 2, 1, 1, 2, 1, 0, 2, 1, 2, 2, 1, 1, 2, 1, 0, 2, 0, 0, 2, 0, 2, 2, 0, 1, 2, 0, 0, 2, 2, 2, 2, 2, 1, 2, 2, 0, 2, 1, 1, 2, 1, 0, 2, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 2, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 2, 1, 0, 1, 1, 0, 0, 1, 2, 2, 1, 2, 1, 1, 2, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 2, 2, 0, 2, 1, 0, 2, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 2, 2, 2, 2, 2, 1, 2, 2, 0, 2, 1, 1, 2, 1, 0, 2, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 1, 2, 2, 0, 2, 2, 2, 2, 2, 1, 2, 2, 0, 2, 1, 1, 2, 1, 0, 2, 1, 2, 2, 1, 1, 2, 1, 0, 2, 0, 0, 2, 0, 2, 2, 0, 1, 2, 0, 0, 2, 2, 2, 2, 2, 1, 2, 2, 0, 2, 1, 1, 2, 1, 0, 2, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 2, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 2, 1, 0, 1, 1, 0, 0, 1, 2, 2, 1, 2, 1, 1, 2, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 2, 2, 0, 2, 1, 0, 2, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 2, 2, 2, 2, 2, 1, 2, 2, 0, 2, 1, 1, 2, 1, 0, 2, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 1, 2, 2, 0, 2, 2, 2, 2, 2, 1, 2, 2, 0, 2, 1, 1, 2, 1, 0, 2, 1, 2, 2, 1, 1, 2, 1, 0, 2, 0, 0, 2, 0, 2, 2, 0, 1, 2, 0, 0, 2, 2, 2, 2, 2, 1, 2, 2, 0, 2, 1, 1, 2, 1, 0, 2, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 2, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 2, 1, 0, 1, 1, 0, 0, 1, 2, 2, 1, 2, 1, 1, 2, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 2, 2, 0, 2, 1, 0, 2, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 2, 2, 2, 2, 2, 1, 2, 2, 0, 2, 1, 1, 2, 1, 0, 2, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 1, 2, 2, 0, 2, 2, 2, 2, 2, 1, 2, 2, 0, 2, 1, 1, 2, 1, 0, 2, 1, 2, 2, 1, 1, 2, 1, 0, 2, 0, 0, 2, 0, 2, 2, 0, 1, 2, 0, 0, 2, 2, 2, 2, 2, 1, 2, 2, 0, 2, 1, 1, 2, 1, 0, 2, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 2, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 2, 1, 0, 1, 1, 0, 0, 1, 2, 2, 1, 2, 1, 1, 2, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 2, 2, 0, 2, 1, 0, 2, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 2, 2, 2, 2, 2, 1, 2, 2, 0, 2, 1, 1, 2, 1, 0, 2, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 1, 2, 2, 0, 2, 2, 2, 2, 2, 1, 2, 2, 0, 2, 1, 1, 2, 1, 0, 2, 1, 2, 2, 1, 1, 2, 1, 0, 2, 0, 0, 2, 0, 2, 2, 0, 1, 2, 0, 0, 2, 2, 2, 2, 2, 1, 2, 2, 0, 2, 1, 1, 2, 1, 0, 2, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 2, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 2, 1, 0, 1, 1, 0, 0, 1, 2, 2, 1, 2, 1, 1, 2, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 2, 2, 0, 2, 1, 0, 2, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 2, 2, 2, 2, 2, 1, 2, 2, 0, 2, 1, 1, 2, 1, 0, 2, 0, 0, 1, 1, 1, 1],
    'pi': [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 2, 3, 8, 4, 6, 2, 6, 4, 3, 3, 8, 3, 2, 7, 9, 5, 2, 8, 8, 4, 1, 9, 7, 1, 6, 9, 3, 9, 9, 3, 7, 5, 1, 5, 8, 2, 9, 7, 4, 9, 4, 4, 5, 9, 2, 3, 7, 8, 1, 6, 4, 6, 2, 8, 6, 2, 8, 9, 9, 8, 6, 2, 8, 3, 4, 8, 2, 5, 3, 4, 2, 1, 1, 7, 6, 7, 9, 8, 2, 1, 4, 8, 8, 6, 5, 1, 3, 2, 8, 2, 3, 6, 6, 4, 7, 9, 3, 8, 4, 4, 6, 9, 5, 5, 5, 8, 2, 2, 3, 1, 7, 2, 5, 3, 5, 9, 4, 8, 1, 2, 8, 4, 8, 1, 1, 1, 7, 4, 5, 2, 8, 4, 1, 2, 7, 1, 9, 3, 8, 5, 2, 1, 1, 5, 5, 5, 9, 6, 4, 4, 6, 2, 2, 9, 4, 8, 9, 5, 4, 9, 3, 3, 8, 1, 9, 6, 4, 4, 2, 8, 8, 1, 9, 7, 5, 6, 6, 5, 9, 3, 3, 4, 4, 6, 1, 2, 8, 4, 7, 5, 6, 4, 8, 2, 3, 3, 7, 8, 6, 7, 8, 3, 1, 6, 5, 2, 7, 1, 2, 1, 9, 9, 1, 4, 5, 6, 4, 8, 5, 6, 6, 9, 2, 3, 4, 6, 3, 4, 8, 6, 1, 4, 5, 4, 3, 2, 6, 6, 4, 8, 2, 1, 3, 3, 9, 3, 6, 7, 2, 6, 2, 4, 9, 1, 4, 1, 2, 7, 3, 7, 2, 4, 5, 8, 7, 6, 6, 6, 3, 1, 5, 5, 8, 8, 1, 7, 4, 8, 8, 1, 5, 2, 9, 2, 9, 6, 2, 8, 2, 9, 2, 5, 4, 9, 1, 7, 1, 5, 3, 6, 4, 3, 6, 7, 8, 9, 2, 5, 9, 3, 6, 1, 1, 3, 3, 5, 3, 5, 4, 8, 8, 2, 4, 6, 6, 5, 2, 1, 3, 8, 4, 1, 4, 6, 9, 5, 1, 9, 4, 1, 5, 1, 1, 6, 9, 4, 3, 3, 5, 7, 2, 7, 3, 6, 5, 7, 5, 9, 5, 9, 1, 9, 5, 3, 9, 2, 1, 8, 6, 1, 1, 7, 3, 8, 1, 9, 3, 2, 6, 1, 1, 7, 9, 3, 1, 5, 1, 1, 8, 5, 4, 8, 7, 4, 4, 6, 2, 3, 7, 9, 9, 6, 2, 7, 4, 9, 5, 6, 7, 3, 5, 1, 8, 8, 5, 7, 5, 2, 7, 2, 4, 8, 9, 1, 2, 2, 7, 9, 3, 8, 1, 8, 3, 1, 1, 9, 4, 9, 1, 2, 9, 8, 3, 3, 6, 7, 3, 3, 6, 2, 4, 4, 6, 5, 6, 6, 4, 3, 8, 6, 2, 1, 3, 9, 4, 9, 4, 6, 3, 9, 5, 2, 2, 4, 7, 3, 7, 1, 9, 7, 2, 1, 7, 9, 8, 6, 9, 4, 3, 7, 2, 7, 7, 5, 3, 9, 2, 1, 7, 1, 7, 6, 2, 9, 3, 1, 7, 6, 7, 5, 2, 3, 8, 4, 6, 7, 4, 8, 1, 8, 4, 6, 7, 6, 6, 9, 4, 5, 1, 3, 2, 5, 6, 8, 1, 2, 7, 1, 4, 5, 2, 6, 3, 5, 6, 8, 2, 7, 7, 8, 5, 7, 7, 1, 3, 4, 2, 7, 5, 7, 7, 8, 9, 6, 9, 1, 7, 3, 6, 3, 7, 1, 7, 8, 7, 2, 1, 4, 6, 8, 4, 4, 9, 1, 2, 2, 4, 9, 5, 3, 4, 3, 1, 4, 6, 5, 4, 9, 5, 8, 5, 3, 7, 1, 5, 7, 9, 2, 2, 7, 9, 6, 8, 9, 2, 5, 8, 9, 2, 3, 5, 4, 2, 1, 9, 9, 5, 6, 1, 1, 2, 1, 2, 9, 2, 1, 9, 6, 8, 6, 4, 3, 4, 4, 1, 8, 1, 5, 9, 8, 1, 3, 6, 2, 9, 7, 7, 4, 7, 7, 1, 3, 9, 9, 6, 5, 1, 8, 7, 7, 2, 1, 1, 3, 4, 9, 9, 9, 9, 9, 9, 8, 3, 7, 2, 9, 7, 8, 4, 9, 9, 5, 1, 5, 9, 7, 3, 1, 7, 3, 2, 8, 1, 6, 9, 6, 3, 1, 8, 5, 9, 5, 2, 4, 4, 5, 9, 4, 5, 5, 3, 4, 6, 9, 8, 3, 2, 6, 4, 2, 5, 2, 2, 3, 8, 2, 5, 3, 3, 4, 4, 6, 8, 5, 3, 5, 2, 6, 1, 9, 3, 1, 1, 8, 8, 1, 7, 1, 1, 3, 1, 3, 7, 8, 3, 8, 7, 5, 2, 8, 8, 6, 5, 8, 7, 5, 3, 3, 2, 8, 3, 8, 1, 4, 2, 6, 1, 7, 1, 7, 7, 6, 6, 9, 1, 4, 7, 3, 3, 5, 9, 8, 2, 5, 3, 4, 9, 4, 2, 8, 7, 5, 5, 4, 6, 8, 7, 3, 1, 1, 5, 9, 5, 6, 2, 8, 6, 3, 8, 8, 2, 3, 5, 3, 7, 8, 7, 5, 9, 3, 7, 5, 1, 9, 5, 7, 7, 8, 1, 8, 5, 7, 7, 8, 5, 3, 2, 1, 7, 1, 2, 2, 6, 8, 6, 6, 1, 3, 1, 9, 2, 7, 8, 7, 6, 6, 1, 1, 1, 9, 5, 9, 9, 2, 1, 6, 4, 2, 1, 9, 8, 9, 3, 8, 9, 5, 2, 5, 7, 2, 1, 6, 5, 4, 8, 5, 8, 6, 3, 2, 7, 8, 8, 6, 5, 9, 3, 6, 1, 5, 3, 3, 8, 1, 8, 2, 7, 9, 6, 8, 2, 3, 3, 1, 9, 5, 2, 3, 5, 3, 1, 8, 5, 2, 9, 6, 8, 9, 9, 5, 7, 7, 3, 6, 2, 2, 5, 9, 9, 4, 1, 3, 8, 9, 1, 2, 4, 9, 7, 2, 1, 7, 7, 5, 2, 8, 3, 4, 7, 9, 1, 3, 1, 5],
    'e': [2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 5, 9, 4, 5, 2, 3, 5, 3, 6, 2, 8, 7, 4, 7, 1, 3, 5, 2, 6, 6, 2, 4, 9, 7, 7, 5, 7, 2, 4, 7, 9, 3, 6, 9, 9, 9, 5, 9, 5, 7, 4, 9, 6, 6, 9, 6, 7, 6, 2, 7, 7, 2, 4, 7, 6, 6, 3, 3, 5, 3, 5, 4, 7, 5, 9, 4, 5, 7, 1, 3, 8, 2, 1, 7, 8, 5, 2, 5, 1, 6, 6, 4, 2, 7, 4, 2, 7, 4, 6, 6, 3, 9, 1, 9, 3, 2, 3, 5, 9, 9, 2, 1, 8, 1, 7, 4, 1, 3, 5, 9, 6, 6, 2, 9, 4, 3, 5, 7, 2, 9, 3, 3, 4, 2, 9, 5, 2, 6, 5, 9, 5, 6, 3, 7, 3, 8, 1, 3, 2, 3, 2, 8, 6, 2, 7, 9, 4, 3, 4, 9, 7, 6, 3, 2, 3, 3, 8, 2, 9, 8, 8, 7, 5, 3, 1, 9, 5, 2, 5, 1, 1, 9, 1, 1, 5, 7, 3, 8, 3, 4, 1, 8, 7, 9, 3, 7, 2, 1, 5, 4, 8, 9, 1, 4, 9, 9, 3, 4, 8, 8, 4, 1, 6, 7, 5, 9, 2, 4, 4, 7, 6, 1, 4, 6, 6, 6, 8, 8, 2, 2, 6, 4, 8, 1, 6, 8, 4, 7, 7, 4, 1, 1, 8, 5, 3, 7, 4, 2, 3, 4, 5, 4, 4, 2, 4, 3, 7, 1, 7, 5, 3, 9, 7, 7, 7, 4, 4, 9, 9, 2, 6, 9, 5, 5, 1, 7, 2, 7, 6, 1, 8, 3, 8, 6, 6, 2, 6, 1, 3, 3, 1, 3, 8, 4, 5, 8, 3, 7, 5, 2, 4, 4, 9, 3, 3, 8, 2, 6, 5, 6, 2, 9, 7, 6, 6, 7, 3, 7, 1, 1, 3, 2, 7, 9, 3, 2, 8, 7, 9, 1, 2, 7, 4, 4, 3, 7, 4, 7, 4, 7, 2, 3, 6, 9, 6, 9, 7, 7, 2, 9, 3, 1, 1, 4, 1, 6, 9, 2, 8, 3, 6, 8, 1, 9, 2, 5, 5, 1, 5, 1, 8, 6, 5, 7, 4, 6, 3, 7, 7, 2, 1, 1, 1, 2, 5, 2, 3, 8, 9, 7, 8, 4, 4, 2, 5, 5, 6, 9, 5, 3, 6, 9, 6, 7, 7, 7, 8, 5, 4, 4, 9, 9, 6, 9, 9, 6, 7, 9, 4, 6, 8, 6, 4, 4, 5, 4, 9, 5, 9, 8, 7, 9, 3, 1, 6, 3, 6, 8, 8, 9, 2, 3, 9, 8, 7, 9, 3, 1, 2, 7, 7, 3, 6, 1, 7, 8, 2, 1, 5, 4, 2, 4, 9, 9, 9, 2, 2, 9, 5, 7, 6, 3, 5, 1, 4, 8, 2, 2, 8, 2, 6, 9, 8, 9, 5, 1, 9, 3, 6, 6, 8, 3, 3, 1, 8, 2, 5, 2, 8, 8, 6, 9, 3, 9, 8, 4, 9, 6, 4, 6, 5, 1, 5, 8, 2, 9, 3, 9, 2, 3, 9, 8, 2, 9, 4, 8, 8, 7, 9, 3, 3, 2, 3, 6, 2, 5, 9, 4, 4, 3, 1, 1, 7, 3, 1, 2, 3, 8, 1, 9, 7, 6, 8, 4, 1, 6, 1, 4, 3, 9, 7, 1, 9, 8, 3, 7, 6, 7, 9, 3, 2, 6, 8, 3, 2, 8, 2, 3, 7, 6, 4, 6, 4, 8, 4, 2, 9, 5, 3, 1, 1, 8, 2, 3, 2, 8, 7, 8, 2, 5, 9, 8, 1, 9, 4, 5, 5, 8, 1, 5, 3, 1, 7, 5, 6, 7, 1, 7, 3, 6, 1, 3, 3, 2, 6, 9, 8, 1, 1, 2, 5, 9, 9, 6, 1, 8, 1, 8, 8, 1, 5, 9, 3, 4, 1, 6, 9, 3, 5, 1, 5, 9, 8, 8, 8, 8, 5, 1, 9, 3, 4, 5, 8, 7, 2, 7, 3, 8, 6, 6, 7, 3, 8, 5, 8, 9, 4, 2, 2, 8, 7, 9, 2, 2, 8, 4, 9, 9, 8, 9, 2, 8, 6, 8, 5, 8, 2, 5, 7, 4, 9, 2, 7, 9, 6, 1, 4, 8, 4, 1, 9, 8, 4, 4, 4, 3, 6, 3, 4, 6, 3, 2, 4, 4, 9, 6, 8, 4, 8, 7, 5, 6, 2, 3, 3, 6, 2, 4, 8, 2, 7, 4, 1, 9, 7, 8, 6, 2, 3, 2, 9, 2, 1, 6, 9, 9, 2, 3, 5, 3, 4, 3, 6, 9, 9, 4, 1, 8, 4, 9, 1, 4, 6, 3, 1, 4, 9, 3, 4, 3, 1, 7, 3, 8, 1, 4, 3, 6, 4, 5, 4, 6, 2, 5, 3, 1, 5, 2, 9, 6, 1, 8, 3, 6, 9, 8, 8, 8, 7, 7, 1, 6, 7, 6, 8, 3, 9, 6, 4, 2, 4, 3, 7, 8, 1, 4, 5, 9, 2, 7, 1, 4, 5, 6, 3, 5, 4, 9, 6, 1, 3, 3, 1, 7, 2, 8, 5, 1, 3, 8, 3, 7, 5, 5, 1, 1, 1, 5, 7, 4, 7, 7, 4, 1, 7, 1, 8, 9, 8, 6, 1, 6, 8, 7, 3, 9, 6, 9, 6, 5, 5, 2, 1, 2, 6, 7, 1, 5, 4, 6, 8, 8, 9, 5, 7, 3, 5, 3, 5, 4, 2, 1, 2, 3, 4, 7, 8, 4, 9, 8, 1, 9, 3, 3, 4, 3, 2, 1, 6, 8, 1, 7, 1, 2, 1, 5, 6, 2, 7, 8, 8, 2, 3, 5, 1, 9, 3, 3, 3, 2, 2, 4, 7, 4, 5, 1, 5, 8, 5, 3, 9, 4, 7, 3, 4, 1, 9, 9, 5, 7, 7, 7, 7, 9, 3, 5, 3, 6, 6, 4, 1, 6, 9, 9, 7, 3, 2, 9, 7, 2, 5, 8, 8, 6, 8, 7, 6, 9, 6, 6, 4, 3, 5, 5, 5, 7, 7, 1, 6]
}
RPSCONTEST_BOTS = {
    "testing_please_ignore": TESTING_PLEASE_IGNORE,
    "centrifugal_bumblepuppy_4": CENTRIFUGAL_BUMBLEPUPPY_4,
    "io2_fightinguuu": IO2_FIGHTINGUUU,
    "dllu1": DLLU1,
    "rps_meta_fix": RPS_META_FIX,
    "are_you_a_lucker": ARE_YOU_A_LUCKER,
    "bayes14": BAYES_14,
}


class RPSAgent(object):
    def __init__(self, configuration):
        self.obs = None
        self.config = configuration
        self.history = pd.DataFrame(columns=["step", "action", "opponent_action"])
        self.history.set_index("step", inplace=True)
        self.history = self.history.astype(
            {"action": np.int, "opponent_action": np.int}
        )

        self.step = 0
        self.score = 0
        self.random = SystemRandom()

    def agent(
        self, observation, configuration=None, history=None
    ) -> Tuple[int, pd.DataFrame]:
        if self.random.randint(0, 10) == 7:
            # Change the random seed
            self.random = SystemRandom()
        if configuration is not None:
            self.config = configuration
        if history is not None:
            self.history = history
        self.obs = observation

        self.step = self.obs.step

        # Append the last action of the opponent to the history
        if self.step > 0:
            self.history.loc[
                self.step - 1, "opponent_action"
            ] = self.obs.lastOpponentAction
            self.score = get_score(self.history)

        if self.score - 20 > (1000 - self.step):
            # Don't waste computation time
            action = self.random.randint(0, 2)
            if self.random.randint(0, 10) <= 3:
                action = (action + 1) % SIGNS
            return action, history

        # Choose an action and append it to the history
        action = self.act()
        # Calculate the probability of a win for random play
        # Taken from https://www.kaggle.com/c/rock-paper-scissors/discussion/197402
        win_prob = 0.5 + 0.5 * math.erf(((observation.reward - 20) + 1) / math.sqrt((2/3) * (1000 - observation.step)))
        if win_prob >= 0.92 and get_score(self.history, 10) < 5 and self.random.randrange(0, 100) <= win_prob * 100:
            # Try to secure the win with random play
            action = self.random.randint(0, 2)
            if self.random.randint(0, 10) <= 3:
                action = (action + 2) % SIGNS
        elif get_score(self.history, 15) < -4:
            # If we got outplayed in the last 15 steps, play the counter of the chosen action´s counter with a
            # certain probability
            if self.random.randint(0, 100) <= 20:
                action = (action + 2) % SIGNS
            else:
                # Play randomly
                action = self.random.randint(0, 2)
                if self.random.randint(0, 10) <= 3:
                    action = (action + 1) % SIGNS

        self.history.loc[self.step] = {"action": action, "opponent_action": None}
        return action, self.history

    def act(self) -> int:
        pass


class Policy(object):
    def __init__(self):
        self.history = []
        self.name = "policy"
        self.is_deterministic = False

    def probabilities(self, step: int, score: int, history: pd.DataFrame) -> np.ndarray:
        """
        Returns probabilities for all possible actions and saves the probabilities to the policy´s history
        """
        probabilities = self._get_probs(step, score, history)
        self.history.append(probabilities)
        return probabilities

    def _get_probs(self, step: int, score: int, history: pd.DataFrame) -> np.ndarray:
        """
        Returns probabilities for all possible actions
        """
        pass


def counters(actions: pd.Series) -> pd.Series:
    """
    Returns the counters for the specified actions
    """
    return (actions + 1) % SIGNS


def get_score(history: pd.DataFrame, steps: int = -1) -> int:
    score = 0
    relevant = history if steps < 0 else history.iloc[-steps:]
    score += len(
        relevant[((relevant["opponent_action"] + 1) % SIGNS) == relevant["action"]]
    )
    score -= len(
        relevant[((relevant["action"] + 1) % SIGNS) == relevant["opponent_action"]]
    )
    return score


def one_hot(action: int) -> np.ndarray:
    encoded_action = np.zeros((3,), dtype=np.float)
    encoded_action[action] = 1
    return encoded_action



class RandomPolicy(Policy):
    """
    returns equal probabilities for all actions
    """

    def __init__(self):
        super().__init__()
        self.name = "random_policy"

    def _get_probs(self, step: int, score: int, history: pd.DataFrame) -> np.ndarray:
        return EQUAL_PROBS


class IncrementPolicy(Policy):
    """
    turns all actions given by a policy into their counters
    """

    def __init__(self, policy: Policy):
        super().__init__()
        self.policy = policy
        self.name = "incremented_" + policy.name
        self.is_deterministic = policy.is_deterministic

    def _get_probs(self, step: int, score: int, history: pd.DataFrame) -> np.ndarray:
        # Return equal probabilities if the history of the policy is empty
        if len(self.policy.history) == 0:
            return EQUAL_PROBS
        return np.roll(self.policy.history[-1], 1)


class CounterPolicy(Policy):
    """
    a policy countering the specified policy assuming the opponent uses this policy
    """

    def __init__(self, policy: Policy):
        super().__init__()
        self.policy = policy
        self.name = "counter_" + policy.name
        self.is_deterministic = policy.is_deterministic

    def _get_probs(self, step: int, score: int, history: pd.DataFrame) -> np.ndarray:
        probs = self.policy._get_probs(
            step,
            -score,
            history.rename(
                columns={"action": "opponent_action", "opponent_action": "action"}
            ),
        )
        return np.roll(probs, 1)


class PhasedCounterPolicy(Policy):
    """
    a policy countering the specified policy based on @superant´s finding regarding the phase shift
    as a counter to stochastical agents
    https://www.kaggle.com/superant/anti-opp-transition-matrix-beating-stochastic
    """

    def __init__(self, policy: Policy):
        super().__init__()
        self.policy = policy
        self.name = "phased_counter_" + policy.name
        self.is_deterministic = False

    def _get_probs(self, step: int, score: int, history: pd.DataFrame) -> np.ndarray:
        probs = self.policy._get_probs(
            step,
            -score,
            history.rename(
                columns={"action": "opponent_action", "opponent_action": "action"}
            ),
        )
        # Add 1 to the action with a probability of 40%
        return 0.4 * np.roll(probs, 1) + 0.6 * probs


class StrictPolicy(Policy):
    """
    always selects the action with the highest probability for a given policy
    """

    def __init__(self, policy: Policy):
        super().__init__()
        self.policy = policy
        self.name = "strict_" + policy.name
        self.is_deterministic = True

    def _get_probs(self, step: int, score: int, history: pd.DataFrame) -> np.ndarray:
        if len(self.policy.history) == 0:
            return EQUAL_PROBS
        probs = np.copy(self.policy.history[-1])
        action = np.argmax(probs)
        probs[:] = 0
        probs[action] = 1
        return probs


class AlternatePolicy(Policy):
    """
    Alternates between the specified policies every interval steps
    """

    def __init__(self, policies: List[Policy], interval: int):
        super().__init__()
        self.name = (
            "alternate_"
            + ("_".join([policy.name.replace("_policy", "") for policy in policies]))
            + "_policies"
        )
        self.is_deterministic = all([policy.is_deterministic for policy in policies])
        self.policies = policies
        self.interval = interval
        self.current_policy = 0

    def _get_probs(self, step: int, score: int, history: pd.DataFrame) -> np.ndarray:
        if step % self.interval == 0:
            # Alternate
            self.current_policy = (self.current_policy + 1) % len(self.policies)
        return self.policies[self.current_policy]._get_probs(step, score, history)


class SequencePolicy(Policy):
    """
    chooses actions from a specified sequence
    """

    def __init__(self, sequence: List[int], sequence_name: str):
        super().__init__()
        self.sequence = sequence
        self.name = sequence_name + "_sequence_policy"
        self.is_deterministic = True

    def _get_probs(self, step: int, score: int, history: pd.DataFrame) -> np.ndarray:
        return one_hot(self.sequence[step] % 3)


class DeterministicRPSContestPolicy(Policy):
    """
    a wrapper to run RPS Contest bots
    Adapted from https://www.kaggle.com/purplepuppy/running-rpscontest-bots
    """

    def __init__(self, code, agent_name):
        super().__init__()
        self.name = "deterministic_" + agent_name + "_policy"
        self.is_deterministic = True
        self.code = compile(code, "<string>", "exec")
        self.gg = dict()
        self.symbols = {"R": 0, "P": 1, "S": 2}

    def _get_probs(self, step: int, score: int, history: pd.DataFrame) -> np.ndarray:
        try:
            inp = (
                ""
                if len(history) < 1
                else "RPS"[int(history.loc[step - 1, "opponent_action"])]
            )
            out = (
                "" if len(history) < 1 else "RPS"[int(history.loc[step - 1, "action"])]
            )
            self.gg["input"] = inp
            self.gg["output"] = out
            exec(self.code, self.gg)
            return one_hot(self.symbols[self.gg["output"]])
        except Exception as exception:
            logging.error("An error ocurred in " + self.name + " : " + str(exception))
            return EQUAL_PROBS


class ProbabilisticRPSContestPolicy(Policy):
    """
    a wrapper to run modified probabilistic RPS Contest bots
    Adapted from https://www.kaggle.com/purplepuppy/running-rpscontest-bots
    """

    def __init__(self, code, agent_name):
        super().__init__()
        self.name = "probabilistic_" + agent_name + "_policy"
        self.is_deterministic = True
        self.code = compile(code, "<string>", "exec")
        self.gg = dict()
        self.symbols = {"R": 0, "P": 1, "S": 2}

    def _get_probs(self, step: int, score: int, history: pd.DataFrame) -> np.ndarray:
        try:
            inp = (
                ""
                if len(history) < 1
                else "RPS"[int(history.loc[step - 1, "opponent_action"])]
            )
            out = (
                "" if len(history) < 1 else "RPS"[int(history.loc[step - 1, "action"])]
            )
            self.gg["input"] = inp
            self.gg["output"] = out
            exec(self.code, self.gg)
            return np.array(self.gg["output"])
        except Exception as exception:
            logging.error("An error ocurred in " + self.name + " : " + str(exception))
            return EQUAL_PROBS


class FrequencyPolicy(Policy):
    """
    chooses actions based on the frequency of the opponent´s last actions
    """

    def __init__(self):
        super().__init__()
        self.name = "frequency_policy"

    def _get_probs(self, step: int, score: int, history: pd.DataFrame) -> np.ndarray:
        if len(history) == 0:
            # Return equal probabilities at the start of the episode
            return EQUAL_PROBS
        probs = counters(history["opponent_action"]).value_counts(
            normalize=True, sort=False
        )
        for i in range(SIGNS):
            if i not in probs.keys():
                probs.loc[i] = 0.0
        probs.sort_index(inplace=True)
        return probs.to_numpy()


class CopyLastActionPolicy(Policy):
    """
    copies the last action of the opponent
    """

    def __init__(self):
        super().__init__()
        self.name = "copy_last_action_policy"
        self.is_deterministic = True

    def _get_probs(self, step: int, score: int, history: pd.DataFrame) -> np.ndarray:
        if len(history) == 0:
            # Return equal probabilities at the start of the episode
            return EQUAL_PROBS
        probs = np.zeros((3,), dtype=np.float)
        probs[int(history.loc[step - 1, "opponent_action"])] = 1.0
        return probs


class TransitionMatrixPolicy(Policy):
    """
    uses a simple transition matrix to predict the opponent´s next action and counter it

    Adapted from https://www.kaggle.com/group16/rps-opponent-transition-matrix
    """

    def __init__(self):
        super().__init__()
        self.name = "transition_matrix_policy"
        self.T = np.zeros((3, 3), dtype=np.int)
        self.P = np.zeros((3, 3), dtype=np.float)

    def _get_probs(self, step: int, score: int, history: pd.DataFrame) -> np.ndarray:
        if len(history) > 1:
            # Update the matrices
            last_action = int(history.loc[step - 1, "opponent_action"])
            self.T[int(history.loc[step - 2, "opponent_action"]), last_action] += 1
            self.P = np.divide(self.T, np.maximum(1, self.T.sum(axis=1)).reshape(-1, 1))
            if np.sum(self.P[last_action, :]) == 1:
                return np.roll(self.P[last_action, :], 1)
        return EQUAL_PROBS


class TransitionTensorPolicy(Policy):
    """
    similar to TransitionMatrixPolicy, but takes both agent´s actions into account
    """

    def __init__(self):
        super().__init__()
        self.name = "transition_tensor_policy"
        self.T = np.zeros((3, 3, 3), dtype=np.int)
        self.P = np.zeros((3, 3, 3), dtype=np.float)

    def _get_probs(self, step: int, score: int, history: pd.DataFrame) -> np.ndarray:
        if len(history) > 1:
            # Update the matrices
            last_action = int(history.loc[step - 1, "action"])
            opponent_last_action = int(history.loc[step - 1, "opponent_action"])
            self.T[
                int(history.loc[step - 2, "opponent_action"]),
                int(history.loc[step - 2, "action"]),
                last_action,
            ] += 1
            self.P = np.divide(
                self.T, np.maximum(1, self.T.sum(axis=2)).reshape(-1, 3, 1)
            )
            if np.sum(self.P[opponent_last_action, last_action, :]) == 1:
                return np.roll(self.P[opponent_last_action, last_action, :], 1)
        return EQUAL_PROBS


class MaxHistoryPolicy(Policy):
    """
    searches for similar situations in the game history and assumes the past is doomed to repeat itself
    prefers the longest matching sequence
    """

    def __init__(self, max_sequence_length: int):
        super().__init__()
        self.name = "max_history_policy"
        self.max_sequence_length = max_sequence_length
        self.sequences = defaultdict(lambda: np.zeros((3,), dtype=np.int))

    def _get_probs(self, step: int, score: int, history: pd.DataFrame) -> np.ndarray:
        if len(history) < 2:
            # return equal probabilities at the start of the game
            return EQUAL_PROBS
        # Update the stored sequences with the opponent´s last move
        for sequence_length in range(
            1, min(len(history) - 1, self.max_sequence_length) + 1
        ):
            sequence = np.array2string(
                history.iloc[-sequence_length - 1 : -1].to_numpy()
            )
            self.sequences[sequence][int(history.loc[step - 1, "opponent_action"])] += 1
        # Try to find a match for the current history and get the corresponding probabilities
        for sequence_length in range(
            min(len(history), self.max_sequence_length), 0, -1
        ):
            # Determine whether the sequence has already occurred
            sequence = np.array2string(history.iloc[-sequence_length:].to_numpy())
            if sequence not in self.sequences.keys():
                continue
            # Return the corresponding probabilities
            return self.sequences[sequence] / sum(self.sequences[sequence])
        return EQUAL_PROBS


class MaxOpponentHistoryPolicy(Policy):
    """
    like MaxHistoryPolicy, but only looks at the moves of the opponent
    """

    def __init__(self, max_sequence_length: int):
        super().__init__()
        self.name = "max_opponent_history_policy"
        self.max_sequence_length = max_sequence_length
        self.sequences = defaultdict(lambda: np.zeros((3,), dtype=np.int))

    def _get_probs(self, step: int, score: int, history: pd.DataFrame) -> np.ndarray:
        if len(history) < 2:
            # return equal probabilities at the start of the game
            return EQUAL_PROBS
        # Update the stored sequences with the opponent´s last move
        for sequence_length in range(
            1, min(len(history) - 1, self.max_sequence_length) + 1
        ):
            sequence = np.array2string(
                history.iloc[-sequence_length - 1 : -1][["opponent_action"]].to_numpy()
            )
            self.sequences[sequence][int(history.loc[step - 1, "opponent_action"])] += 1
        # Try to find a match for the current history and get the corresponding probabilities
        for sequence_length in range(
            min(len(history), self.max_sequence_length), 0, -1
        ):
            # Determine whether the sequence has already occurred
            sequence = np.array2string(
                history.iloc[-sequence_length:][["opponent_action"]].to_numpy()
            )
            if sequence not in self.sequences.keys():
                continue
            # Return the corresponding probabilities
            return self.sequences[sequence] / sum(self.sequences[sequence])
        return EQUAL_PROBS


class RandomForestPolicy(Policy):
    """
    uses a random forest classificator to predict the opponent´s action using the last moves as data
    """

    def __init__(self, n_estimators: int, max_train_size: int, prediction_window: int):
        super().__init__()
        self.name = "random_forest_policy"
        self.is_deterministic = True
        self.model = RandomForestClassifier(n_estimators=n_estimators)
        self.max_train_size = max_train_size
        self.prediction_window = prediction_window
        self.X_train = np.ndarray(shape=(0, prediction_window * 2), dtype=np.int)
        self.y_train = np.ndarray(shape=(0, 1), dtype=np.int)

    def _get_probs(self, step: int, score: int, history: pd.DataFrame) -> np.ndarray:
        if len(history) < self.prediction_window + 1:
            # Return equal probabilities until we have enough data
            return EQUAL_PROBS
        # Add the last prediction_window steps to the training data
        last_steps = history.iloc[-self.prediction_window - 1 : -1][
            ["action", "opponent_action"]
        ].to_numpy()
        self.X_train = np.append(
            self.X_train, last_steps.reshape(1, self.prediction_window * 2)
        )
        self.y_train = np.append(self.y_train, history.iloc[-1]["opponent_action"])
        self.X_train = self.X_train.reshape(-1, self.prediction_window * 2)
        # Ensure we don´t use more than max_train_size samples
        if len(self.X_train) > self.max_train_size:
            self.X_train = self.X_train[1:]
            self.y_train = self.y_train[1:]
        # Fit the model
        self.model.fit(self.X_train, self.y_train)
        # Predict the opponent´s next action
        X_predict = history.iloc[-self.prediction_window :][
            ["action", "opponent_action"]
        ].to_numpy()
        prediction = self.model.predict(
            X_predict.reshape(1, self.prediction_window * 2)
        )
        # Return the counter of the action
        return np.roll(one_hot(int(prediction[0])), 1)


class WinTieLosePolicy(Policy):
    """
    chooses the next move based on the result of the last one, e.g. repeats winning moves and switches when losing
    """

    def __init__(self, on_win: int, on_tie: int, on_lose: int):
        super().__init__()
        self.name = (
            "on_win_"
            + str(on_win)
            + "_on_tie_"
            + str(on_tie)
            + "_on_lose_"
            + str(on_lose)
            + "_policy"
        )
        self.is_deterministic = True
        self.on_win = on_win
        self.on_tie = on_tie
        self.on_lose = on_lose
        self.last_score = 0

    def _get_probs(self, step: int, score: int, history: pd.DataFrame) -> np.ndarray:
        if len(history) < 1:
            # Return equal probabilities on the first step
            return EQUAL_PROBS
        result = score - self.last_score
        self.last_score = score
        if result == 1:
            shift = self.on_win
        elif result == 0:
            shift = self.on_tie
        else:
            shift = self.on_lose
        return one_hot((int(history.loc[step - 1, "action"]) + shift) % 3)


class FlattenPolicy(Policy):
    """
    core concept developed by Tony Robinson (https://www.kaggle.com/tonyrobinson)
    Adapted from https://www.kaggle.com/tonyrobinson/flatten
    """

    def __init__(self):
        super().__init__()
        self.name = "flatten_policy"
        self.is_deterministic = False
        self.countInc = 1e-30
        self.countOp = self.countInc * np.ones((3, 3, 3))
        self.countAg = self.countInc * np.ones((3, 3, 3))
        self.reward = np.array([[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]])
        self.offset = 2.0
        self.halfLife = 100.0
        self.countPow = math.exp(math.log(2) / self.halfLife)
        self.histAgent = []  # Agent history
        self.histOpponent = []  # Opponent history
        self.nwin = 0

    def _get_probs(self, step: int, score: int, history: pd.DataFrame) -> np.ndarray:
        if len(history) == 0:
            return EQUAL_PROBS
        opp_action = int(history.loc[step - 1, 'opponent_action'])
        self.histOpponent.append(opp_action)
        self.histAgent.append(int(history.loc[step - 1, 'action']))
        # score last game
        self.nwin += self.reward[self.histAgent[-1], opp_action]

        if step > 1:
            # increment predictors
            self.countOp[self.histOpponent[-2], self.histAgent[-2], self.histOpponent[-1]] += self.countInc
            self.countAg[self.histOpponent[-2], self.histAgent[-2], self.histAgent[-1]] += self.countInc
        if len(self.histOpponent) < 2:
            return EQUAL_PROBS
        # stochastically flatten the distribution
        count = self.countAg[self.histOpponent[-1], self.histAgent[-1]]
        dist = (self.offset + 1) * count.max() - self.offset * count.min() - count
        self.countInc *= self.countPow
        if np.sum(np.abs(dist)) == 0:
            return EQUAL_PROBS
        else:
            if np.min(dist) < 0:
                dist -= np.min(dist)
            dist *= 1 / np.sum(dist)
            return dist


class RockPolicy(Policy):
    """
    chooses Rock the whole time
    """

    def __init__(self):
        super().__init__()
        self.name = "rock_policy"
        self.probs = np.array([1, 0, 0], dtype=np.float)
        self.is_deterministic = True

    def _get_probs(self, step: int, score: int, history: pd.DataFrame) -> np.ndarray:
        return self.probs


class PaperPolicy(Policy):
    """
    chooses Paper the whole time
    """

    def __init__(self):
        super().__init__()
        self.name = "paper_policy"
        self.probs = np.array([0, 1, 0], dtype=np.float)
        self.is_deterministic = True

    def _get_probs(self, step: int, score: int, history: pd.DataFrame) -> np.ndarray:
        return self.probs


class ScissorsPolicy(Policy):
    """
    chooses Scissors the whole time
    """

    def __init__(self):
        super().__init__()
        self.name = "scissors_policy"
        self.probs = np.array([0, 0, 1], dtype=np.float)
        self.is_deterministic = True

    def _get_probs(self, step: int, score: int, history: pd.DataFrame) -> np.ndarray:
        return self.probs


def get_policies():
    """
    Returns a list of many polices
    """
    # Initialize the different sets of policies
    random_policies = [RandomPolicy()]
    # Policies we shouldn´t derive incremented ones from
    basic_policies = [RockPolicy(), PaperPolicy(), ScissorsPolicy()]
    advanced_policies = [
        GeometryPolicy(),
        SeedSearchPolicy(10000)
    ]
    # Add some popular sequences
    for seq_name, seq in SEQUENCES.items():
        advanced_policies.append(SequencePolicy(seq, seq_name))
    # Add some RPS Contest bots to the ensemble
    for agent_name, code in RPSCONTEST_BOTS.items():
        advanced_policies.append(DeterministicRPSContestPolicy(code, agent_name))
    # Strict versions of the advanced policies
    strict_policies = [
        StrictPolicy(policy)
        for policy in advanced_policies
        if not policy.is_deterministic
    ]
    # Counter policies
    counter_policies = [
        CounterPolicy(FrequencyPolicy()),
        CounterPolicy(CopyLastActionPolicy()),
        CounterPolicy(TransitionMatrixPolicy()),
        CounterPolicy(TransitionTensorPolicy()),
        CounterPolicy(MaxHistoryPolicy(15)),
        CounterPolicy(MaxOpponentHistoryPolicy(15)),
        CounterPolicy(IocainePolicy()),
        CounterPolicy(GreenbergPolicy()),
        AntiGeometryPolicy(),
        CounterPolicy(AntiGeometryPolicy())
    ]
    # Add some RPS Contest bots to the ensemble
    for agent_name, code in RPSCONTEST_BOTS.items():
        counter_policies.append(
            CounterPolicy(DeterministicRPSContestPolicy(code, agent_name))
        )
    strict_counter_policies = [
        StrictPolicy(policy)
        for policy in counter_policies
        if not policy.is_deterministic
    ]
    # Sicilian reasoning
    incremented_policies = [
        IncrementPolicy(policy)
        for policy in (
            advanced_policies
            + strict_policies
            + counter_policies
            + strict_counter_policies
        )
    ]
    double_incremented_policies = [
        IncrementPolicy(policy) for policy in incremented_policies
    ]
    policies = (
        random_policies
        + advanced_policies
        + strict_policies
        + incremented_policies
        + double_incremented_policies
        + counter_policies
        + strict_counter_policies
    )
    return policies
"""
Adapted from @superant´s "RPS Geometry" notebook
https://www.kaggle.com/superant/rps-geometry-silver-rank-by-minimal-logic
"""


basis = np.array(
    [1, cmath.exp(2j * cmath.pi * 1 / 3), cmath.exp(2j * cmath.pi * 2 / 3)]
)

HistMatchResult = namedtuple("HistMatchResult", "idx length")


def find_all_longest(seq, max_len=None) -> List[HistMatchResult]:
    """
    Find all indices where end of `seq` matches some past.
    """
    result = []
    i_search_start = len(seq) - 2

    while i_search_start > 0:
        i_sub = -1
        i_search = i_search_start
        length = 0

        while i_search >= 0 and seq[i_sub] == seq[i_search]:
            length += 1
            i_sub -= 1
            i_search -= 1

            if max_len is not None and length > max_len:
                break

        if length > 0:
            result.append(HistMatchResult(i_search_start + 1, length))

        i_search_start -= 1

    result = sorted(result, key=operator.attrgetter("length"), reverse=True)
    return result


def probs_to_complex(p):
    return p @ basis


def _fix_probs(probs):
    """
    Put probs back into triangle. Sometimes this happens due to rounding errors or if you
    use complex numbers which are outside the triangle.
    """
    if min(probs) < 0:
        probs -= min(probs)

    probs /= sum(probs)
    return probs


def complex_to_probs(z):
    probs = (2 * (z * basis.conjugate()).real + 1) / 3
    probs = _fix_probs(probs)
    return probs


def z_from_action(action):
    return basis[action]


def sample_from_z(z):
    probs = complex_to_probs(z)
    return np.random.choice(3, p=probs)


def bound(z):
    return probs_to_complex(complex_to_probs(z))


def norm(z):
    return bound(z / abs(z))


class Pred:
    def __init__(self, *, alpha):
        self.offset = 0
        self.alpha = alpha
        self.last_feat = None

    def train(self, target):
        if self.last_feat is not None:
            offset = target * self.last_feat.conjugate()  # fixed

            self.offset = (1 - self.alpha) * self.offset + self.alpha * offset

    def predict(self, feat):
        """
        feat is an arbitrary feature with a probability on 0,1,2
        anything which could be useful anchor to start with some kind of sensible direction
        """
        feat = norm(feat)

        # offset = mean(target - feat)
        # so here we see something like: result = feat + mean(target - feat)
        # which seems natural and accounts for the correlation between target and feat
        # all RPSContest bots do no more than that as their first step, just in a different way

        result = feat * self.offset
        self.last_feat = feat
        return result


class BaseAgent(Policy):
    def __init__(self):
        super().__init__()
        self.name = "geometry_policy"
        self.is_deterministic = False
        self.my_hist = []
        self.opp_hist = []
        self.my_opp_hist = []
        self.outcome_hist = []
        self.step = None

    def _get_probs(self, step: int, score: int, history: pd.DataFrame) -> np.ndarray:
        try:
            if step == 0:
                return EQUAL_PROBS
            else:
                self.my_hist.append(int(history.loc[step - 1, "action"]))

            self.step = step

            opp = int(history.loc[step - 1, "opponent_action"])
            my = self.my_hist[-1]
            self.my_opp_hist.append((my, opp))
            self.opp_hist.append(opp)

            outcome = {0: 0, 1: 1, 2: -1}[(my - opp) % 3]
            self.outcome_hist.append(outcome)

            probs = self.calculate_probs()
            return probs
        except Exception:
            traceback.print_exc(file=sys.stderr)
            return EQUAL_PROBS

    def calculate_probs(self) -> np.ndarray:
        pass


class GeometryPolicy(BaseAgent):
    def __init__(self, alpha=0.01):
        super().__init__()
        self.name = "geometry_" + str(alpha) + "_policy"
        self.predictor = Pred(alpha=alpha)

    def calculate_probs(self) -> np.ndarray:
        self.train()
        pred = self.preds()
        return complex_to_probs(pred)

    def train(self):
        last_beat_opp = z_from_action((self.opp_hist[-1] + 1) % 3)
        self.predictor.train(last_beat_opp)

    def preds(self):
        hist_match = find_all_longest(self.my_opp_hist, max_len=20)

        if not hist_match:
            return 0

        feat = z_from_action(self.opp_hist[hist_match[0].idx])
        pred = self.predictor.predict(feat)
        return pred
"""
Iocaine Powder by Dan Egnor translated into Python by DAvid Bau
Code adapted from http://davidbau.com/downloads/rps/rps-iocaine.py
"""


def recall(age, hist):
    """Looking at the last 'age' points in 'hist', finds the
    last point with the longest similarity to the current point,
    returning 0 if none found."""
    end, length = 0, 0
    for past in range(1, min(age + 1, len(hist) - 1)):
        if length >= len(hist) - past:
            break
        for i in range(-1 - length, 0):
            if hist[i - past] != hist[i]:
                break
        else:
            for length in range(length + 1, len(hist) - past):
                if hist[-past - length - 1] != hist[-length - 1]:
                    break
            else:
                length += 1
            end = len(hist) - past
    return end


def beat(i):
    return (i + 1) % 3


def loseto(i):
    return (i - 1) % 3


class Stats:
    """Maintains three running counts and returns the highest count based
    on any given time horizon and threshold."""

    def __init__(self):
        self.sum = [[0, 0, 0]]

    def add(self, move, score):
        self.sum[-1][move] += score

    def advance(self):
        self.sum.append(self.sum[-1])

    def max(self, age, default, score):
        if age >= len(self.sum):
            diff = self.sum[-1]
        else:
            diff = [self.sum[-1][i] - self.sum[-1 - age][i] for i in range(3)]
        m = max(diff)
        if m > score:
            return diff.index(m), m
        return default, score


class Predictor:
    """The basic iocaine second- and triple-guesser.    Maintains stats on the
    past benefits of trusting or second- or triple-guessing a given strategy,
    and returns the prediction of that strategy (or the second- or triple-
    guess) if past stats are deviating from zero farther than the supplied
    "best" guess so far."""

    def __init__(self):
        self.stats = Stats()
        self.lastguess = -1

    def addguess(self, lastmove, guess):
        if lastmove != -1:
            diff = (lastmove - self.prediction) % 3
            self.stats.add(beat(diff), 1)
            self.stats.add(loseto(diff), -1)
            self.stats.advance()
        self.prediction = guess

    def bestguess(self, age, best):
        bestdiff = self.stats.max(age, (best[0] - self.prediction) % 3, best[1])
        return (bestdiff[0] + self.prediction) % 3, bestdiff[1]


ages = [1000, 100, 10, 5, 2, 1]


class IocainePolicy(Policy):
    def __init__(self):
        """Build second-guessers for 50 strategies: 36 history-based strategies,
        12 simple frequency-based strategies, the constant-move strategy, and
        the basic random-number-generator strategy.    Also build 6 meta second
        guessers to evaluate 6 different time horizons on which to score
        the 50 strategies' second-guesses."""
        super().__init__()
        self.name = "iocaine_powder_policy"
        self.is_deterministic = True
        self.predictors = []
        self.predict_history = self.predictor((len(ages), 2, 3))
        self.predict_frequency = self.predictor((len(ages), 2))
        self.predict_fixed = self.predictor()
        self.predict_random = self.predictor()
        self.predict_meta = [Predictor() for a in range(len(ages))]
        self.stats = [Stats() for i in range(2)]
        self.histories = [[], [], []]

    def predictor(self, dims=None):
        """Returns a nested array of predictor objects, of the given dimensions."""
        if dims:
            return [self.predictor(dims[1:]) for i in range(dims[0])]
        self.predictors.append(Predictor())
        return self.predictors[-1]

    def _get_probs(self, step: int, score: int, history: pd.DataFrame) -> np.ndarray:
        """The main iocaine "move" function."""
        if len(history) == 0:
            them = -1
        else:
            them = int(history.loc[step - 1, "opponent_action"])
            self.histories[0].append(int(history.loc[step - 1, "action"]))

        # histories[0] stores our moves (last one already previously decided);
        # histories[1] stores their moves (last one just now being supplied to us);
        # histories[2] stores pairs of our and their last moves.
        # stats[0] and stats[1] are running counters our recent moves and theirs.
        if them != -1:
            self.histories[1].append(them)
            self.histories[2].append((self.histories[0][-1], them))
            for watch in range(2):
                self.stats[watch].add(self.histories[watch][-1], 1)

        # Execute the basic RNG strategy and the fixed-move strategy.
        rand = random.randrange(3)
        self.predict_random.addguess(them, rand)
        self.predict_fixed.addguess(them, 0)

        # Execute the history and frequency stratgies.
        for a, age in enumerate(ages):
            # For each time window, there are three ways to recall a similar time:
            # (0) by history of my moves; (1) their moves; or (2) pairs of moves.
            # Set "best" to these three timeframes (zero if no matching time).
            best = [recall(age, hist) for hist in self.histories]
            for mimic in range(2):
                # For each similar historical moment, there are two ways to anticipate
                # the future: by mimicing what their move was; or mimicing what my
                # move was.    If there were no similar moments, just move randomly.
                for watch, when in enumerate(best):
                    if not when:
                        move = rand
                    else:
                        move = self.histories[mimic][when]
                    self.predict_history[a][mimic][watch].addguess(them, move)
                # Also we can anticipate the future by expecting it to be the same
                # as the most frequent past (either counting their moves or my moves).
                mostfreq, score = self.stats[mimic].max(age, rand, -1)
                self.predict_frequency[a][mimic].addguess(them, mostfreq)

        # All the predictors have been updated, but we have not yet scored them
        # and chosen a winner for this round.    There are several timeframes
        # on which we can score second-guessing, and we don't know timeframe will
        # do best.    So score all 50 predictors on all 6 timeframes, and record
        # the best 6 predictions in meta predictors, one for each timeframe.
        for meta, age in enumerate(ages):
            best = (-1, -1)
            for predictor in self.predictors:
                best = predictor.bestguess(age, best)
            self.predict_meta[meta].addguess(them, best[0])

        # Finally choose the best meta prediction from the final six, scoring
        # these against each other on the whole-game timeframe.
        best = (-1, -1)
        for meta in range(len(ages)):
            best = self.predict_meta[meta].bestguess(len(self.histories[0]), best)

        # And return it.
        return one_hot(best[0])
"""
greenberg roshambo bot, winner of 2nd annual roshambo programming competition
http://webdocs.cs.ualberta.ca/~darse/rsbpc.html

original source by Andrzej Nagorko
http://www.mathpuzzle.com/greenberg.c

Python translation by Travis Erdman
https://github.com/erdman/roshambo
"""


def player(my_moves, opp_moves):

    rps_to_text = ("rock", "paper", "scissors")
    rps_to_num = {"rock": 0, "paper": 1, "scissors": 2}
    wins_with = (1, 2, 0)  # superior
    best_without = (2, 0, 1)  # inferior

    lengths = (10, 20, 30, 40, 49, 0)
    p_random = random.choice([0, 1, 2])  # called 'guess' in iocaine

    TRIALS = 1000
    score_table = ((0, -1, 1), (1, 0, -1), (-1, 1, 0))
    T = len(opp_moves)  # so T is number of trials completed

    def min_index(values):
        return min(enumerate(values), key=itemgetter(1))[0]

    def max_index(values):
        return max(enumerate(values), key=itemgetter(1))[0]

    def find_best_prediction(l):  # l = len
        bs = -TRIALS
        bp = 0
        if player.p_random_score > bs:
            bs = player.p_random_score
            bp = p_random
        for i in range(3):
            for j in range(24):
                for k in range(4):
                    new_bs = player.p_full_score[T % 50][j][k][i] - (
                        player.p_full_score[(50 + T - l) % 50][j][k][i] if l else 0
                    )
                    if new_bs > bs:
                        bs = new_bs
                        bp = (player.p_full[j][k] + i) % 3
                for k in range(2):
                    new_bs = player.r_full_score[T % 50][j][k][i] - (
                        player.r_full_score[(50 + T - l) % 50][j][k][i] if l else 0
                    )
                    if new_bs > bs:
                        bs = new_bs
                        bp = (player.r_full[j][k] + i) % 3
            for j in range(2):
                for k in range(2):
                    new_bs = player.p_freq_score[T % 50][j][k][i] - (
                        player.p_freq_score[(50 + T - l) % 50][j][k][i] if l else 0
                    )
                    if new_bs > bs:
                        bs = new_bs
                        bp = (player.p_freq[j][k] + i) % 3
                    new_bs = player.r_freq_score[T % 50][j][k][i] - (
                        player.r_freq_score[(50 + T - l) % 50][j][k][i] if l else 0
                    )
                    if new_bs > bs:
                        bs = new_bs
                        bp = (player.r_freq[j][k] + i) % 3
        return bp

    if not my_moves:
        player.opp_history = [
            0
        ]  # pad to match up with 1-based move indexing in original
        player.my_history = [0]
        player.gear = [[0] for _ in range(24)]
        # init()
        player.p_random_score = 0
        player.p_full_score = [
            [[[0 for i in range(3)] for k in range(4)] for j in range(24)]
            for l in range(50)
        ]
        player.r_full_score = [
            [[[0 for i in range(3)] for k in range(2)] for j in range(24)]
            for l in range(50)
        ]
        player.p_freq_score = [
            [[[0 for i in range(3)] for k in range(2)] for j in range(2)]
            for l in range(50)
        ]
        player.r_freq_score = [
            [[[0 for i in range(3)] for k in range(2)] for j in range(2)]
            for l in range(50)
        ]
        player.s_len = [0] * 6

        player.p_full = [[0, 0, 0, 0] for _ in range(24)]
        player.r_full = [[0, 0] for _ in range(24)]
    else:
        player.my_history.append(rps_to_num[my_moves[-1]])
        player.opp_history.append(rps_to_num[opp_moves[-1]])
        # update_scores()
        player.p_random_score += score_table[p_random][player.opp_history[-1]]
        player.p_full_score[T % 50] = [
            [
                [
                    player.p_full_score[(T + 49) % 50][j][k][i]
                    + score_table[(player.p_full[j][k] + i) % 3][player.opp_history[-1]]
                    for i in range(3)
                ]
                for k in range(4)
            ]
            for j in range(24)
        ]
        player.r_full_score[T % 50] = [
            [
                [
                    player.r_full_score[(T + 49) % 50][j][k][i]
                    + score_table[(player.r_full[j][k] + i) % 3][player.opp_history[-1]]
                    for i in range(3)
                ]
                for k in range(2)
            ]
            for j in range(24)
        ]
        player.p_freq_score[T % 50] = [
            [
                [
                    player.p_freq_score[(T + 49) % 50][j][k][i]
                    + score_table[(player.p_freq[j][k] + i) % 3][player.opp_history[-1]]
                    for i in range(3)
                ]
                for k in range(2)
            ]
            for j in range(2)
        ]
        player.r_freq_score[T % 50] = [
            [
                [
                    player.r_freq_score[(T + 49) % 50][j][k][i]
                    + score_table[(player.r_freq[j][k] + i) % 3][player.opp_history[-1]]
                    for i in range(3)
                ]
                for k in range(2)
            ]
            for j in range(2)
        ]
        player.s_len = [
            s + score_table[p][player.opp_history[-1]]
            for s, p in zip(player.s_len, player.p_len)
        ]

    # update_history_hash()
    if not my_moves:
        player.my_history_hash = [[0], [0], [0], [0]]
        player.opp_history_hash = [[0], [0], [0], [0]]
    else:
        player.my_history_hash[0].append(player.my_history[-1])
        player.opp_history_hash[0].append(player.opp_history[-1])
        for i in range(1, 4):
            player.my_history_hash[i].append(
                player.my_history_hash[i - 1][-1] * 3 + player.my_history[-1]
            )
            player.opp_history_hash[i].append(
                player.opp_history_hash[i - 1][-1] * 3 + player.opp_history[-1]
            )

    # make_predictions()

    for i in range(24):
        player.gear[i].append((3 + player.opp_history[-1] - player.p_full[i][2]) % 3)
        if T > 1:
            player.gear[i][T] += 3 * player.gear[i][T - 1]
        player.gear[i][
            T
        ] %= 9  # clearly there are 9 different gears, but original code only allocated 3 gear_freq's
        # code apparently worked, but got lucky with undefined behavior
        # I fixed by allocating gear_freq with length = 9
    if not my_moves:
        player.freq = [[0, 0, 0], [0, 0, 0]]
        value = [[0, 0, 0], [0, 0, 0]]
    else:
        player.freq[0][player.my_history[-1]] += 1
        player.freq[1][player.opp_history[-1]] += 1
        value = [
            [
                (1000 * (player.freq[i][2] - player.freq[i][1])) / float(T),
                (1000 * (player.freq[i][0] - player.freq[i][2])) / float(T),
                (1000 * (player.freq[i][1] - player.freq[i][0])) / float(T),
            ]
            for i in range(2)
        ]
    player.p_freq = [
        [wins_with[max_index(player.freq[i])], wins_with[max_index(value[i])]]
        for i in range(2)
    ]
    player.r_freq = [
        [best_without[min_index(player.freq[i])], best_without[min_index(value[i])]]
        for i in range(2)
    ]

    f = [[[[0, 0, 0] for k in range(4)] for j in range(2)] for i in range(3)]
    t = [[[0, 0, 0, 0] for j in range(2)] for i in range(3)]

    m_len = [[0 for _ in range(T)] for i in range(3)]

    for i in range(T - 1, 0, -1):
        m_len[0][i] = 4
        for j in range(4):
            if player.my_history_hash[j][i] != player.my_history_hash[j][T]:
                m_len[0][i] = j
                break
        for j in range(4):
            if player.opp_history_hash[j][i] != player.opp_history_hash[j][T]:
                m_len[1][i] = j
                break
        for j in range(4):
            if (
                player.my_history_hash[j][i] != player.my_history_hash[j][T]
                or player.opp_history_hash[j][i] != player.opp_history_hash[j][T]
            ):
                m_len[2][i] = j
                break

    for i in range(T - 1, 0, -1):
        for j in range(3):
            for k in range(m_len[j][i]):
                f[j][0][k][player.my_history[i + 1]] += 1
                f[j][1][k][player.opp_history[i + 1]] += 1
                t[j][0][k] += 1
                t[j][1][k] += 1

                if t[j][0][k] == 1:
                    player.p_full[j * 8 + 0 * 4 + k][0] = wins_with[
                        player.my_history[i + 1]
                    ]
                if t[j][1][k] == 1:
                    player.p_full[j * 8 + 1 * 4 + k][0] = wins_with[
                        player.opp_history[i + 1]
                    ]
                if t[j][0][k] == 3:
                    player.p_full[j * 8 + 0 * 4 + k][1] = wins_with[
                        max_index(f[j][0][k])
                    ]
                    player.r_full[j * 8 + 0 * 4 + k][0] = best_without[
                        min_index(f[j][0][k])
                    ]
                if t[j][1][k] == 3:
                    player.p_full[j * 8 + 1 * 4 + k][1] = wins_with[
                        max_index(f[j][1][k])
                    ]
                    player.r_full[j * 8 + 1 * 4 + k][0] = best_without[
                        min_index(f[j][1][k])
                    ]

    for j in range(3):
        for k in range(4):
            player.p_full[j * 8 + 0 * 4 + k][2] = wins_with[max_index(f[j][0][k])]
            player.r_full[j * 8 + 0 * 4 + k][1] = best_without[min_index(f[j][0][k])]

            player.p_full[j * 8 + 1 * 4 + k][2] = wins_with[max_index(f[j][1][k])]
            player.r_full[j * 8 + 1 * 4 + k][1] = best_without[min_index(f[j][1][k])]

    for j in range(24):
        gear_freq = [
            0
        ] * 9  # was [0,0,0] because original code incorrectly only allocated array length 3

        for i in range(T - 1, 0, -1):
            if player.gear[j][i] == player.gear[j][T]:
                gear_freq[player.gear[j][i + 1]] += 1

        # original source allocated to 9 positions of gear_freq array, but only allocated first three
        # also, only looked at first 3 to find the max_index
        # unclear whether to seek max index over all 9 gear_freq's or just first 3 (as original code)
        player.p_full[j][3] = (player.p_full[j][1] + max_index(gear_freq)) % 3

    # end make_predictions()

    player.p_len = [find_best_prediction(l) for l in lengths]

    return rps_to_num[rps_to_text[player.p_len[max_index(player.s_len)]]]


class GreenbergPolicy(Policy):
    def __init__(self):
        super().__init__()
        self.name = "greenberg_policy"
        self.is_deterministic = True

    def _get_probs(self, step: int, score: int, history: pd.DataFrame) -> np.ndarray:
        rps_to_text = ("rock", "paper", "scissors")
        act = player(
            [rps_to_text[int(action)] for action in history.loc[:, "action"]],
            [rps_to_text[int(action)] for action in history.loc[:, "opponent_action"]],
        )
        return one_hot(act)

# Global Variables


class SeedSearchPolicy(Policy):
    """
    Trying to crack seeds
    Adapted from Taaha Khan´s notebook "RPS: Cracking Random Number Generators"
    https://www.kaggle.com/taahakhan/rps-cracking-random-number-generators
    """

    def __init__(self, seed_count: int):
        super().__init__()
        self.name = "seed_search_policy"
        self.is_deterministic = True  # Actually not, but we don´t need a strict version
        self.seeds = list(range(seed_count))
        self.previous_moves = []

    def _get_probs(self, step: int, score: int, history: pd.DataFrame) -> np.ndarray:
        # Saving the current state
        init_state = random.getstate()
        next_move = -1
        # If there still are multiple candidates
        if len(history) > 0 and len(self.seeds) > 1:
            # Saving previous moves
            self.previous_moves.append(int(history.loc[step - 1, 'opponent_action']))
            # Checking each possible seed
            for i in range(len(self.seeds) - 1, -1, -1):
                # Running for previous moves
                random.seed(self.seeds[i])
                for s in range(step):
                    move = random.randint(0, 2)
                    # Testing their move order
                    if move != self.previous_moves[s]:
                        self.seeds.pop(i)
                        break
        # Seed found: Get the next move
        elif len(self.seeds) == 1:
            random.seed(self.seeds[0])
            for _ in range(step):
                move = random.randint(0, 2)
            next_move = random.randint(0, 2)

        # Resetting the state to not interfere with the opponent
        random.setstate(init_state)
        if next_move > -1:
            return one_hot((next_move + 1) % 3)
        return EQUAL_PROBS



class AntiGeometryPolicy(Policy):
    """
    a counter to the popular Geometry bot
    written by @robga
    adapted from https://www.kaggle.com/robga/beating-geometry-bot/output
    """

    def __init__(self):
        super().__init__()
        self.name = "anti_geometry_policy"
        self.is_deterministic = False
        self.opp_hist = []
        self.my_opp_hist = []
        self.offset = 0
        self.last_feat = None
        self.basis = np.array([1, cmath.exp(2j * cmath.pi * 1 / 3), cmath.exp(2j * cmath.pi * 2 / 3)])
        self.HistMatchResult = namedtuple("HistMatchResult", "idx length")

    def find_all_longest(self,seq, max_len=None):
        result = []
        i_search_start = len(seq) - 2
        while i_search_start > 0:
            i_sub = -1
            i_search = i_search_start
            length = 0
            while i_search >= 0 and seq[i_sub] == seq[i_search]:
                length += 1
                i_sub -= 1
                i_search -= 1
                if max_len is not None and length > max_len: break
            if length > 0: result.append(self.HistMatchResult(i_search_start + 1, length))
            i_search_start -= 1
        return sorted(result, key=operator.attrgetter("length"), reverse=True)

    def complex_to_probs(self, z):
        probs = (2 * (z * self.basis.conjugate()).real + 1) / 3
        if min(probs) < 0: probs -= min(probs)
        return probs / sum(probs)

    def _get_probs(self, step: int, score: int, history: pd.DataFrame) -> np.ndarray:

        if len(history) == 0:
            return EQUAL_PROBS
        else:
            self.action = int(history.loc[step - 1, 'action'])
            self.my_opp_hist.append((int(history.loc[step - 1, 'opponent_action']), self.action))
            self.opp_hist.append(self.action)

            if self.last_feat is not None:
                this_offset = (self.basis[(self.opp_hist[-1] + 1) % 3]) * self.last_feat.conjugate()
                self.offset = (1 - .01) * self.offset + .01 * this_offset

            hist_match = self.find_all_longest(self.my_opp_hist, 20)
            if not hist_match:
                pred = 0
            else:
                feat = self.basis[self.opp_hist[hist_match[0].idx]]
                self.last_feat = self.complex_to_probs(feat / abs(feat)) @ self.basis
                pred = self.last_feat * self.offset * cmath.exp(2j * cmath.pi * 1 / 9)

            probs = self.complex_to_probs(pred)
            if probs[np.argmax(probs)] > .334:
                return one_hot((int(np.argmax(probs)) + 1) % 3)
            else:
                return probs

logging.basicConfig(level=logging.INFO)


class StatisticalPolicyEnsembleAgent(RPSAgent):
    """
    evaluates the performance of different policies and assigns each policy a weight based on the policy´s
    historical performance
    After that the combined weighted probabilities from the policies are used as a probability distribution
    for the agent´s actions
    """

    def __init__(self, configuration, strict: bool = False):
        super().__init__(configuration)
        self.strict_agent = strict
        self.policies = get_policies()
        if self.strict_agent:
            self.policies = [
                policy for policy in self.policies if policy.is_deterministic
            ]
        self.name_to_policy = {policy.name: policy for policy in self.policies}

        # The different combinations of decay values, reset probabilities and zero clips
        self.configurations = [
            (0.8, 0.0, False),
            (0.8866, 0.0, False),
            (0.93, 0.0, False),
            (0.9762, 0.05, True),
            (0.9880, 0.0, False),
            (0.99815, 0.1, False),
            (1.0, 0.0, False),
        ]

        self.configuration_performance_decay = 0.95

        # Create a data frame with the historical performance of the policies
        policy_names = [policy.name for policy in self.policies]
        self.policies_performance = pd.DataFrame(columns=["step"] + policy_names)
        self.policies_performance.set_index("step", inplace=True)

        # The last scores for each configuration
        self.policy_scores_by_configuration = np.zeros(
            (len(self.configurations), len(self.policies)), dtype=np.float64
        )

        # Also record the performance of the different configurations
        self.last_probabilities_by_configuration = {
            decay: EQUAL_PROBS for decay in self.configurations
        }
        self.configurations_performance = pd.DataFrame(
            columns=["step"] + [str(config) for config in self.configurations]
        )
        self.configurations_performance.set_index("step", inplace=True)

    def act(self) -> int:
        if len(self.history) > 0:
            # Update the historical performance for each policy and for each decay value
            self.update_performance()

        # Get the new probabilities from every policy
        policy_probs = np.array(
            [
                policy.probabilities(self.step, self.score, self.history)
                for policy in self.policies
            ]
        )

        if len(self.history) > 0:
            # Determine the performance scores of the policies for each configuration and calculate their respective weights using a dirichlet distribution
            config_probs = []
            for config_index, conf in enumerate(self.configurations):
                decay, reset_prob, clip_zero = conf
                policy_scores = self.policy_scores_by_configuration[config_index, :]
                scale = 5 / (np.sum(np.power(decay, np.arange(0, 12))))
                policy_weights = np.random.dirichlet(scale * (policy_scores - np.min(policy_scores)) + 0.1)
                # Calculate the resulting probabilities for the possible actions
                p = np.sum(
                    policy_weights.reshape((policy_weights.size, 1)) * policy_probs,
                    axis=0,
                )
                highest = (-policy_weights).argsort()[:3]
                p = 0.7 * policy_probs[highest[0]] + 0.2 * policy_probs[highest[1]] + 0.1 * policy_probs[highest[2]]
                if self.strict_agent:
                    p = one_hot(int(np.argmax(p)))
                config_probs.append(p)
                # Save the probabilities to evaluate the performance of this decay value in the next step
                self.last_probabilities_by_configuration[conf] = p
                logging.debug(
                    "Configuration " + str(conf) + " probabilities: " + str(p)
                )
            # Determine the performance scores for the different configurations and calculate their respective weights
            # Apply a decay to the historical scores
            configuration_scores = (
                self.configurations_performance
                * np.flip(
                    np.power(
                        self.configuration_performance_decay,
                        np.arange(len(self.configurations_performance)),
                    )
                ).reshape((-1, 1))
            ).sum(axis=0) * 3
            configuration_weights = np.random.dirichlet(
                configuration_scores - np.min(configuration_scores) + 0.01
            )
            for decay_index, probs in enumerate(config_probs):
                if np.min(probs) > 0.25:
                    # Don't take predictions with a high amount of uncertainty into account
                    configuration_weights[decay_index] = 0
            if np.sum(configuration_weights) > 0.2:
                configuration_weights *= 1 / np.sum(configuration_weights)
                # Select the configuration with the highest value
                probabilities = config_probs[np.argmax(configuration_weights)]
            else:
                probabilities = EQUAL_PROBS
            logging.info(
                "Statistical Policy Ensemble | Step "
                + str(self.step)
                + " | score: "
                + str(self.score)
                + " probabilities: "
                + str(probabilities)
            )

        # Play randomly for the first 100-200 steps
        if self.step < 100 + randint(0, 100):
            action = self.random.randint(0, 2)
            if self.random.randint(0, 3) == 1:
                # We don´t want our random seed to be cracked.
                action = (action + 1) % SIGNS
            return action
        if self.strict_agent:
            action = int(np.argmax(probabilities))
        else:
            action = int(np.random.choice(range(SIGNS), p=probabilities))
        return action

    def update_performance(self):
        # Determine the scores for the different actions (Win: 1, Tie: 0, Loss: -1)
        scores = [0, 0, 0]
        opponent_action = self.obs.lastOpponentAction
        scores[(opponent_action + 1) % 3] = 1
        scores[(opponent_action + 2) % 3] = -1

        # Policies
        for policy_name, policy in self.name_to_policy.items():
            # Calculate the policy´s score for the last step
            probs = policy.history[-1]
            score = np.sum(probs * scores)
            # Save the score to the performance data frame
            self.policies_performance.loc[self.step - 1, policy_name] = score

        #  Configurations
        for config_index, config in enumerate(self.configurations):
            decay, reset_prob, clip_zero = config
            # Calculate the score for the last step
            probs = self.last_probabilities_by_configuration[config]
            score = np.sum(probs * scores)
            # Apply the decay to the current score and add the new scores
            new_scores = (
                self.policy_scores_by_configuration[config_index] * decay
                + self.policies_performance.loc[self.step - 1, :].to_numpy()
            )
            # Zero clip
            if clip_zero:
                new_scores[new_scores < 0] = 0
            # Reset losing policies with a certain probability
            if reset_prob > 0:
                policy_scores = self.policies_performance.loc[
                    self.step - 1, :
                ].to_numpy()
                to_reset = np.logical_and(
                    policy_scores < -0.4,
                    new_scores > 0,
                    np.random.random(len(self.policies)) < reset_prob,
                )
                new_scores[to_reset] = 0
            self.policy_scores_by_configuration[config_index] = new_scores
            # Save the score to the performance data frame
            self.configurations_performance.loc[self.step - 1, str(config)] = score


AGENT = None


def statistical_policy_ensemble(observation, configuration) -> int:
    global AGENT
    if AGENT is None:
        AGENT = StatisticalPolicyEnsembleAgent(configuration)
    action, history = AGENT.agent(observation)
    return action
