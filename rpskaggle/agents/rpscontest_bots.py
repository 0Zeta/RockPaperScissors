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