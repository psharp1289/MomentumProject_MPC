'''
Simulate 4 datasets for momentum study:

1. moody RL in momentum environment
2. standard RL in momentum environment
3. moody RL in volatile environment (no momentum)
4. standard RL in volatile environment (no momentum)

Paul Sharp 5-5-2019

'''
from simulation_functions import *
import numpy as np
from numpy.random import gamma
from numpy.random import beta
from numpy.random import normal
import scipy as sp
import pandas as pd
import os


'''
generate simulated data

'''

# define variables
lrv=beta(2,5,75) #learning rate for expected values for each choice
lrm=beta(2,5,75) #learning rate for mood
invtemp=gamma(7,1,75) #inverse temperature for decision-making function (higher=more greedy)
moodbias=normal(2.5,0.7,75) #mood bias parameter that inflates reward based on level of mood
trials=125 #number of trials
outcome_prob_momentum=[0.1,0.3] #probability of punishment in a momentum environment
outcome_prob_volatile=[0.4,0.6] #probability of punishment in a volatile environment without momentum
outcome_val=-1 #punishment due to aversive learning task
arms=2 #number of bandits
shifts=3 #number of shifts
num_subjects=75 #number of simulated subjects to-be-generated

#Paths to types of simulations defined above
path_to_1='/home/moodyRL_momentum'
path_to_2='/home/standardRL_momentum'
path_to_3='/home/moodyRL_volatility'
path_to_4='/home/standardRL_volatility'

#iterate through subjects, create all simulated data

for subject in range(num_subjects):
    lrv_sub=lrv[subject]
    lrm_sub=lrm[subject]
    invtemp_sub=invtemp[subject]
    moodbias_sub=moodbias[subject]
    os.chdir(path_to_1)
    simulate_moodyRL(subject=subject,lrv=lrv_sub,lrm=lrm_sub,inv_temp=invtemp_sub,trials=trials,
    outcome_prob=outcome_prob_momentum,outcome_val=outcome_val,arms=arms,nShifts=shifts,
    environment='momentum',momentum_force=0.2,mood_bias=moodbias_sub)
    os.chdir(path_to_2)
    simulate_standardRL(subject=subject,lrv=lrv_sub,inv_temp=invtemp_sub,trials=trials,
    outcome_prob=outcome_prob_momentum,outcome_val=outcome_val,arms=arms,nShifts=shifts,
    environment='momentum',momentum_force=0.2)
    os.chdir(path_to_3)
    simulate_moodyRL(subject=subject,lrv=lrv_sub,lrm=lrm_sub,inv_temp=invtemp_sub,trials=trials,
    outcome_prob=outcome_prob_volatile,outcome_val=outcome_val,arms=arms,nShifts=shifts,
    environment='oscillate',momentum_force=0,mood_bias=moodbias_sub)
    os.chdir(path_to_4)
    simulate_standardRL(subject=subject,lrv=lrv_sub,inv_temp=invtemp_sub,trials=trials,
    outcome_prob=outcome_prob_volatile,outcome_val=outcome_val,arms=arms,nShifts=shifts,
    environment='oscillate',momentum_force=0.2)
