''' Simulation Functions for Momentum Study 2019

Written by Paul B. Sharp 5/5/2019'''

from scipy.misc import logsumexp
import numpy as np
from numpy import exp
from numpy import tanh
import pandas as pd
from numpy.random import multinomial as sample_probs
import csv
import os

#simulate data with standard RL in a given environment

# INPUTS:
    # lrv = learning rate for expected values
    # inv_temp = inverse temperature for softmax
    # trials = number of trials in bandit order_task
    # outcome_prob = beginning probabilities for obtaining each reward/punishment
    # outcome_val = value of reward/punishment one will recieve
    # arms = number of arms
    # nShifts = number of times probabilities shift
    # environment can take on 4 STRING values:
    #   a. stable - probabilities stay the same
    #   b. momentum - positive or negative momentum
    #   c. oscillate - vascilates between probabilities
    #   d. random=add random volatility to each option
    # momentum_force = how much momentum to add each contingency shift, default=0

# OUTPUTS
    # .csv file is written with subject data:
        # Column 1 is the choice
        # Column 2 is the outcome

def simulate_standardRL(subject,lrv,inv_temp,trials,outcome_prob,outcome_val,arms,nShifts,environment,momentum_force=0):
    csv_file=[]
    values=np.zeros(arms) #initial values at 0
    shifts = np.linspace(0,trials,nShifts-2).round() #list of when to shift probabilites if environment is volatile
    for trial in range(1,trials+1):
        csv_line=[]
        softmax_probs= exp(inv_temp*values- logsumexp(inv_temp*values)) #choice softmax equation to determine which choice is likely to be made
        choice=sample_probs(1,softmax_probs) # returns array with 1 at choice and 0 at non-choice
        choice_index=np.where(choice==1)[0][0] #get index of which reward was chosen
        outcome=np.random.binomial(1,outcome_prob[choice_index])*(outcome_val) #sample from reward probability of option chosen to generate reward or punishment
        values[choice_index]=values[choice_index]+(lrv*(outcome-values[choice_index])) #update value according to prediction error * learning lrate
        csv_line.append(choice_index+1) #since index starts at 0
        csv_line.append(outcome)
        csv_file.append(csv_line)

        if trial in shifts:
            print(trial)
            if environment != 'stable':
                if environment=='oscillate':
                    temp1=outcome_prob[0]
                    temp2=outcome_prob[len(outcome_prob)-1]
                    outcome_prob[0]=temp2
                    outcome_prob[len(outcome_prob)-1]=temp1
                elif environment=='momentum':
                    outcome_prob=[x+momentum_force for x in outcome_prob]
    with open('{}_simulated_data_{}_env_standardRL.csv'.format(subject,environment), 'a') as f:
        w=csv.writer(f)
        w.writerows(csv_file)


#simulate data with moody RL in a given environment

# INPUTS:
    # lrv = learning rate for expected values
    # lrm = learning rate for mood bias
    # mood_bias = weights how much mood biases perception of outcomes
    # inv_temp = inverse temperature for softmax
    # trials = number of trials in bandit order_task
    # outcome_prob = beginning probabilities for obtaining each reward/punishment
    # outcome_val = value of reward/punishment one will recieve
    # arms = number of arms
    # nShifts = number of times probabilities shift
    # environment can take on 4 STRING values:
    #   a. stable - probabilities stay the same
    #   b. momentum - positive or negative momentum (i.e., get better or get worse over time, respectively)
    #   c. oscillate - contingencies oscillate
    #   d. random=add random volatility to each option
    # momentum_force = how much momentum to add each contingency shift, default=0

# OUTPUTS
    # .csv file is written with subject data:
        # Column 1 is the choice
        # Column 2 is the outcome



def simulate_moodyRL(subject,lrv,lrm,inv_temp,trials,outcome_prob,
outcome_val,arms,nShifts,environment,momentum_force=0,mood_bias=1):
    csv_file=[]
    values=np.zeros(arms) #initial values at 0
    shifts = np.linspace(0,trials,nShifts+2).round() #list of when to shift probabilites if environment is volatile
    mood=0
    for trial in range(1,trials+1):
        csv_line=[]
        softmax_probs= exp(inv_temp*values- logsumexp(inv_temp*values)) #choice softmax equation to determine which choice is likely to be made
        choice=sample_probs(1,softmax_probs) # returns array with 1 at choice and 0 at non-choice
        choice_index=np.where(choice==1)[0][0] #get index of which reward was chosen
        outcome=np.random.binomial(1,outcome_prob[choice_index])*(outcome_val) #sample from reward probability of option chosen to generate reward or punishment
        mood_sigmoid=tanh(mood) #restrict mood to [-1,1] via sigmoid function
        perceived_outcome=outcome+(mood_bias*mood_sigmoid) #compute perceived outcome biased additively by current mood
        prediction_error=perceived_outcome-values[choice_index]
        values[choice_index]=values[choice_index]+(lrv*prediction_error) #update value according to prediction error * learning lrate
        mood=mood+lrm*(prediction_error-mood)
        csv_line.append(choice_index+1) #since index starts at 0
        csv_line.append(outcome)
        csv_file.append(csv_line)
        if trial in shifts:
            if environment != 'stable':
                if environment=='oscillate':
                    temp1=outcome_prob[0]
                    temp2=outcome_prob[len(outcome_prob)-1]
                    outcome_prob[0]=temp2
                    outcome_prob[len(outcome_prob)-1]=temp1
                elif environment=='momentum':
                    outcome_prob=[x+momentum_force for x in outcome_prob]
    with open('{}_simulated_data_ENV{}_mooodyRL.csv'.format(subject,environment), 'a') as f:
        w=csv.writer(f)
        w.writerows(csv_file)
