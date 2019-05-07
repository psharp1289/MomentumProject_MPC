'''

logistic regressions for momentum study on 2-armed aversive learning bandit task

Paul Sharp 05-2019

logistic model: y = b*X+e

y: 1 if participant switched behavior on current trial relative
to behavior on previous trial, 0 if stayed with the same behavior

X: design matrix contains (1) an intercept (2) outcome on the last trial,
(3) tally of #outcomes into past for non-chosen rock

INPUT: .csv files with behavioral data, including what the participant chose
Column 1: (option 1 or 2 on a 2-armed bandit)
Column 2: the outcome (-1 for shock, 0 for no shock)

 '''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from sklearn.linear_model import LogisticRegression as logistic_reg
import statsmodels.api as sm
from math import isnan

path_to_m1_data='/home/paulsharp/Documents/projects/Eran_MomentumModelAnxiousMood_2018/Behavioral_Data/analyses_and_data/simulations/moodyRL_momentum'
path_to_m2_data='/home/paulsharp/Documents/projects/Eran_MomentumModelAnxiousMood_2018/Behavioral_Data/analyses_and_data/simulations/standardRL_volatility'
path_to_output='/home/paulsharp/Documents/projects/Eran_MomentumModelAnxiousMood_2018/Behavioral_Data/analyses_and_data/'
data_type='simulated'
m1_algorithm='moodyRL'
m1_environment='Momentum'
m2_algorithm='standardRL'
m2_environment='Volatility'


# Possible paths to simulated data
# home/paulsharp/Documents/projects/Eran_MomentumModelAnxiousMood_2018/Behavioral_Data/analyses_and_data/simulations/moodyRL_momentum
# home/paulsharp/Documents/projects/Eran_MomentumModelAnxiousMood_2018/Behavioral_Data/analyses_and_data/simulations/moodyRL_volatility
# home/paulsharp/Documents/projects/Eran_MomentumModelAnxiousMood_2018/Behavioral_Data/analyses_and_data/simulations/standardRL_momentum
# home/paulsharp/Documents/projects/Eran_MomentumModelAnxiousMood_2018/Behavioral_Data/analyses_and_data/simulations/standardRL_volatility


failed_to_converge_fitting_regression_subject=[['subjects']]

''' Get subjects from momentum and volatile conditions (m1 = momentum, m2=volatile) '''
subs_m1=[sub for sub in os.listdir(path_to_m1_data) if sub.endswith('.csv')]
subs_m2=[sub for sub in os.listdir(path_to_m2_data) if sub.endswith('.csv')]
intercept=np.ones(125)

other_dict={2:1,1:2} #get identity of other rock from current choice

''' Run tests on subjects in mood 1 --- with momentum '''

number_switches_m1_all_subs=[]
number_shocks_mcurren1_all_subs=[]
regression_results_momentum_tallies=[['intercept','other_choice','last_choice','interaction']]
regression_results_momentum_tallies_both=[['intercept','other_tally','chosen_tally','interaction']]
std_errors_momentum=[['intercept','other_choice','last_choice','interaction']]
tallies_past=5


for sub in subs_m1:
	last_outcome=[0]
	chosen_tally=[0]
	other_choice_tally=[0,0]
	os.chdir(path_to_m1_data)
	df_sub_m1=pd.read_csv(sub,header=None)
	df_sub_m1[1]=df_sub_m1[1]*(-1) #shocks now coded as 1's
	df_sub_m1['intercept']=intercept
	stay_switch_list=[0]
	#compute stay-switch variable
	for row in range(1,len(df_sub_m1)):
		if df_sub_m1[0][row]!=df_sub_m1[0][row-1]:
			stay_switch_list.append(1) #switched choice from previous trial
		else:
			stay_switch_list.append(0)
	stay_switch_array=np.asarray(stay_switch_list)
	df_sub_m1['stay_switch']=stay_switch_array
	#create last outcome variable without looping, add as variable to dataframe
	last_outcome=last_outcome+(df_sub_m1[1][0:-1]).tolist()
	last_outcome_array=np.asarray(last_outcome)
	df_sub_m1['last_outcome']=last_outcome_array
	#print("{}".format(df_sub_m1))
    #fill last choice tally for first N outcomes -- N defined by tallies variable
	for row in range(1,tallies_past):
		current_chosen_tally=0
		most_recent_chosen=df_sub_m1[0][row-1]
		for past in range(row):
			if df_sub_m1[0][past]==most_recent_chosen:
				current_chosen_tally+=(df_sub_m1[1][past])
		chosen_tally.append(current_chosen_tally)
	#fill last choice tally from outcome N and onwards
	for row in range(tallies_past,len(df_sub_m1)):
		current_chosen_tally=0
		most_recent_chosen=df_sub_m1[0][row-1]
		for past in range(1,tallies_past+1):
			if df_sub_m1[0][row-past]==most_recent_chosen:
				current_chosen_tally=current_chosen_tally+(df_sub_m1[1][row-past])
		chosen_tally.append(current_chosen_tally)
	# np.asarray changes list to numpy array
	chosen_tally_array=np.asarray(chosen_tally)
	df_sub_m1['chosen_tally']=chosen_tally_array
	#fill other choice tally for first N outcomes -- N defined by tallies variable
	for row in range(2,tallies_past):
		current_other_tally=0
		most_recent_other=other_dict[df_sub_m1[0][row-1]]
		for past in range(row-1):
			if df_sub_m1[0][past]==most_recent_other:
				current_other_tally+=(df_sub_m1[1][past])
		other_choice_tally.append(current_other_tally)
	#fill other choice tally from outcome N and onwards
	for row in range(tallies_past,len(df_sub_m1)):
		current_other_tally=0
		most_recent_other=other_dict[df_sub_m1[0][row-1]]
		for past in range(1,tallies_past+1):
			if df_sub_m1[0][row-past]==most_recent_other:
				current_other_tally+=(df_sub_m1[1][row-past])
		other_choice_tally.append(current_other_tally)
	other_choice_tally_array=np.asarray(other_choice_tally)
	df_sub_m1['other_choice_tally']=other_choice_tally_array
	#create mean cenetered variables and interaction
	df_sub_m1['last_outcome_mean_centered']=df_sub_m1['last_outcome']-df_sub_m1['last_outcome'].mean()
	df_sub_m1['other_choice_tally_mean_centered']=df_sub_m1['other_choice_tally']-df_sub_m1['other_choice_tally'].mean()
	df_sub_m1['chosen_tally_mean_centered']=df_sub_m1['chosen_tally']-df_sub_m1['chosen_tally'].mean()
	df_sub_m1['other_choice_tally_by_last_choice_interaction']=df_sub_m1['other_choice_tally_mean_centered']*df_sub_m1['last_outcome_mean_centered']
	df_sub_m1['other_choice_tally_by_chosen_tally_interaction']=df_sub_m1['other_choice_tally_mean_centered']*df_sub_m1['chosen_tally_mean_centered']
	os.chdir(path_to_m1_data)

    #conduct logistic regression on model 1 with chosen and oher-choice tallies
	stay_switch_outcomes=df_sub_m1['stay_switch']
	IV_matrix_tallies_both=df_sub_m1.loc[:,['intercept','other_choice_tally_mean_centered','chosen_tally_mean_centered','other_choice_tally_by_chosen_tally_interaction']]
	try:
		skip=0
		logit = sm.Logit(stay_switch_outcomes,IV_matrix_tallies_both)
		result = logit.fit(maxiter=1000)
		std_errors=result.bse.tolist()
		for item in std_errors:
			if item>10:
				skip=1
			elif isnan(item):
				skip=1
		beta_weights=result.params.tolist()
		for beta in beta_weights:
				if beta>10 or beta <-10:
					skip=1
		if skip==0:
			regression_results_momentum_tallies_both.append(beta_weights)
	except:
		failed_to_converge_fitting_regression_subject.append([sub])

    #conduct logistic regression on model 2 with last-ouctome (non-tally) and other-choice tally
	IV_matrix_tallies=df_sub_m1.loc[:,['intercept','other_choice_tally_mean_centered','last_outcome_mean_centered',
	'other_choice_tally_by_last_choice_interaction']]
	try:
		skip=0
		logit = sm.Logit(stay_switch_outcomes,IV_matrix_tallies)
		result = logit.fit(maxiter=1000)
		std_errors=result.bse.tolist()
		for item in std_errors:
			if item>10:
				skip=1
			elif isnan(item):
				skip=1
		beta_weights=result.params.tolist()
		for beta in beta_weights:
				if beta>10 or beta <-10:
					skip=1
		if skip==0:
			regression_results_momentum_tallies.append(beta_weights)
			std_errors_momentum.append(std_errors)
	except:
		failed_to_converge_fitting_regression_subject.append([sub])

#dataframe of all beta weights for fitted logistic regressions per subject
df_beta_weights_momentum_tallies=pd.DataFrame(regression_results_momentum_tallies[1:],columns=regression_results_momentum_tallies[0])
df_beta_weights_momentum_tallies_both=pd.DataFrame(regression_results_momentum_tallies_both[1:],columns=regression_results_momentum_tallies_both[0])
df_std_errors_momentum=pd.DataFrame(std_errors_momentum[1:],columns=std_errors_momentum[0])

os.chdir(path_to_output)

#median T-scores for each regressor across all subjects in momentum condition
df_T_scores_momentum=df_beta_weights_momentum_tallies.median()/df_std_errors_momentum.median()
df_T_scores_momentum.to_csv('Momentum_TScores_{}_{}.csv'.format(tallies_past,m1_algorithm))


''' Run tests on subjects in mood 2 --- with volatility '''
number_switches_m2_all_subs=[]
number_shocks_m2_all_subs=[]
regression_results_volatility_tallies=[['intercept','other_choice_tally','last_choice','interaction']]
regression_results_volatility_tallies_both=[['intercept','other_choice_tally','chosen_tally','interaction']]
std_errors_volatility=[['intercept','other_choice_tally','last_choice','interaction']]



for sub in subs_m2:
	last_outcome=[0]
	chosen_tally=[0]
	other_choice_tally=[0,0]
	os.chdir(path_to_m2_data)
	df_sub_m2=pd.read_csv(sub,header=None)
	df_sub_m2[1]=df_sub_m2[1]*(-1) #shocks now coded as 1's
	df_sub_m2['intercept']=intercept
	stay_switch_list=[0]
	regressor_outcome_given_choice=[]

	#calculate stay or switch
	for row in range(1,len(df_sub_m2)):
		if df_sub_m2[0][row]!=df_sub_m2[0][row-1]:
			stay_switch_list.append(1) #switched choice from previous trial
		else:
			stay_switch_list.append(0)
	stay_switch_array=np.asarray(stay_switch_list)
	df_sub_m2['stay_switch']=stay_switch_array

    #compute last outcome
	last_outcome=last_outcome+df_sub_m2[1][0:-1].tolist()
	last_outcome_array=np.asarray(last_outcome)
	df_sub_m2['last_outcome']=last_outcome_array

	for row in range(1,tallies_past):
		current_chosen_tally=0
		most_recent_chosen=df_sub_m2[0][row-1]
		for past in range(row):
			if df_sub_m2[0][past]==most_recent_chosen:
				current_chosen_tally+=(df_sub_m2[1][past])
		chosen_tally.append(current_chosen_tally)

	for row in range(tallies_past,len(df_sub_m2)):
		current_chosen_tally=0
		most_recent_chosen=df_sub_m2[0][row-1]
		for past in range(1,tallies_past+1):
			if df_sub_m2[0][row-past]==most_recent_chosen:
				current_chosen_tally+=(df_sub_m2[1][row-past])
		chosen_tally.append(current_chosen_tally)

	chosen_tally_array=np.asarray(chosen_tally)
	df_sub_m2['chosen_tally']=chosen_tally_array

	#fill other choice tally for first 5 outcomes
	for row in range(2,tallies_past):
		current_other_tally=0
		most_recent_other=other_dict[df_sub_m2[0][row-1]]
		for past in range(row-1):
			if df_sub_m2[0][past]==most_recent_other:
				current_other_tally+=(df_sub_m2[1][past])
		other_choice_tally.append(current_other_tally)



	#fill other choice tally from outcome 6 and onwards
	for row in range(tallies_past,len(df_sub_m2)):
		current_other_tally=0
		most_recent_other=other_dict[df_sub_m2[0][row-1]]
		for past in range(1,tallies_past+1):
			if df_sub_m2[0][row-past]==most_recent_other:
				current_other_tally+=(df_sub_m2[1][row-past])
		other_choice_tally.append(current_other_tally)

	other_choice_tally_array=np.asarray(other_choice_tally)
	df_sub_m2['other_choice_tally']=other_choice_tally_array

	#create mean cenetered variables and interaction
	df_sub_m2['last_outcome_mean_centered']=df_sub_m2['last_outcome']-df_sub_m2['last_outcome'].mean()
	df_sub_m2['chosen_tally_mean_centered']=df_sub_m2['chosen_tally']-df_sub_m2['chosen_tally'].mean()
	df_sub_m2['other_choice_tally_mean_centered']=df_sub_m2['other_choice_tally']-df_sub_m2['other_choice_tally'].mean()
	df_sub_m2['other_choice_tally_by_chosen_tally_interaction']=df_sub_m2['other_choice_tally_mean_centered']*df_sub_m2['chosen_tally_mean_centered']
	df_sub_m2['other_choice_tally_by_last_choice_interaction']=df_sub_m2['other_choice_tally_mean_centered']*df_sub_m2['last_outcome_mean_centered']


	#logistic regression tallies of BOTH last chosen rock and last unchosen rock predicting staying or switching
	stay_switch_outcomes=df_sub_m2['stay_switch']
	IV_matrix_tallies_both=df_sub_m2.loc[:,['intercept','other_choice_tally_mean_centered','chosen_tally_mean_centered','other_choice_tally_by_chosen_tally_interaction']]
	try:
		skip=0
		logit = sm.Logit(stay_switch_outcomes,IV_matrix_tallies_both)
		result = logit.fit(maxiter=1000)
		std_errors=result.bse.tolist()
		for item in std_errors:
			if item>10:
				skip=1
			elif isnan(item):
				print('IS NAN')
				skip=1
		beta_weights=result.params.tolist()
		for beta in beta_weights:
				if beta>10 or beta <-10:
					skip=1
		if skip==0:
			regression_results_volatility_tallies_both.append(beta_weights)
	except:
		failed_to_converge_fitting_regression_subject.append([sub])



	IV_matrix_tallies=df_sub_m2.loc[:,['intercept','other_choice_tally_mean_centered','last_outcome_mean_centered',
	'other_choice_tally_by_last_choice_interaction']]

	try:
		skip=0
		logit = sm.Logit(stay_switch_outcomes,IV_matrix_tallies)
		result = logit.fit(maxiter=1000)
		std_errors=result.bse.tolist()
		for item in std_errors:
			if item>10:
				skip=1
			elif isnan(item):
				skip=1
		beta_weights=result.params.tolist()
		for beta in beta_weights:
				if beta>10 or beta <-10:
					skip=1
		if skip==0:
			regression_results_volatility_tallies.append(beta_weights)
			std_errors_volatility.append(std_errors)
	except:
		failed_to_converge_fitting_regression_subject.append([sub])


df_beta_weights_volatility_tallies=pd.DataFrame(regression_results_volatility_tallies[1:],columns=regression_results_volatility_tallies[0])
df_beta_weights_volatility_tallies_both=pd.DataFrame(regression_results_volatility_tallies_both[1:],columns=regression_results_volatility_tallies_both[0])
df_std_errors_volatility=pd.DataFrame(std_errors_volatility[1:],columns=std_errors_volatility[0])

os.chdir(path_to_output)
df_T_scores_volatility=df_beta_weights_volatility_tallies.median()/df_std_errors_volatility.median()
df_T_scores_volatility.to_csv('Volatility_TScores_{}_{}.csv'.format(tallies_past,m2_algorithm))

# ''' Violin plots of beta-weights in logistic regression from momentum and volatile blocks '''
#plot momentum block beta-weights model-free analysis
sns.set(style="whitegrid")
ax = sns.boxplot(data=df_beta_weights_momentum_tallies, palette="Set2")
ax.set_title('{} {} Tallies {} Data {}'.format(m1_environment,tallies_past,data_type,m1_algorithm))
plt.show()

#plot volatility block beta-weights model-free analysis
sns.set(style="whitegrid")
ax = sns.boxplot(data=df_beta_weights_volatility_tallies, palette="Set2")
ax.set_title('{} {} Tallies {} Data {}'.format(m2_environment,tallies_past,data_type,m2_algorithm))
plt.show()


# # sns.set(style="whitegrid")
# ax = sns.boxplot(data=df_beta_weights_momentum_tallies_both, palette="Set2")
# ax.set_title('Model Free Analysis Momentum {} Tallies'.format(tallies_past))
# plt.show()
#
#
# # sns.set(style="whitegrid")
# ax = sns.boxplot(data=df_beta_weights_volatility_tallies_both, palette="Set2")
# ax.set_title('Model Free Analysis Volatility {} Tallies'.format(tallies_past))
# plt.show()
