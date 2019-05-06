# MomentumProject_MPC

Empirical test if humans employ moody RL (see Eldar et al., 2016; Eldar and Niv, 2015) when environment has aversive momentum (i.e., things are getting worse). Uses a 2-armed bandit in two conditions: one with volatility without momentum, and one with aversive momentum. Both conditions have the same degree of volatility (absolute value of contingency changes is the same across each contingency shift). The only difference is one environment has a direction to such volatility (momentum condition) whereas the other oscillates between one option being better than the other.

Included are general tools that might be useful for those beginning their training in computational modeling:
(1) Simulating agents in various bandit environments
(2) Basic logistic regression analyses on simulated and real data
(3) Adapted scripts to employ model comparison and parameter fitting.
