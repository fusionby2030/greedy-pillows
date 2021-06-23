# Meta-modeling via joint distributions

- Compare joint distributions of main engineering params and neped
- split dataset into subsets of joint distributions
- train models on subsets
- ???
- profit 
- compare
	- if bayes, how to coeff changes
	- general prediction quality
	- UQ

Table ideas in pictures

### Initial Results

- Remove Ptot since cross correlation
- Standard Scaling (0, 1), in order to keep relative importance
- splits are hand made
- Current matrix shows the relative changes in coefficients between the splits for each variable
- **a** has big change when Ip is split
- **BT** changes somewhere
- Ip obviously plays a big roll
- Random Forest added

```
main_engineer = ['Ip(MA)', 'B(T)', 'a(m)', 'averagetriangularity',
							 'P_NBI(MW)', 'P_ICRH(MW)',
							 'plasmavolume(m3)', 'q95', 'gasflowrateofmainspecies1022(es)']
splits = [2.5, 2.5, 0.9, 0.322, 15, 1.1, 77.5, 3.657, 2.5]
```



![initial results](joint-dist-results-22-06-21.png)

### Next Steps

- Check RMSE difference for regressors (requires CV? can check also against the other split.)
	- Initial check makes no sense, but probably very depends on data availability
- Understand what the hell it means
