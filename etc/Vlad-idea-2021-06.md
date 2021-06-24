# Meta-modeling via joint distributions

- Compare joint distributions of main engineering params and neped/nesep
- split dataset into subsets of joint distributions using KDE
- train models on subsets
- ???
- profit
- compare
	- if bayes, how to coeff changes
	- general prediction quality
	- UQ


### Initial Results

- Remove Ptot since cross correlation
	- Remove q95 since other info already included?
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



![initial results](https://github.com/fusionby2030/greedy-pillows/blob/master/etc/joint-dist-results-23-06-21.png)

### Next Steps

- Plot Bar plots instead since above plot makes no sense if anyone were to look at it
	- coefs without split
	- Split 1 coefs vs spilt 2 for each variable 
	- Show exactly where split occurs using KDE
- Splitting 
	- automated with KDE instead of by hand 

- Bayes, could then compare KLD of the two posteriors.
- Understand what the hell it means

