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

### Setup
- Remove Ptot from inputs since cross correlation
	- TODO: Remove q95 since other info already included?
- Standard Scaling (0, 1), in order to keep relative importance
- Splits are determined via KDE (see `src/joint_split_exp`)

### Results


![Initial KDE vs Coefs. results for plasma current](https://github.com/fusionby2030/greedy-pillows/blob/master/src/out/splits/KDE_vs_COEF-%24I_P%24.png)


![Initial KDE vs Coefs. results for minor radius](https://github.com/fusionby2030/greedy-pillows/blob/master/src/out/splits/KDE_vs_COEF-%24a%24.png)

More figures found in `src/out/splits/`

### Next Steps

- Choose a different point in the KDE, instead of maximum, maybe the minimum?
- Bayes, could then compare KLD of the two posteriors.
- Understand what the hell it means
