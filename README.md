# greedy-pillows, or lets mosey down the gradient to nesep

## Current Dataset
Currently using JET pedestal database, 3000+ shots from JET, all containing main engineering params (current, BT, uzw.,) and the pedestal profile from fit params (see below).

![initial results](https://github.com/fusionby2030/greedy-pillows/blob/master/doc/images/MTANH_fit_21.png)
## Next-step ideas
1. VAE using raw HRTS data (`etc/VAE-idea-2021-06`)
  - Check nomachine for raw data for each shot
  - This is the blue data from above plot, whereas red it fit
2. Unsupervised learning
  - Randomly drop values from main engineering cols and train unsupervised model to determine them from other cols.
3. KDE separation of input variables (`etc/Vlad-idea-2021-06`)
  - into two datasets (for each input variable), fit linear models and compare their coefficients.
4. Tabnet (`etc/Chris-idea-20212-06`)
  - new fancy architectures
