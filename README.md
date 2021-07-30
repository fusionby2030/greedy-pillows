# greedy-pillows, or lets mosey down the gradient to nesep

## Current Dataset
- JET pedestal database, 3000+ shots from JET, all containing main engineering params (current, BT, uzw.,) and the pedestal profile from fit params (see below).

## Current Project
- Transfer Learning to attain high neped.
  - ANN's in `/src/transfer_learning.py`
    - Make a model trained on low neped, then freeze layers except for last layer, train again on high neped. 
- Autoencoders in `/src/vae-shenanigans/`
	- to run experiments, check `python3 /src/vae-shenanigans/run.py --help` but be warned about dependencies.
	- see `/src/vae_shenanigans/{name_TBD}.pdf`, i.e., the only pdf file in the dir

![Raw HRTS vs MTANH Fit](https://github.com/fusionby2030/greedy-pillows/blob/master/doc/images/MTANH_fit_21.png)


## Next-step ideas
2. VAE using raw HRTS data (`etc/VAE-idea-2021-06`)
  - Check nomachine for raw data for each shot
  - This is the blue data from above plot, whereas red it fit
3. Unsupervised learning
  - Randomly drop values from main engineering cols and train unsupervised model to determine them from other cols.
4. KDE separation of input variables (`etc/Vlad-idea-2021-06`)
  - into two datasets (for each input variable), fit linear models and compare their coefficients.

![initial KDE results ](https://github.com/fusionby2030/greedy-pillows/blob/master/src/out/splits/KDE_vs_COEF-%24I_P%24.png)

4. Tabnet (`etc/Chris-idea-20212-06`)
  - new fancy architectures
