# Experiment List

## ANNs

### Transfer Learning
High neped is difficult to capture (see [B.Sc.thesis](github.com/fusionby2030/bsc_thesis)) within the JET pedestal database. High neped is defined as nedped >= 0.95 x 10^{22}. Additionally, the amount of datapoints with high neped is very small, therefore the idea of transfer learning arises, where a ANN is trained firstly on low neped, then *transfered* to be able to predict on high neped. This is done by freezing all but the last layers of the ANN trained on low neped such that the knowledge of low neped is retained while small changes can be made to be able to catpure high neped.

`transfer_learning.py`
  - cd into `src`, and run `python tranfer_learning.py --help` to find all the variables you can change via CLI argument parsing


## Other
- Joint distribution split coefficient checking.
![initial KDE results ](https://github.com/fusionby2030/greedy-pillows/blob/master/src/out/splits/KDE_vs_COEF-%24\Gamma%24.png)
