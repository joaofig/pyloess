# pyloess
A simple implementation of the LOESS algorithm using numpy based on 
[NIST](https://www.itl.nist.gov/div898/handbook/pmd/section1/dep/dep144.htm).

The purpose of this code is to illustrate a possible implementation of the 
LOESS smoothing algorithm. The code has been fully implemented in Python, 
heavily using NumPy and vectorization.

## Using the code
To use the code simply instantiate an object of the Loess class like so:
~~~
xx = np.array([0.5578196, 2.0217271, 2.5773252, 3.4140288, 4.3014084,
                4.7448394, 5.1073781, 6.5411662, 6.7216176, 7.2600583,
                8.1335874, 9.1224379, 11.9296663, 12.3797674, 13.2728619,
                4.2767453, 15.3731026, 15.6476637, 18.5605355, 18.5866354,
                18.7572812])
yy = np.array([18.63654, 103.49646, 150.35391, 190.51031, 208.70115,
                213.71135, 228.49353, 233.55387, 234.55054, 223.89225,
                227.68339, 223.91982, 168.01999, 164.95750, 152.61107,
                160.78742, 168.55567, 152.42658, 221.70702, 222.69040,
                243.18828])

loess = Loess(xx, yy)

for x in xx:
    y = loess.estimate(x, window=7, use_matrix=False, degree=1)
    print(x, y)
~~~

### Notes
The included `TorchLoess.py` file is experimental. If you can find a better
and faster implementation using PyTorch, please let me know.

### References
[LOESS - Medium](https://medium.com/@joao.figueira/loess-373d43b03564)

[NIST](https://www.itl.nist.gov/div898/handbook/pmd/section1/dep/dep144.htm)

[LOESS - Wikipedia](https://en.wikipedia.org/wiki/Local_regression)

[Tricubic weight function](https://en.wikipedia.org/wiki/Kernel_(statistics)#Kernel_functions_in_common_use)
