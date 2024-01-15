# Bright-banana-states
Scientific project related to the article https://arxiv.org/abs/2311.18395

# Abstract of the article
Non-Gaussian quantum states, described by negative valued Wigner functions, are important both for fundamental tests of quantum physics and for emerging quantum information technologies. One of the promising ways of generation of the non-Gaussian states is the use of the cubic (Kerr) optical non-linearity, which produces the characteristic banana-like shape of the resulting quantum states. However, the Kerr effect in highly transparent optical materials is weak. Therefore, big number of the photons in the optical mode (n≳10^6) is necessary to generate an observable non-Gaussianity. In this case, the direct approach to calculation of the Wigner function becomes extremely computationally expensive.

In this work, we develop quick algorithms for computing the Husimi and Wigner quasiprobability functions of these non-Gaussin states by means of the Kerr nonlinearity. This algorithm can be used for any realistic values of the photons number and the non-linearity. 

# Structure of the project
The main functions of interest are along the paths Husimi.husimi.husimi and Wigner.wigner.wigner. The algorithm of their work completely follows what is described in the article.

The Steepest_descent folder contains the functions necessary to visualize the curves of the steepest descent. They take an arbitrary analytical function as an argument, which makes them useful in analyzing the pattern of constant phase curves not only in this project.

The Graphs folder contains the code that generates all the graphs used in the article. Unlike the functions mentioned above, the code here is not so elegant) 

The normalize_check file is dedicated to verifying that the generated quasi-probability distribution is normalized by one. This condition is indeed satisfied with an accuracy of the order of accuracy of numerical integration.

Testing of most functions is hidden in the same file where they are declared in the block эif '\_\_name\_\_=="\_\_main\_\_"'. I know it's not very good, but it's very convenient.
