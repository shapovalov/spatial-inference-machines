This is the code for the paper “Spatial Inference Machines” (alpha version).

INSTALLATION

The code uses the randomforest-matlab library:
http://code.google.com/p/randomforest-matlab/

Check out the subversion sources to the ./src-matlab/randomforest-matlab directory,
so that ./src-matlab/randomforest-matlab/RF_Class_C/mexClassRF_train.mexw64 is available.
If needed, recompile the MEX file using the provided scripts.
IMPORTANT: do not use the release downloads, critical bugs were fixed since then.


DEFINING CUSTOM FACTOR TYPES

In order to use the code for your own data, please extend the SteadyFactorSample class.
Please see its documentation. Extend the FactorSample directly class only if you know 
what you are doing.


REFERENCE

Please cite the following paper:
R. Shapovalov, D. Vetrov, P. Kohli. Spatial Inference Machines. CVPR 2013.
