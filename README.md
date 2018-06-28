# epta-dr2 #
Getting something together for an EPTA data release 2 "lite".
Some things to do include:
* making figures of merit for pulsar selection
* single pulsar noise analyses
* preliminary gravitational wave analysis

####  Creating a python script from the notebook ####
We write (enterprise) analysis scripts using ipython notebooks. In order to run these on a cluster (or for some other reason), you need to convert them to a python script. You can do this with the ipython notebook open: 
-> Go to the 'file' tab, click 'download as' and choose python script. 
Alternatively run the following on the command line:

```
$ jupyter-notebook [my_notebook].ipynb
```
For developers, please don't commit the resulting python scripts to the repository, since this effectively duplicates code. Any changes should be made in the original ipython notebooks.
