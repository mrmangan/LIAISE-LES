# LIAISE-LES
The configuration files for the MicroHH LES simulations for the LIAISE experiment. 

The LES model is Micro-HH () using the develop-old branch last updated on 20 December 2023. Boundary conditions are doubly periodic and are forced using downscaled ERA5 data using the python package LS2D(). 

There are two cases: the offline case and the online case. The offline case has prescribed surface fluxes using the flux map products as described in Mangan et al, 2023. 
