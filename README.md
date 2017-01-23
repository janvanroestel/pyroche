# roche.py
roche lobe calculator written in python. The implementation is based on 
"A calculator for Roche lobe properties" by Denis A. Leahy and Janet C. Leahy
DOI: 10.1186/s40668-015-0008-8

Usage
==============
the main function you want to use is 'get_radii'. This calculates the different
radii for the roche lobe of the primary(!) star: the L1, front, back, y, z, 
volume and Eggleton radii.


About the code
==============
Note that this code is not tested properly so use it at your own risk. A few
quick comparisons with the results from the fortran code by Leahy and Leahy
show identical results in the test cases. Problems can occur for high values 
for q. This is likely caused by the rootfinding near the L1 point. If you get 
an error f(a) and f(b) should have the opposite sign; you can try decreasing 
the value for xatol in the get_radius function.

ToDo
==============
* properly test the code
* have some kind of global accuracy setting
* add eccentricity?


contact
=======
if you have questions:
    j.vanroestel@astro.ru.nl
