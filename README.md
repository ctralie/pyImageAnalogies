# pyImageAnalogies

A Python implementation of the basic features Image Analogies technique for style transfer ([1]), which is an incredibly simple but effective classical technique which is an alternative to complicated deep learning approaches if the example image pair is in perfect correspondence.  Features in this implementation include multiresolution pyramids and coherence search weighted against approximate nearest neighbor search over all patches.

* ([1]) A. Hertzmann, C. Jacobs, N. Oliver, B. Curless, D. Salesin. ``Image Analogies.''  SIGGRAPH 2001 Conference Proceedings.


## Dependencies
All of the dependencies below are pip installable
* Numpy/Scipy/Matplotlib
* scikit-image
* imageio
* sklearn


## Running
To see all options, run the script as follows
~~~~~ bash
python ImageAnalogies.py --help
~~~~~

Below are some examples

~~~~~ bash
python ImageAnalogies.py --A images/me-mask.png --Ap images/me.jpg --B images/cyclopsmask.png --Bp results/mecyclops.png --Kappa 0.1 --NLevels 2 
~~~~~

<table>
<tr><td><img src = "images/me-mask.png"></td><td><img src = "images/me.jpg"></td></tr>
<tr><td><img src = "images/cyclopsmask.png"></td><td><img src = "results/mecyclops.png"></td></tr>
</table>


[1]: <https://mrl.nyu.edu/projects/image-analogies/>
