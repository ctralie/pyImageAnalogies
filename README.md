# pyImageAnalogies

A Python implementation of the basic features Image Analogies technique for style transfer ([1]), which is an incredibly simple but effective classical technique which is an alternative to complicated deep learning approaches if the example image pair is in perfect correspondence.  Features in this implementation include multiresolution pyramids and coherence search weighted against approximate nearest neighbor search over all patches.

* ([1]) A. Hertzmann, C. Jacobs, N. Oliver, B. Curless, D. Salesin. ``Image Analogies.''  SIGGRAPH 2001 Conference Proceedings.


## Dependencies
* Numpy/Scipy/Matplotlib
* scikit-image


## Running
To see all options, run the script as follows
~~~~~ bash
python ImageAnalogies.py --help
~~~~~



[1]: <https://mrl.nyu.edu/projects/image-analogies/>
