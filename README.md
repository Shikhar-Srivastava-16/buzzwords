# buzzwords

## Python venv requirements

You may need to reset the vitual environment. Important to note that venv is currently trying to access python at /usr/bin/local. This may need to be changed or you can duse your own venv. 

## PC requirements

### Operating System 
The current version uses `PyTorch` and was developed on `AlmaLinux`. Now, this means that most mainstream `Linux` distros will support this project. It should not be too hard to get this to work on windows, but like all good computer scientists, _*Just Use Linux*_.

### Programming Language specifics
This deployment was done using `Python v3.9.21` and it should run on all subsequent versions.  
It uses the `pip` package manager.

### Graphics Processing
While this was developed on a machine with no GPU _(boo, St A's School of CS)_ it should be able to run on device that supports `CUDA 11.8` as a computation platform.

### CPU
This was developed on a 12-core machine running x86_64 ISA on the Intel(R) Core(TM) i5-8400 CPU @ 2.80GHz chipset. None of this should actually affect your results in running this program

For ideal results, run this program on a platform that fulfills these requirements. However, Python and PyTorch are very forgiving platforms, so it should work on your machine _(no guarantees)_. If not, it should be easy to re-jig it to work on your machine.

# References

## Deep Learning
https://developer.ibm.com/articles/cc-machine-learning-deep-learning-architectures/                             : On architectures
https://arxiv.org/abs/2410.01201                                                                                : On RNNs vs Transformers
https://rbcborealis.com/research-blogs/minimal-lstms-and-grus-simple-efficient-and-fully-parallelizable/        : On minGRU
https://news.ycombinator.com/item?id=41734344                                                                   : On building minGRU