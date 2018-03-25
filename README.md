# bmi203-final
Neural Net - Distinguishing binding sites of a transcription factor, RAP1

[![Build
Status](https://travis-ci.org/snow13bbc/Final-Project.svg?branch=master)](https://travis-ci.org/snow13bbc/Final-Project)

Travis - https://travis-ci.org/snow13bbc/Final-Project

Example python project with testing.

## usage

To use the package, first make a new conda environment and activate it

```
conda create -n exampleenv python=3
source activate exampleenv
```

then run

```
conda install --yes --file requirements.txt
```

to install all the dependencies in `requirements.txt`. Then the package's
main function (located in `neuralnet_final/__main__.py`) can be run as follows. There are options for this command.

```
python -m neuralnet_final
```
Options:
'-i <#>' or '--iter <#>' for number of iterations
'-a <#>' or '--alpha <#>' to set value for alpha
'-l <#>' or '--lambda <#>' to set lambda value
'-t' or '--train' to train the model (stochastic)
'-b' or '--batch' to train the model (batch gradient descent)



## testing

Testing is as simple as running

```
python -m test
```

from the root directory of this project.
