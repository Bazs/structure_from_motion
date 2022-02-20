[![CI](https://github.com/Bazs/structure_from_motion/actions/workflows/python-test.yml/badge.svg?branch=main)](https://github.com/Bazs/structure_from_motion/actions)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## Structure From Motion From Scratch

Basic SFM implementation "from scratch", using only basic linear algebra support from numpy. Based on the Challenge
Question in section 4.8
in [Introduction to Autonomous Mobile Robots, Second Edition](https://mitpress.mit.edu/books/introduction-autonomous-mobile-robots-second-edition)
.

## Sources

Besides the Autonomous Mobile Robots book, I used several other sources to implement the various steps of the vision
pipeline.

### Essential Matrix Estimation and Decomposition

* [Wikipedia](https://en.wikipedia.org/wiki/Eight-point_algorithm#Normalized_algorithm)
* [University of Edinburgh](https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/MOHR_TRIGGS/node50.html)
* [Hartley, Richard; Andrew Zisserman (2004). Multiple view geometry in computer vision (2nd ed.). Cambridge, UK.](https://www.robots.ox.ac.uk/~vgg/hzbook/)
