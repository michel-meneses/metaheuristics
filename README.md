# Metaheuristics
A set of classical metaheuristics that solve optimization problems.

## Overview

Metaheuristics are a set of algorithms that employ some degree of randomness to **try to find optimal solutions to difficult problems** [1]. This repository holds a single Python script, which implements some classical metaheuristics and applies them to a set of optimization problems. 

The metaheuristics implemented in this repository are:

 - Hill-Climbing (HC)
 - Steepest Ascent Hill-Climbing (SAHC)
 - Steepest Ascent Hill-Climbing with Replacement (SAHCR)
 - Simulated Annealing (SA)
 - Tabu Search (TS)
 - Iterated Local Search (ILS)

Whereas the suported optimization problems are:

 - Sphere
 - Schwefels
 - Rosenbrocks
 - Rastringins

For more details about these algorithms and this whole metaheuristics subject, please refer to this [PDF file](https://cs.gmu.edu/~sean/book/metaheuristics/Essentials.pdf).

## Usage

Clone this repository:

    git clone https://github.com/michel-meneses/metaheuristics.git

Enter into the project directory:

    cd metaheuristics

Use *pip* to install the Python dependencies:

    pip install requeriments.txt

Run the main Python script:

    python main.py [problem] [algorithm]

Where:
 - Problem: `sphere`, `schwefels`, `rosenbrocks` or `rastringins`
 - Algorithm: `hc`, `sahc`, `sahcr`, `sa`, `tb` or `ils`

## Disclaimer
This repository was developed as an assignment for the *Search and Optimization* course offered by the graduation program [PROCC](https://www.sigaa.ufs.br/sigaa/public/programa/portal.jsf?id=710) during my master's degree in CS. If you are a student enrolled in this same graduation course, be aware that this repository is public, therefore it is crawlable by plagiarism detector tools. Be responsible and use this repository wisely. 😉 

## References

 1. [https://cs.gmu.edu/~sean/book/metaheuristics/Essentials.pdf](https://cs.gmu.edu/~sean/book/metaheuristics/Essentials.pdf)
 
 
## License
This repository is distributed under the terms of the [MIT License](https://opensource.org/licenses/MIT).
