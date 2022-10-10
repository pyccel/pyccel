---
title: 'Pyccel: a Python-to-X transpiler for scientific high-performance computing'

tags:
  - Python
  - transpiler
  - Fortran
  - C language
  - HPC
  - scientific computing

authors:

  - name: Emily Bourne
    orcid: 0000-0002-3469-2338
    equal-contrib: true
    affiliation: 1

  - name: Yaman Güçlü
    orcid: 0000-0003-2619-5152
    equal-contrib: true
    corresponding: true
    affiliation: 2

  - name: Said Hadjout
    orcid:
    equal-contrib: true
    affiliation: "2, 3"

  - name: Ahmed Ratnani
    orcid: 0000-0001-9035-1231
    equal-contrib: true
    affiliation: 4

affiliations:

  - name: IRFM, Centre d’Energie Atomique, Cadarache, France
    index: 1

  - name: NMPP division, Max-Planck-Institut für Plasmaphysik, Garching bei München, Germany
    index: 2

  - name: Dept. of Mathematics, Technische Universität München, Garching bei München, Germany
    index: 3

  - name: Lab. MSDA, Mohammed VI Polytechnic University, Benguerir, Morocco
    index: 4

date: 10 October 2022
bibliography: paper.bib

---

# Summary

# Statement of need

Pyccel is a static compiler for Python 3, using Fortran or C as a backend language, with a focus on high-performance computing (HPC) applications.

Pyccel's main goal is to resolve the principal bottleneck in scientific computing: the transition from prototype to production. Programmers usually develop their prototype code in a user-friendly interactive language like Python, but their final application requires an HPC implementation and therefore a new production code. In most cases this is written in a statically compiled language like Fortran/C/C++, and it uses SIMD vectorization, parallel multi-threading, MPI parallelization, GPU offloading, etc.

We believe that this expensive process can be avoided, or at least drastically reduced, by using Pyccel to accelerate the most computationally intensive parts of the Python prototype. Not only is the Pyccel-generated Fortran or C code very fast, but it is human-readable; hence the expert programmer can easily profile the code on the target machine and further optimize it.

# Acknowledgments

# References
