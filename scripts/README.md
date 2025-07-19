Files in scripts/ make use of this library, but do not need to be directly integrated into 
the library because they're high level. By keeping them as stand alone scripts they can
be more easily tweaked and modified on-demand for customized analysis pipelines. They can
be used directly, but in most cases are still general purpose enough that they need to be
invoked in particular ways to regenerate the analyses of the accompanying study. macros/
contains examples of the particular invocations needed.
