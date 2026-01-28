The purpose of macros/ scripts is as the main entry point for population derivatives/ or as
a source for examples on how to use other scripts in this library to populate derivatives/.
In most cases macros/ calls on lower level scripts and programs from the library, like those 
in scripts/ or bin/, but regardless its purpose is high level. Every script in here should
produce derivatives/ outputs that are directly used by scripts in figures/.

Scripts use relative paths, mainly to query config.json, and are likely to break if you move 
them.
