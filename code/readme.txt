This is a short write up to summarize how to use the stuff in this folder.

The important folders are damiens_code and slurm_files. 
In these files you only really need to worry about the stuff in good_code and good_slurm.
The other stuff were mostly things that I was developing that did not end up working out.
In good_code there are three equations that are tested: pendulum, wave, and Allen-Cahn.
The titles of the directories have extensions "dd" and "causal". The "dd" denotes that this
code breaks up the domain every level. The "causal" indicates that causal weights were used.
If you want to look at the runs I did with this code, only worry about "saved_files2" in all
the directories. Nothing else is well documented. The paths in the files are not relative, so you
will need to change them for your device.