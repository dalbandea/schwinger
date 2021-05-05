## Original code

This is a code implementing a Hybrid-Monte Carlo (HMC) algorithm for
the 2 dimensional Schwinger model with N_f=2 fermions. 

Originally it is based on a code developed by Nils Christian, wich
Pavel Buividovic and Carsten Urbach have simplyfied and adopted for a
summer school in Dubna, Russia. 


## `dalbandea` fork

This fork adds several folders with different implementations of the
original code in `qed/`:
- `qed-openmp/`: adds CPU paralelllization with OpenMP, mainly for the
use with `no_timescales=1` (defined in `qed.c`).
- `qed-openmp-notimescales1`: simplifies `hmc.c` for the use of
`no_timescales=1`.
- `qed-winding-notimescales1`: adds winding function in `hmc.c`.
- `qed-winding-openmp-notimescales1`: adds winding to the parallelization
with OpenMP.
- `qed-windingN-HMC1-openmp-notimescales1`: implementation of algorithm
windingN-HMC1, which performs N winding steps + 1 HMC step.
- `qed-quenched`: removes fermions.
- `qed-quenched-openacc`: adds GPU parallelization with OpenACC.
