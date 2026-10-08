# BLAS Threads for the Pressure Solve

## Prediction

Prediction (orchestrator, 2026-10-08). Threaded `np.vdot` pays a fixed cost of about 0.33 ms per
call (the section 11 table) and saves per-element time only once the vector streams from memory
at more than one core's bandwidth. With one core reading about 10 GB/s, two float64 vectors of N
elements take about 1.6 N ns on one thread; break-even against 0.33 ms is a few hundred thousand
elements. Predicted: (a) at every length up to 100,000 one thread is faster; (b) the crossover, if
any, lies between 300,000 and 3,000,000 elements; (c) under the in-process limit, a 15,000-element
`vdot` costs within 20% of the `OPENBLAS_NUM_THREADS=1` figure (about 4.3 microseconds); (d) the
`val001_80x40` final faces under the limit hash identically to the step 2 baseline; (e) a 200x75
correction under the process's default environment, with the limit, has a median under 0.5 s and
within 20% of the 136 to 143 ms one-thread figures.
