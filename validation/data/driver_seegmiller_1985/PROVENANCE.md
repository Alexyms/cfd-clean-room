# Driver and Seegmiller (1985), the backward-facing step data

The three files in this directory are the experimental data the NASA
Turbulence Modeling Resource publishes for its 2D backward-facing step case
(ADR-012 source 13), fetched unchanged on 2026-10-12 for prompt 52 (ECR-002
step 7, VAL-019). They are United States government work published for this
use; the page is https://tmbwg.github.io/turbmodels/backstep_val.html (its
"Last Updated" line reads 11/18/2021).

| File | Address | SHA-256 |
|---|---|---|
| `profiles.exp.dat` | https://tmbwg.github.io/turbmodels/Backstep_validation/profiles.exp.dat | `23e8b44db52bfc36c66af6b743a4080c34f0165eb213a3db29c0914e12cf598d` |
| `cf.exp.dat` | https://tmbwg.github.io/turbmodels/Backstep_validation/cf.exp.dat | `cd9b6434b7956bc58e859de6af654cd7f891e25303babf1dd8633d6c58f74276` |
| `cp.expnew.dat` | https://tmbwg.github.io/turbmodels/Backstep_validation/cp.expnew.dat | `2b3ec9f8d8025b958d6ddcf6767daaf3c93083d45ef350a8cf9c4b46f93a882a` |

The experiment: Driver, D. M. and Seegmiller, H. L., "Features of a
Reattaching Turbulent Shear Layer in Divergent Channel Flow", AIAA Journal
23(2), 1985, pp. 163-171, https://doi.org/10.2514/3.8890. The page states
the flow conditions quoted in `validation/backstep.py`: Re_theta 5,000 at the
step, Re_H about 36,000 on the step height (D. M. Driver, personal
communication to the page's curators; earlier versions said 50,000), the
boundary layer about 1.5 H thick before the step, Mach 0.128, the straight
top wall, U_ref 44.2 m/s at the channel centre near x/H = -4, the
reattachment at x/H = 6.26 +/- 0.10 by laser oil-flow interferometry.

The files' own headers give the columns: in `profiles.exp.dat`, y is y/H
measured from the downstream floor, u and v are over U_ref, the stresses
are 1000 times their value over U_ref^2 and the triple moments 10,000 times
theirs over U_ref^3, at x/H = -4, 1, 4, 6 and 10; in `cf.exp.dat`, x is
x/H along the bottom wall with the coefficient and its error; in
`cp.expnew.dat`, x is x/H on the bottom and the top wall, `cp_orig` is
referenced to the conditions near x/H = -4 and `cp` is shifted uniformly so
that it is zero near x/H = 40 (its last bottom-wall station, x/H = 37.5,
reads exactly zero), as Eca et al. plotted it (AIAA-2009-3647,
AIAA-2007-4089, AIAA-2005-4728).

`docs/reports/probe52/data52.py` reads these files and prints the tuples
`validation/backstep.py` holds; `python data52.py check` prints the
checksums above.
