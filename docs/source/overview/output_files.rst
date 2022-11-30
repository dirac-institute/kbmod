Output Files
============

KBMoD outputs a range of information about the discovered trajectories. Each filename includes a user defined suffix, allowing user to easily save and compare files from different runs. Below we use SUFFIX to indicate the user-defined suffix.

The main file that most users will want to access is `results_SUFFIX.txt`. This file contains one line for each trajectory with the trajectory information (x pixel start, y pixel start, x velocity, y velocity), the number of observations seen, the estimated flux, and the estimated likelihood.

The full list of output files is:

* all_ps_SUFFIX.txt - All of the postage stamp images for each found trajectory.
* filtered_likes_SUFFIX.txt - The likelihood of the trajectory computed only after some observations are filtered
* lc_SUFFIX.txt - The likelihood curves for each trajectory (:math:`L = \frac{\psi}{\phi}`)
* psi_SUFFIX.txt - The psi curves. Each curve contains a list of psi values corresponding to the predicted trajectory position at that time.
* phi_SUFFIX.txt - The phi curves. Each curve contains a list of phi values corresponding to the predicted trajectory position at that time.
* ps_SUFFIX.txt - The aggregated (mean, median, etc.) postage stamp images. One for each trajectory.
* res_per_px_stats_SUFFIX.pkl - Per-pixel statistics. For each pixel, records statistics on the minimum likelihood, the maximum likelihood, and the number of results for trajectories starting at that pixel.
* results_SUFFIX.txt - The main results file including the found trajectories, their likelihoods, and fluxes.
* times_SUFFIX.txt - For each trajectory a list of the valid (unfiltered) times.

