Release Notes
=============

Version 1.0.0 (2023-02-23)
--------------------------

Version 1.0.0 represents a full refactoring and redesign of KBMOD, including major changes to the features, functionality, and performance. Significant testing and documentation have also been added to facilitate future maintenance and collaboration. Below we detail many of the individual changes included since the v0.5.0 release.

**New Features**

* Add ability to compare results to known objects looked up from SkyBoT or JPL (`133 <https://github.com/dirac-institute/kbmod/pull/133>`_, `198 <https://github.com/dirac-institute/kbmod/pull/198>`_)
* Add JointFit functions (`135 <https://github.com/dirac-institute/kbmod/pull/135>`_)
* Add the ability to load per-image PSFs from an auxiliary file (`139 <https://github.com/dirac-institute/kbmod/pull/139>`_, `161 <https://github.com/dirac-institute/kbmod/pull/161>`_)
* Add ability to pass 8-bit or 16-bit images to GPU functions to save memory (`152 <https://github.com/dirac-institute/kbmod/pull/152>`_)
* Make external time file optional (`165 <https://github.com/dirac-institute/kbmod/pull/165>`_)
* Rework build system to make package pip installable (`169 <https://github.com/dirac-institute/kbmod/pull/169>`_)
* Add Sphinx documentation (`184 <https://github.com/dirac-institute/kbmod/pull/184>`_)
* Remove unused filtering options to simplify configuration (`197 <https://github.com/dirac-institute/kbmod/pull/197>`_)

**Testing Improvements**

* Many unit tests added (Various PRs)
* Add a diff test (`124 <https://github.com/dirac-institute/kbmod/pull/124>`_)
* Add a regression test (`144 <https://github.com/dirac-institute/kbmod/pull/144>`_)
* Add continuous integration tests (`201 <https://github.com/dirac-institute/kbmod/pull/201>`_)

**Efficiency Improvements**

* Reduce internal copies (`115 <https://github.com/dirac-institute/kbmod/pull/115>`_, `119 <https://github.com/dirac-institute/kbmod/pull/119>`_)
* Speed up growMask function (`129 <https://github.com/dirac-institute/kbmod/pull/129>`_)
* Move growMask to GPU (`153 <https://github.com/dirac-institute/kbmod/pull/153>`_)
* Improve handling of single/multi-threading in post processing filtering (`155 <https://github.com/dirac-institute/kbmod/pull/155>`_)
* Skip masking functions if there are no masking keys (`164 <https://github.com/dirac-institute/kbmod/pull/164>`_)
* Move coadded stamp generation to GPU (`179 <https://github.com/dirac-institute/kbmod/pull/179>`_, `189 <https://github.com/dirac-institute/kbmod/pull/189>`_)

**Bug Fixes**

* Update DBSCAN parameters (`116 <https://github.com/dirac-institute/kbmod/pull/116>`_)
* Fix initial value bug in GPU Max Pooling (`120 <https://github.com/dirac-institute/kbmod/pull/120>`_)
* Correctly account for number of good images in median stamp creation (`123 <https://github.com/dirac-institute/kbmod/pull/123>`_)
* Fix unwanted append when saving layered images (`136 <https://github.com/dirac-institute/kbmod/pull/136>`_)
* Fix median computation with masked images (`137 <https://github.com/dirac-institute/kbmod/pull/137>`_)
* Drop masked pixels before conducting sigmaG filtering on GPU (`142 <https://github.com/dirac-institute/kbmod/pull/142>`_, `180 <https://github.com/dirac-institute/kbmod/pull/180>`_, `181 <https://github.com/dirac-institute/kbmod/pull/181>`_)
* Reset global mask when `createGlobalMask` is created ((`164 <https://github.com/dirac-institute/kbmod/pull/164>`_)
* Correctly apply `mask_threshold` (`164 <https://github.com/dirac-institute/kbmod/pull/164>`_)
* Account for masked pixels in `createAveTemplate` and `simpleDifference` (`164 <https://github.com/dirac-institute/kbmod/pull/164>`_)
* Fix bugs in on-GPU stamp generation (`182 <https://github.com/dirac-institute/kbmod/pull/182>`_)
* Add bounds checking in `getResults` (`192 <https://github.com/dirac-institute/kbmod/pull/192>`_)
* Check for divide by zero in clustering function (`215 <https://github.com/dirac-institute/kbmod/pull/215>`_)
* Update pybind version (`217 <https://github.com/dirac-institute/kbmod/pull/217>`_)
* Ignore invalidly named files during file loading (`233 <https://github.com/dirac-institute/kbmod/pull/233>`_)
