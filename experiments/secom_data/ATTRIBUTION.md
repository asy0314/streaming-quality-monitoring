# SECOM dataset — source, license, and required attribution

The two files in this directory (`secom.data`, `secom_labels.data`) are **not**
original to this repository. They are redistributed unmodified from the UCI
Machine Learning Repository under the terms of that dataset's license.

* **Dataset:** SECOM (semiconductor manufacturing process)
* **Donors:** Michael McCann and Adrian Johnston (2008)
* **Source:** <https://archive.ics.uci.edu/dataset/179/secom>
* **DOI:** [10.24432/C54305](https://doi.org/10.24432/C54305)
* **License:** Creative Commons Attribution 4.0 International (**CC BY 4.0**),
  <https://creativecommons.org/licenses/by/4.0/> — "allows for the sharing and
  adaptation of the datasets for any purpose, provided that the appropriate
  credit is given."

Redistribution here is therefore permitted, and this file provides the credit
CC BY 4.0 requires. If you use these files, cite the donors:

```bibtex
@misc{secom2008,
  author    = {McCann, Michael and Johnston, Adrian},
  title     = {{SECOM}},
  year      = {2008},
  publisher = {UCI Machine Learning Repository},
  doi       = {10.24432/C54305},
  url       = {https://archive.ics.uci.edu/dataset/179/secom}
}
```

**Contents.** 1,567 production runs × 590 in-line process sensor readings, plus a
pass/fail label per run (`secom_labels.data`, column 1; column 2 is a timestamp
this project does not use). Missing readings are `NaN`. The data are anonymized
process-sensor measurements and contain no personal or personally identifiable
information.

**Modifications.** None. The files are byte-identical to the UCI distribution;
all preprocessing happens at runtime in the experiment scripts (see README §4).
