# correlations_and_bursts
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

The [AllenSDK](https://github.com/AllenInstitute/AllenSDK) is a software package meant to process and analyze data in the [Allen Brain Atlas](http://brain-map.org/). This project specifically handles the data from the [Neuropixels](https://allensdk.readthedocs.io/en/latest/visual_coding_neuropixels.html) project. The [AllenSDK](https://github.com/AllenInstitute/AllenSDK) package provides powerful tools to navigate the data, but does not have tools to detect burst events in spike trains. Therefore, this package provides tools to
1. Detect burst events in spike trains
2. Separate the detected "burst-train" from the whole spike train
3. Assess the information held in burst-trains and non-burst-trains compared to the whole spike train

## Organization
The package is organized around the `SessionProcessor` object. `SessionProcessor` is the primary tool for training and assessing Scikit-learn compatible decoders (using whole spike trains, burst-trains, and non-burst-trains) and assessing correlations between neuronal activity and decoder weights.

`SessionNavigator` provides tools for finding Neuropixels sessions with specific experiment criteria.

### Burst detection
`SessionProcessor` does not find or isolate activity bursts itself. To isolate burst-trains, use the [burst detection script provided.](https://github.com/mwblackburn/correlations_and_bursts/blob/main/scripts/Burst%20Detection.ipynb) The outputs of the script are direct inputs for `SessionProcessor`.

## Tutorials
[Script controlling tutorial file paths](https://github.com/mwblackburn/correlations_and_bursts/blob/main/scripts/path_entry.py)

[Tutorial for using the package *without* incorperating burst analysis](https://github.com/mwblackburn/correlations_and_bursts/blob/main/scripts/example_analysis_functionalconnectivity.ipynb)

[Tutorial for using the package with burst trains](https://github.com/mwblackburn/correlations_and_bursts/blob/main/scripts/example_analysis_with_bursts.ipynb)

[Example showing detected bursts](https://github.com/mwblackburn/correlations_and_bursts/blob/main/scripts/burst_detection_figure.ipynb)

[Example comparison between decoding accuracy of burst-trains and whole-trains](https://github.com/mwblackburn/correlations_and_bursts/blob/main/scripts/decoding_comparison_figure.ipynb)

[Example of noise correlation analysis between burst-trains and whole-trains](https://github.com/mwblackburn/correlations_and_bursts/blob/main/scripts/correlation_characterization_figures.ipynb)

## Installation

### Spike train (and burst-train) analyisis
Installing and running the .py (burst analysis) components of this package should be as simple as downloading the package and [creating an anaconda environment from the provided .yml file.](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)

### Burst-train detection and separation
To integrate the burst detection code, you need R with some pretty specific packages,
and rpy2 (2.9.5). This can be finicky, and it may take some work to get rpy2 to function properly with R.
1) Download R and Rstudio using anaconda. DO NOT INSTALL ANY R PACKAGES YET
2) [Download rpy2 using anaconda](https://anaconda.zendesk.com/hc/en-us/articles/360023857134-Setting-up-rpy2-on-Windows)

3) In python, run:

```py
>>> import rpy2.rinterface
>>> from rpy2.robjects.packages import importr
>>> base = importr('base')
>>> print(base._libPaths())
```
Only one path should show up, note it. I will refer to this path as `PATH`

4) Open Rstudio as an administrator and run the commands in [r_admin_setup_instructions.txt](https://github.com/mwblackburn/correlations_and_bursts/blob/main/docs/r_admin_setup_instructions.txt).
    The first command is .libPaths(). That should return multiple paths, including `PATH`.
    Then run the command .libPaths(`PATH`). That will set the package installation library to the same library that python is looking in, allowing you to install the packages needed in R to run the burst detection algorithms.

Useful links for additional help:

[RPY2 not finding package](https://stackoverflow.com/questions/28367799/rpy2-not-finding-package)

[Get and set directory path of installed packages in r](https://statisticsglobe.com/get-and-set-directory-path-of-installed-packages-in-r)

[R documentation on package installation](https://www.rdocumentation.org/packages/utils/versions/3.6.2/topics/install.packages)

[Running R as an administrator](https://stackoverflow.com/questions/45316742/run-r-as-administrator)

[Import errors with robjects](https://stackoverflow.com/questions/39347782/getting-segmentation-fault-core-dumped-error-while-importing-robjects-from-rpy2/53639407#53639407)

Finally, the package refered to as "sjemea" is found [here](https://github.com/sje30/sjemea)

## Citations and References
    Ellese Cotterill, Paul Charlesworth, Christopher W. Thomas, Ole
    Paulsen, Stephen J. Eglen
    J. Neurophysiol. (2016).


[Link to Cotterill et al.](http://jn.physiology.org/content/116/2/306)

rpy2 Burst detection script written by [Peter Ledochowitsch](https://alleninstitute.org/what-we-do/brain-science/about/team/staff-profiles/peter-ledochowitsch/)

## Authors
Marcus Blackburn, advised by Gabriel K. Ocker, PhD