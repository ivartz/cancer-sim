# cancer-sim

Robust radial deformation model to simulate overall displacement of tissue in the brain caused by tumor growth or treatment. Check out this [interactive example](https://cancer-sim.com/) using the radial deformation model with [grid search](https://github.com/ivartz/cancer-sim-search) to approximate tissue displacement in longitudinal MRI of glioblastoma.

*Latest version: 2*
![cancer-sim-v1-3](https://user-images.githubusercontent.com/10455104/115448354-ebdddd00-a219-11eb-9988-fd6ad716f82d.jpg)

**Fig. 1:** Description and release schedule. Outlined tumor masks are shown for two time points (vertical). cancer-sim v1 (A) and v2 (A, B) are robust radial growth models that, when tuned, produces realistic-looking second time-point MRI examinations of either pushing tumor growth (v1, v2) or shrinking tumor from successful treatment (v2). This is accomplished by deforming first time-pont MRI using a displacement field produced by cancer-sim and linear interpolation. Version three (not yet available) builds upon the ideas of previous versions and produces more realistic displacement fields by using second time-point MRI and gradient-based optimization similar to non-rigid registration.

## Requirements

The lesionmask and brainmask input nifti files need to be stored in LPI voxel order, see [1](https://andysbrainbook.readthedocs.io/en/latest/FrequentlyAskedQuestions/FrequentlyAskedQuestions.html) and [2](http://www.grahamwideman.com/gw/brain/orientation/orientterms.htm).

## How to run

- To create a single displacement field based on a brain and lesion mask, follow printed instructions from `python3 cancer-displacement.py --help`
- To find best fit model parameters on longitudinal data, follow instructions from [the grid search repository](https://github.com/ivartz/cancer-sim-search)

## Parameters (v1, v2)

- **Maximum tissue displacement [mm]**: The largest tissue displacement produced, which is the scaled magnitude of vectors normal to the ellipsis in **Fig. 1** A and B.
- **Infiltration [0-1]**: The extent of brain coverage, or smoothess of the displacement field in and outside of tumoral regions.
- **Irregularity <0,1]**: The granularity of Perlin noise added to displacements to simulate irregularity of tumoral displacements.

## Use cases

#### Benchmarking non-rigid registration methods on tumor MRI

Version 1 and 2 can be used to deform MRIs in tumor regions mimicking pathology or treatment changes, and thereby create synthetic second time-point MRIs with associated ground truth displacement fields. This data can be used to measure how well a non-rigid registration method produces the simulated ground truth displacement field.
![fig](https://user-images.githubusercontent.com/10455104/115455401-35cac100-a222-11eb-8813-8221582f8c1d.PNG)

**Fig. 2:** Comparing the displacement field from the radial growth model (v1) with the estimated field from [ANTs SyN](https://github.com/ANTsX/ANTs) (on the post-contrast T1-weighted MRI pair) reveals inconsistencies in displacement estimation in regions with poor textural features (such as necrosis).

#### Quantifying maximum tissue displacement, infiltration and growth irregularity on real data

Having a pair of structural MRIs and lesion mask, describe the structural change according to the three parameters, by using a [grid search extension](https://github.com/ivartz/cancer-sim-search) to fit the cancer-sim (v1 or v2) model.
