# cancer-sim
Simulating displacement of tissue in the brain caused by tumor growth
## Versions
![cancer-sim-v1-3](https://user-images.githubusercontent.com/10455104/115448354-ebdddd00-a219-11eb-9988-fd6ad716f82d.jpg)
**Fig. 1:** Planned release schedule. Outlined tumor masks are shown for two time points (vertical). cancer-sim v1 (A) and v2 (A, B) are simplistic (by purpose) radial groth models that, when tuned, produces realistic-looking second time-point MRI examinations of either pushing tumor growth (v1, v2) or shrinking tumor from successful treatment(v2). This is accomplished by deforming first time-pont MRI using a displacement field produced by cancer-sim and linear interpolation. Version three builds upon the ideas of previous versions and produces more realistic displacement fields by using second time-point MRI and gradient-based optimization similar to non-rigid registration.
