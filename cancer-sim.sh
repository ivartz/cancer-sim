#bash cancer-sim <maximum radial displacement [mm]>

# How much displacement to simulate [mm]
displacement=$1

: '
rm gaussian.nii
rm ellipsoid-mask.nii
rm outer-ellipsoid-mask.nii
rm normal-ellipsoid-mask.nii
#rm directional-binary-masks.nii
rm directional-binary-masks-max.nii
rm original-bounding-box-vector-max.nii
rm interp-bounding-box-vector-max.nii
rm interp-ellipsoid-mask.nii
rm interp-outer-ellipsoid-mask.nii
rm field.nii
rm neg-field.nii
rm interp-field.nii
rm interp-neg-field.nii
rm interp-gaussian.nii
rm field-${displacement}mm.nii
rm neg-field-${displacement}mm.nii
rm interp-field-${displacement}mm.nii
rm interp-neg-field-${displacement}mm.nii
rm field-ants.nii
rm neg-field-ants.nii
rm interp-field-ants.nii
rm interp-neg-field-ants.nii
rm interp-gaussian.nii
rm warped.nii
'
# angle_thr of 20 and num_sel of 100 resulted in 100 % coverage
# angle_thr of 20 and num_sel of 50 almost resulted in 100 % coverage

# Displacement can also be filled into the model generation,
# but here 1mm (normalized displacement for isotropic voxels) is used when making the model

: '
Good set 1:
bash cancer-sim.sh 3
                               --displacement 1 \
                               --gaussian_range_one_sided 5 \
                               --max_radial_displacement_to_brainmask_fraction 0.6 \
                               --max_radial_displacement_to_outer_ellipsoid_mask_fraction 1 \
                               --num_vecs 55  \
                               --angle_thr 20 \
                               --num_splits 4 \
                               --spline_order 1 \
                               --smoothing 4 \

Good set 2:
bash cancer-sim.sh 3
                               --displacement 1 \
                               --gaussian_range_one_sided 5 \
                               --max_radial_displacement_to_brainmask_fraction 0.6 \
                               --max_radial_displacement_to_outer_ellipsoid_mask_fraction 0.9 \
                               --num_vecs 55  \
                               --angle_thr 30 \
                               --num_splits 4 \
                               --spline_order 1 \
                               --smoothing 4 \

3
bash cancer-sim.sh 3
                               --displacement 1 \
                               --gaussian_range_one_sided 5 \
                               --max_radial_displacement_to_brainmask_fraction 0.6 \
                               --max_radial_displacement_to_outer_ellipsoid_mask_fraction 0.9 \
                               --num_vecs 55  \
                               --angle_thr 30 \
                               --num_splits 4 \
                               --spline_order 1 \
                               --smoothing 4 \
4 GOOD!
bash cancer-sim.sh 6
                               --displacement 1 \
                               --gaussian_range_one_sided 5 \
                               --max_radial_displacement_to_brainmask_fraction 0.4 \
                               --max_radial_displacement_to_outer_ellipsoid_mask_fraction 0.25 \
                               --num_vecs 55  \
                               --angle_thr 20 \
                               --num_splits 4 \
                               --spline_order 1 \
                               --smoothing 4 \

Good
bash cancer-sim.sh 6
                               --displacement 1 \
                               --gaussian_range_one_sided 5 \
                               --max_radial_displacement_to_brainmask_fraction 0.4 \
                               --max_radial_displacement_to_outer_ellipsoid_mask_fraction 1 \
                               --num_vecs 55  \
                               --angle_thr 20 \
                               --num_splits 4 \
                               --spline_order 1 \
                               --smoothing 4 \

bash cancer-sim.sh 6

                               --displacement 1 \
                               --gaussian_range_one_sided 5 \
                               --max_radial_displacement_to_brainmask_fraction 0.4 \
                               --max_radial_displacement_to_outer_ellipsoid_mask_fraction 1 \
                               --num_vecs 55  \
                               --angle_thr 5 \
                               --num_splits 4 \
                               --spline_order 1 \
                               --smoothing 4 \


'

python3 cancer-displacement.py --ref 2-T1c.nii.gz \
                               --tumormask 2-Tumormask.nii.gz \
                               --brainmask 2-BrainExtractionMask.nii.gz \
                               --displacement 1 \
                               --gaussian_range_one_sided 5 \
                               --max_radial_displacement_to_brainmask_fraction 1 \
                               --max_radial_displacement_to_outer_ellipsoid_mask_fraction 1 \
                               --num_vecs 55  \
                               --angle_thr 5 \
                               --num_splits 4 \
                               --spline_order 1 \
                               --smoothing 4 \
                               --eout ellipsoid-mask.nii.gz \
                               --fout field.nii.gz

bash displace-3d.sh $displacement

bash copy-to-win-sh $displacement

: '
bash converttoantstransform.sh field.nii.gz field-ants.nii.gz
bash converttoantstransform.sh neg-field.nii.gz neg-field-ants.nii.gz
bash converttoantstransform.sh interp-field.nii.gz interp-field-ants.nii.gz
bash converttoantstransform.sh interp-neg-field.nii.gz interp-neg-field-ants.nii.gz

fslmaths field.nii.gz -mul $displacement field-${displacement}mm.nii.gz
fslmaths neg-field.nii.gz -mul $displacement neg-field-${displacement}mm.nii.gz

fslmaths interp-field.nii.gz -mul $displacement interp-field-${displacement}mm.nii.gz
#fslmaths interp-field.nii.gz -mul 5 interp-field-5mm.nii.gz

fslmaths interp-neg-field.nii.gz -mul $displacement interp-neg-field-${displacement}mm.nii.gz

bash converttoantstransform.sh field-${displacement}mm.nii.gz field-${displacement}mm-ants.nii.gz
bash converttoantstransform.sh neg-field-${displacement}mm.nii.gz neg-field-${displacement}mm-ants.nii.gz

bash converttoantstransform.sh interp-field-${displacement}mm.nii.gz interp-field-${displacement}mm-ants.nii.gz
#bash converttoantstransform.sh interp-field-5mm.nii.gz interp-field-5mm-ants.nii.gz

bash converttoantstransform.sh interp-neg-field-${displacement}mm.nii.gz interp-neg-field-${displacement}mm-ants.nii.gz
#
c="antsApplyTransforms --dimensionality 3 \
                       --input 2-T1c.nii.gz \
                       --reference-image 2-T1c.nii.gz \
                       --output warped.nii.gz \
                       --interpolation linear \
                       --transform interp-field-${displacement}mm-ants.nii.gz \
                       --verbose 1"
eval $c

#antsApplyTransforms --dimensionality 3 \
#                    --input 2-T1c.nii.gz \
#                    --reference-image 2-T1c.nii.gz \
#                    --output warped.nii.gz \
#                    --interpolation linear \
#                    --transform interp-field-5mm-ants.nii.gz \
#                    --verbose 1


gunzip -f gaussian.nii.gz
gunzip -f ellipsoid-mask.nii.gz
gunzip -f outer-ellipsoid-mask.nii.gz
gunzip -f normal-ellipsoid-mask.nii.gz
#gunzip -f directional-binary-masks.nii.gz
gunzip -f directional-binary-masks-max.nii.gz
gunzip -f original-bounding-box-vector-max.nii.gz
gunzip -f interp-bounding-box-vector-max.nii.gz
gunzip -f interp-ellipsoid-mask.nii.gz
gunzip -f interp-outer-ellipsoid-mask.nii.gz
gunzip -f field.nii.gz
gunzip -f neg-field.nii.gz
gunzip -f interp-field.nii.gz
gunzip -f interp-neg-field.nii.gz
gunzip -f rm interp-gaussian.nii.gz
gunzip -f field-${displacement}mm.nii.gz
gunzip -f neg-field-${displacement}mm.nii.gz
gunzip -f interp-field-${displacement}mm.nii.gz
gunzip -f interp-neg-field-${displacement}mm.nii.gz
#c="gunzip -f field-${displacement}mm.nii.gz"
#eval $c
#c="gunzip -f neg-field-${displacement}mm.nii.gz"
#eval $c
#c="gunzip -f interp-field-${displacement}mm.nii.gz"
#eval $c
#c="gunzip -f interp-neg-field-${displacement}mm.nii.gz"
#eval $c
gunzip -f field-ants.nii.gz
gunzip -f neg-field-ants.nii.gz
gunzip -f interp-field-ants.nii.gz
gunzip -f interp-neg-field-ants.nii.gz
gunzip -f interp-gaussian.nii.gz
gunzip -f warped.nii.gz
'
