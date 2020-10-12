#bash displace-3d.sh results 5
resdir=$1
LC_ALL=C printf -v displacement %.1f $2 # Format the float input to string with one decimal and always use . as the decimal separator


# Interpolated positive field
bash converttoantstransform.sh $resdir/interp-field-${displacement}mm.nii.gz $resdir/interp-field-${displacement}mm-ants.nii.gz

# Interpolated negative field
bash converttoantstransform.sh $resdir/interp-neg-field-${displacement}mm.nii.gz $resdir/interp-neg-field-${displacement}mm-ants.nii.gz

# Original negative field
bash converttoantstransform.sh $resdir/neg-field-${displacement}mm.nii.gz $resdir/neg-field-${displacement}mm-ants.nii.gz

# Use interpolated positive field to displace the MRI
antsApplyTransforms --dimensionality 3 \
                    --input 2-T1c.nii.gz \
                    --reference-image 2-T1c.nii.gz \
                    --output $resdir/warped.nii.gz \
                    --interpolation linear \
                    --transform $resdir/interp-field-${displacement}mm-ants.nii.gz \
                    --verbose 1
: '
aliza 2-T1c.nii.gz \
      3-T1c.nii.gz \
      warped.nii.gz \
      neg-field-${displacement}mm-ants.nii.gz \
      interp-neg-field-${displacement}mm-ants.nii.gz \
      directional-binary-masks-max.nii.gz \
      original-bounding-box-vector-max.nii.gz \
      interp-bounding-box-vector-max.nii.gz &
'
