#bash displace-3d.sh 5
displacement=$1

# Interpolated positive field multiplied
#fslmaths interp-field.nii.gz -mul $displacement interp-field-${displacement}mm.nii.gz
cp -v interp-field.nii.gz interp-field-${displacement}mm.nii.gz
bash converttoantstransform.sh interp-field-${displacement}mm.nii.gz interp-field-${displacement}mm-ants.nii.gz

# Interpolated negative field multiplied
#fslmaths interp-neg-field.nii.gz -mul $displacement interp-neg-field-${displacement}mm.nii.gz
cp -v interp-neg-field.nii.gz interp-neg-field-${displacement}mm.nii.gz
bash converttoantstransform.sh interp-neg-field-${displacement}mm.nii.gz interp-neg-field-${displacement}mm-ants.nii.gz

# Interpolated negative field
bash converttoantstransform.sh interp-neg-field.nii.gz interp-neg-field-ants.nii.gz

# Original negative field
#fslmaths neg-field.nii.gz -mul $displacement neg-field-${displacement}mm.nii.gz
cp -v neg-field.nii.gz neg-field-${displacement}mm.nii.gz
bash converttoantstransform.sh neg-field-${displacement}mm.nii.gz neg-field-${displacement}mm-ants.nii.gz

# Use interpolated positive field to displace the MRI
antsApplyTransforms --dimensionality 3 \
                    --input 2-T1c.nii.gz \
                    --reference-image 2-T1c.nii.gz \
                    --output warped.nii.gz \
                    --interpolation linear \
                    --transform interp-field-${displacement}mm-ants.nii.gz \
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
