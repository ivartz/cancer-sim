#bash convert-apply-transform.sh inimg refimg results <displacement>
input=$1
ref=$2
resdir=$3
LC_ALL=C printf -v displacement %.1f $4 # Format the float input to string with one decimal and always use . as the decimal separator

START=$(date +%s)

# Interpolated positive field
bash converttoantstransform.sh $resdir/interp-field-${displacement}mm.nii.gz $resdir/interp-field-${displacement}mm-ants.nii.gz

# Interpolated negative field
bash converttoantstransform.sh $resdir/interp-neg-field-${displacement}mm.nii.gz $resdir/interp-neg-field-${displacement}mm-ants.nii.gz

# Original negative field
bash converttoantstransform.sh $resdir/neg-field-${displacement}mm.nii.gz $resdir/neg-field-${displacement}mm-ants.nii.gz

# Use interpolated positive field to displace the MRI
antsApplyTransforms --dimensionality 3 \
                    --input $input \
                    --reference-image $ref \
                    --output $resdir/warped.nii.gz \
                    --interpolation linear \
                    --transform $resdir/interp-field-${displacement}mm-ants.nii.gz \
                    --verbose 0

END=$(date +%s)
DIFF=$(( $END - $START ))
echo "Script execution time: $DIFF s"
echo "$0 done"
