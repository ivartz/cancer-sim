#bash convert-apply-transform.sh inimg refimg results <displacement>
scriptdir=$(dirname $0)
input=$1
ref=$2
resdir=$3
disp=($4)

START=$(date +%s)

readarray -t subdirs < <(ls -d $resdir/*/ | sort)

numsubdirs=${#subdirs[*]}

for ((i=0; i<$numsubdirs; ++i)); do
    
    d=${subdirs[$i]}
    
    LC_ALL=C printf -v displacement %.2f ${disp[$i]} # Format the float input to string with one decimal and always use . as the decimal separator
    
    # Interpolated positive field
    bash $scriptdir/converttoantstransform.sh ${d}interp-field-${displacement}mm.nii.gz ${d}interp-field-${displacement}mm-ants.nii.gz

    # Interpolated negative field
    #bash $scriptdir/converttoantstransform.sh ${d}interp-neg-field-${displacement}mm.nii.gz ${d}interp-neg-field-${displacement}mm-ants.nii.gz

    # Original negative field
    #bash $scriptdir/converttoantstransform.sh ${d}neg-field-${displacement}mm.nii.gz ${d}neg-field-${displacement}mm-ants.nii.gz

    # Use interpolated positive field to displace the MRI
    
    antsApplyTransforms --dimensionality 3 \
                        --input $input \
                        --reference-image $ref \
                        --output ${d}warped.nii.gz \
                        --interpolation linear \
                        --transform ${d}interp-field-${displacement}mm-ants.nii.gz \
                        --verbose 0
done

END=$(date +%s)
DIFF=$(( $END - $START ))
echo "Script execution time: $DIFF s"
echo "$0 done"
