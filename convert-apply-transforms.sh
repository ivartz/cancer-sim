#bash convert-apply-transform.sh mri lmask results <displacement>
scriptdir=$(dirname $0)
mri=$1
mriname=$(basename $mri)
lmask=$2
lmaskname=$(basename $lmask)
resdir=$3
disp=($4)

START=$(date +%s)

readarray -t subdirs < <(ls -d $resdir/*/ | sort -V)

numsubdirs=${#subdirs[*]}

for ((i=0; i<$numsubdirs; ++i)); do
    
    d=${subdirs[$i]}
    
    LC_ALL=C printf -v displacement %.2f ${disp[$i]} # Format the float input to string with one decimal and always use . as the decimal separator
    
    # Interpolated positive field
    bash $scriptdir/vec-to-comp.sh ${d}interp-field-${displacement}mm.nii.gz ${d}interp-field-${displacement}mm-comp.nii.gz

    # Use interpolated positive field to displace the MRI
    antsApplyTransforms --dimensionality 3 \
                        --input $mri \
                        --reference-image $mri \
                        --output ${d}${mriname%.*.*}-aug-${displacement}mm.nii.gz \
                        --interpolation Linear \
                        --transform ${d}interp-field-${displacement}mm-comp.nii.gz \
                        --verbose 0
    # Use interpolated positive field to displace the lesion mask
    antsApplyTransforms --dimensionality 3 \
                        --input $lmask \
                        --reference-image $lmask \
                        --output ${d}${lmaskname%.*.*}-aug-${displacement}mm.nii.gz \
                        --interpolation NearestNeighbor \
                        --transform ${d}interp-field-${displacement}mm-comp.nii.gz \
                        --verbose 0
done

END=$(date +%s)
DIFF=$(( $END - $START ))
echo "Script execution time: $DIFF s"
echo "$0 done"
