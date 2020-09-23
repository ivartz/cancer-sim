# bash converttoantstransform.sh field.nii.gz field-ants.nii.gz

infield=$1
outfield=$2
outfieldwoe=${outfield%.*.*}

splitcmd="fslsplit $infield $outfieldwoe"
echo $splitcmd
eval $splitcmd

convcmd="c3d ${outfieldwoe}0000.nii.gz ${outfieldwoe}0001.nii.gz ${outfieldwoe}0002.nii.gz -omc 3 $outfield"
echo $convcmd
eval $convcmd

rmcmd="rm ${outfieldwoe}0000.nii.gz"
echo $rmcmd
eval $rmcmd
rmcmd="rm ${outfieldwoe}0001.nii.gz"
echo $rmcmd
eval $rmcmd
rmcmd="rm ${outfieldwoe}0002.nii.gz"
echo $rmcmd
eval $rmcmd
