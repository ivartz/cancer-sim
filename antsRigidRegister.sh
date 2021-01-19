: '
bash antsRigidRegister.sh 1-T1c.nii.gz 2-T1c.nii.gz 1-brainmask.nii.gz reg
'

run_evals=1
fixedimg=$1
movingimg=$2
brainmask=$3
outdir=$4
ANTsDirectory=$(dirname $(which antsRegistration))

command="mkdir -p $outdir"
#echo $command
if [ $run_evals == 1 ]; then
    eval $command
fi

command="$ANTsDirectory/antsRegistration \
    --dimensionality 3 \
    --float 0 \
    --output $outdir/transform \
    --interpolation Linear \
    --winsorize-image-intensities [0.001,0.999] \
    --use-histogram-matching 0 \
    --transform Rigid[0.1] \
    --metric MI[$fixedimg,$movingimg,1,32,Regular,0.25] \
    --convergence [100x70x50x20,1e-6,10] \
    --shrink-factors 8x4x2x1 \
    --smoothing-sigmas 3x2x1x0vox \
    -x $brainmask \
    --verbose 0"

#echo $command
if [ $run_evals == 1 ]; then
    eval $command
fi

command="$ANTsDirectory/antsApplyTransforms \
    --dimensionality 3 \
    --input $movingimg \
    --reference-image $fixedimg \
    --output $outdir/${movingimg%.*.*}-reg.nii.gz \
    --interpolation linear \
    --transform $outdir/transform0GenericAffine.mat \
    --verbose 0"

#echo $command
if [ $run_evals == 1 ]; then
    eval $command
fi
