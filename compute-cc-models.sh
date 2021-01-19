: '
bash compute-cc-models.sh <image.nii.gz> <cancer-sim model dir>
'

run_evals=1
im=$1
#mask=$2
mdir=$2

readarray -t models < <(find $mdir -type f -name warped.nii.gz | sort)

for model in ${models[*]}; do
    #echo $model
    mask=$(dirname $model)/interp-outer-ellipsoid-mask.nii.gz
    #echo $mask
    : '
    If splitting the output of fslcc using space as delimiter,
    and only using binary mask, the cross-correlation value
    us at column 7
    '
    cmd="fslcc -m $mask -p 10 $im $model | cut -d ' ' -f 7"
    if [ $run_evals == 1 ]; then
        eval $cmd
    fi
done
