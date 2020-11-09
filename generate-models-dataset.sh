# bash generate-models-dataset.sh 2>&1 | tee /mnt/HDD3TB/derivatives/cancer-sim-SAILOR_PROCESSED_MNI-01/runlog.txt

run_evals=1

# GitHub directory of cancer-sim
simdir="/mnt/HDD3TB/code/cancer-sim"

# Directory of dataset to use as input for simulation
dataset="/mnt/HDD3TB/derivatives/SAILOR_PROCESSED_MNI"

# Output directory to store generated models
outdir="/mnt/HDD3TB/derivatives/cancer-sim-SAILOR_PROCESSED_MNI-01"

# Log the repository version
#echo "https://github.com/ivartz/ants-bcond/commits/master" > $outdir/ants-bcond-version.txt
# shortened hash
#echo $(git log --pretty=format:'%h' -n 1) >> $outdir/ants-bcond-version.txt

# Make array of patient folder names
readarray -t patients < <(ls $dataset)

# Generate models for each patient based on first time point
for patient in ${patients[*]}; do
    # Make output directory
    out="$outdir/$patient"
    c="mkdir $out"
    echo $c
    if [ $run_evals == 1 ]; then
        eval $c
    fi
    # Run model cancer-sim generate models script
    # and storing output log
    c="bash $simdir/generate-models.sh $simdir/params.txt $dataset/$patient/01/T1c.nii.gz $dataset/$patient/01/Segmentation.nii.gz $dataset/$patient/01/BrainExtractionMask.nii.gz $out 2>&1 | tee $out/runlog.txt"
    echo $c
    if [ $run_evals == 1 ]; then
        eval $c
    fi
    #sleep 0.3
done


