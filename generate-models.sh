: '
bash generate-models.sh <param-space.txt> <mris> <lesionmask> <lesionval> <brainmask> <mdir> <minimal outputs>

,for instance:

bash generate-models.sh params.txt 2-T1c.nii.gz 2-Tumormask.nii.gz 1 2-BrainExtractionMask.nii.gz /mnt/HDD3TB/derivatives/cancer-sim-1 1 2>&1 | tee /mnt/HDD3TB/derivatives/cancer-sim-1/runlog.txt
'
scriptdir=$(dirname $0)
params=$1
mris=($2)
lmask=$3
lmaskval=$4
bmask=$5
mdir=$6
minimalo=$7

if [[ $minimalo -eq 1 ]]
then
    cancersimfile="cancer-sim.sh"
else
    cancersimfile="cancer-sim-fulloutput.sh"
fi

# Make output model directory if not existing
mkdircmd="mkdir -p $mdir"
eval $mkdircmd

# Copy mri image, lesion mask and brain mask to model directory
: '
cpcmd="cp $mri $mdir"
eval $cpcmd
cpcmd="cp $lmask $mdir"
eval $cpcmd
cpcmd="cp $bmask $mdir"
eval $cpcmd
'

# Log the repository version
#echo "https://github.com/ivartz/cancer-sim/commits/master" > $mdir/cancer-sim-version.txt
# shortened hash
#echo $(git --git-dir=$scriptdir/.git log --pretty=format:'%h' -n 1) >> $mdir/cancer-sim-version.txt

IFS="=" # Internal Field Separator, used for word splitting
while read -r param values; do
    readarray -d " " $param < <(echo -n $values)
done < $params # < creates input stream from file

verbose=0
idx=1

# Store the parameter space file in generated dataset
#cpcmd="cp $params $mdir"
#eval $cpcmd

# Remoe the original params file
rmcmd="rm $params"
eval $rmcmd

# Start of file for saving model parameters
#echo $(printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "idx" "grange" "idf" "vecs" "angle" "splo" "sm" "pres" "pabs") > $mdir/params-all.txt
    
for grange in ${gaussian_range_one_sided[*]}; do
    if [ $verbose == 1 ]; then
        echo "1"
    fi
    for idf in ${intensity_decay_fraction[*]}; do
        if [ $verbose == 1 ]; then
            echo "2"
        fi
        for vecs in ${num_vecs[*]}; do
            if [ $verbose == 1 ]; then
                echo "3"
            fi
            for angle in ${angle_thr[*]}; do
                if [ $verbose == 1 ]; then
                    echo "4"
                fi
                for splo in ${spline_order[*]}; do
                    if [ $verbose == 1 ]; then
                        echo "5"
                    fi
                    for sm in ${smoothing_std[*]}; do
                        if [ $verbose == 1 ]; then
                            echo "6"
                        fi
                        if [ ${#perlin_noise_abs_max[*]} == 1 ] && (( $(echo "${perlin_noise_abs_max[0]} == 0" | bc -l) )); then
                            # Skipping all combinations of perlin_noise_res
                            # since perlin_noise_abs_max is 0 and have a length of 1
                            if [ $verbose == 1 ]; then
                                echo "7"
                            fi
                            ofname=$(printf %03d $idx)
                            
                            # Set parameters
                            pres=${perlin_noise_res[0]}
                            pabs=0
                            
                            # Save model parameters
                            #echo $(printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" $idx $grange $idf $vecs $angle $splo $sm $pres $pabs) >> $mdir/params-all.txt
                            
                            # Create output folder
                            mkdir -p $mdir/$ofname
                            
                            runcmd="bash $scriptdir/$cancersimfile '${mris[*]}' $lmask $lmaskval $bmask '${displacement[*]}' $grange $idf $vecs $angle $splo $sm $pres $pabs $mdir/$ofname"

                            #echo $runcmd
                            eval $runcmd
                            
                            idx=$((idx+1))
                        else
                            # More combinations
                            #for pres in ${perlin_noise_res[*]}; do
                            for pres_i in ${!perlin_noise_res[*]}; do
                                if [ $verbose == 1 ]; then
                                    echo "8"
                                fi
                                pres=${perlin_noise_res[$pres_i]}
                                for pabs in ${perlin_noise_abs_max[*]}; do
                                    if [ $verbose == 1 ]; then
                                        echo "9"
                                    fi
                                    
                                    if (( $(echo "${perlin_noise_abs_max[0]} > 0" | bc -l) )); then
                                        ofname=$(printf %03d $idx)
                                        
                                        # Save model parameters
                                        #echo $(printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" $idx $grange $idf $vecs $angle $splo $sm $pres $pabs) >> $mdir/params-all.txt
                                        
                                        # Create output folder
                                        mkdir -p $mdir/$ofname
                                        
                                        runcmd="bash $scriptdir/$cancersimfile '${mris[*]}' $lmask $lmaskval $bmask '${displacement[*]}' $grange $idf $vecs $angle $splo $sm $pres $pabs $mdir/$ofname"
                                        #echo $runcmd
                                        eval $runcmd
                                        
                                        idx=$((idx+1))
                                        
                                    elif (( $(echo "${perlin_noise_abs_max[0]} == 0" | bc -l) )); then
                                        if (( $(echo "$pabs == 0" | bc -l) )) && [ $pres_i -gt 0 ]; then
                                            # No operation, pass
                                            :
                                        else
                                            ofname=$(printf %03d $idx)
                                            
                                            # Save model parameters
                                            #echo $(printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" $idx $grange $idf $vecs $angle $splo $sm $pres $pabs) >> $mdir/params-all.txt
                                            
                                            # Create output folder
                                            mkdir -p $mdir/$ofname
                                            
                                            runcmd="bash $scriptdir/$cancersimfile '${mris[*]}' $lmask $lmaskval $bmask '${displacement[*]}' $grange $idf $vecs $angle $splo $sm $pres $pabs $mdir/$ofname"
                                            #echo $runcmd
                                            eval $runcmd
                                            
                                            idx=$((idx+1))
                                        fi
                                    fi
                                done
                            done
                        fi
                    done
                done
            done
        done
    done
done

#echo "All models generated"
