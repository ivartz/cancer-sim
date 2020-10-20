: '
bash generate-models.sh <param-space.txt> <ref> <tumormask> <brainmask> <mdir>

,for instance:

bash generate-models.sh params.txt 2-T1c.nii.gz 2-Tumormask.nii.gz 2-BrainExtractionMask.nii.gz /mnt/HDD3TB/derivatives/cancer-sim-1 2>&1 | tee /mnt/HDD3TB/derivatives/cancer-sim-1/runlog.txt
'
params=$1
ref=$2
tmask=$3
bmask=$4
mdir=$5

# Make output model directory if not existing
mkdircmd="mkdir -p $mdir"
eval $mkdircmd

# Log the repository version
echo "https://github.com/ivartz/cancer-sim/commits/master" > $mdir/cancer-sim-version.txt
# shortened hash
echo $(git log --pretty=format:'%h' -n 1) >> $mdir/cancer-sim-version.txt

IFS="=" # Internal Field Separator, used for word splitting
while read -r param values; do
    if [ $param == "displacement" ]; then        
        #readarray -d " " displacements <<< $values # <<< redirects stdin from string. Will add trailing newline character
        #readarray -d " " displacements < <(printf %s $values) # Using process substitusion < <() will not add trailing newline character
        readarray -d " " displacement < <(echo -n $values) # Using process substitusion < <() will not add trailing newline character
        # https://tldp.org/LDP/abs/html/process-sub.html
    elif [ $param == "gaussian_range_one_sided" ]; then
        readarray -d " " gaussian_range_one_sided < <(echo -n $values)
    elif [ $param == "brain_coverage_fraction" ]; then
        readarray -d " " brain_coverage_fraction < <(echo -n $values)
    elif [ $param == "intensity_decay_fraction" ]; then
        readarray -d " " intensity_decay_fraction < <(echo -n $values)
    elif [ $param == "num_vecs" ]; then
        readarray -d " " num_vecs < <(echo -n $values)
    elif [ $param == "angle_thr" ]; then
        readarray -d " " angle_thr < <(echo -n $values)
    elif [ $param == "num_splits" ]; then
        readarray -d " " num_splits < <(echo -n $values)
    elif [ $param == "spline_order" ]; then
        readarray -d " " spline_order < <(echo -n $values)
    elif [ $param == "smoothing_std" ]; then
        readarray -d " " smoothing_std < <(echo -n $values)
    elif [ $param == "perlin_noise_res" ]; then
        readarray -d " " perlin_noise_res < <(echo -n $values)
    elif [ $param == "perlin_noise_abs_max" ]; then
        readarray -d " " perlin_noise_abs_max < <(echo -n $values)
    fi
done < $params # < creates input stream from file

verbose=0
idx=1

# Store the parameter space file in generated dataset
cp $params $mdir

# Start of file for saving model parameters
echo $(printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "idx" "disp" "grange" "bcf" "idf" "vecs" "angle" "splits" "splo" "sm" "pres" "pabs") > $mdir/params-all.txt

for disp in ${displacement[*]}; do
    if [ $verbose == 1 ]; then
        echo "1"
    fi
    for grange in ${gaussian_range_one_sided[*]}; do
        if [ $verbose == 1 ]; then
            echo "2"
        fi
        for bcf in ${brain_coverage_fraction[*]}; do
            if [ $verbose == 1 ]; then
                echo "3"
            fi
            for idf in ${intensity_decay_fraction[*]}; do
                if [ $verbose == 1 ]; then
                    echo "4"
                fi
                for vecs in ${num_vecs[*]}; do
                    if [ $verbose == 1 ]; then
                        echo "5"
                    fi
                    for angle in ${angle_thr[*]}; do
                        if [ $verbose == 1 ]; then
                            echo "6"
                        fi
                        for splits in ${num_splits[*]}; do
                            if [ $verbose == 1 ]; then
                                echo "7"
                            fi
                            for splo in ${spline_order[*]}; do
                                if [ $verbose == 1 ]; then
                                    echo "8"
                                fi
                                for sm in ${smoothing_std[*]}; do
                                    if [ $verbose == 1 ]; then
                                        echo "9"
                                    fi
                                    if [ ${#perlin_noise_abs_max[*]} == 1 ] && [ ${perlin_noise_abs_max[0]} == 0 ]; then
                                        # Skipping all combinations of perlin_noise_res
                                        # since perlin_noise_abs_max is 0 and have a length of 1
                                        if [ $verbose == 1 ]; then
                                            echo "10"
                                        fi
                                        ofname=$(printf %04d $idx)
                                        
                                        # Set parameters
                                        pres=${perlin_noise_res[0]}
                                        pabs=0
                                        
                                        # Save model parameters
                                        echo $(printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" $idx $disp $grange $bcf $idf $vecs $angle $splits $splo $sm $pres $pabs) >> $mdir/params-all.txt
                                        
                                        # Create output folder
                                        mkdir -p $mdir/$ofname
                                        
                                        runcmd=$(printf "bash cancer-sim.sh %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s" $ref $tmask $bmask $disp $grange $bcf $idf $vecs $angle $splits $splo $sm $pres $pabs $mdir/$ofname)
                                        echo $runcmd
                                        eval $runcmd
                                        
                                        idx=$((idx+1))
                                    else
                                        # More combinations
                                        for pres in ${perlin_noise_res[*]}; do
                                            if [ $verbose == 1 ]; then
                                                echo "10"
                                            fi
                                            for pabs in ${perlin_noise_abs_max[*]}; do
                                                if [ $verbose == 1 ]; then
                                                    echo "11"
                                                fi
                                                ofname=$(printf %04d $idx)
                                                
                                                # Save model parameters
                                                echo $(printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" $idx $disp $grange $bcf $idf $vecs $angle $splits $splo $sm $pres $pabs) >> $mdir/params-all.txt
                                                
                                                # Create output folder
                                                mkdir -p $mdir/$ofname
                                                
                                                runcmd=$(printf "bash cancer-sim.sh %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s" $ref $tmask $bmask $disp $grange $bcf $idf $vecs $angle $splits $splo $sm $pres $pabs $mdir/$ofname)
                                                echo $runcmd
                                                eval $runcmd
                                                
                                                idx=$((idx+1))
                                            done
                                        done
                                    fi
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

echo "All models generated"
