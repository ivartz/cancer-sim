#bash cancer-sim <ref> <tumormask> <brainmask> <disp> <grange> <bcf> <icf> <vecs> <angle> <splits> <splo> <sm> <pres> <pabs> <odir>

ref=$1
tmask=$2
bmask=$3
disp=$4
grange=$5
bcf=$6
idf=$7
vecs=$8
angle=$9
splits=${10}
splo=${11}
sm=${12}
pres=${13}
pabs=${14}
odir=${15}

# Store the parameters used
echo "displacement=$disp" > $odir/params.txt
echo "gaussian_range_one_sided=$grange" >> $odir/params.txt
echo "brain_coverage_fraction=$bcf" >> $odir/params.txt
echo "intensity_decay_fraction=$idf" >> $odir/params.txt
echo "num_vecs=$vecs" >> $odir/params.txt
echo "angle_thr=$angle" >> $odir/params.txt
echo "num_splits=$splits" >> $odir/params.txt
echo "spline_order=$splo" >> $odir/params.txt
echo "smoothing_std=$sm" >> $odir/params.txt
echo "perlin_noise_res=$pres" >> $odir/params.txt
echo "perlin_noise_abs_max=$pabs" >> $odir/params.txt

python3 cancer-displacement.py --ref $ref \
                               --tumormask $tmask \
                               --brainmask $bmask \
                               --displacement $disp \
                               --gaussian_range_one_sided $grange \
                               --brain_coverage_fraction $bcf \
                               --intensity_decay_fraction $idf \
                               --num_vecs $vecs \
                               --angle_thr $angle \
                               --num_splits $splits \
                               --spline_order $splo \
                               --smoothing_std $sm \
                               --perlin_noise_res $pres \
                               --perlin_noise_abs_max $pabs \
                               --out $odir \
                               --verbose 0

bash displace-3d.sh $ref $ref $odir $disp
