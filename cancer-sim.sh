#bash cancer-sim.sh <ref> <tumormask> <brainmask> <disp> <grange> <idf> <vecs> <angle> <splo> <sm> <pres> <pabs> <odir>

scriptdir=$(dirname $0)
ref=$1
tmask=$2
bmask=$3
disp=$4
grange=$5
idf=$6
vecs=$7
angle=$8
splo=$9
sm=${10}
pres=${11}
pabs=${12}
odir=${13}

# Store the parameters used
echo "displacement="$disp"" > $odir/params.txt
echo "gaussian_range_one_sided=$grange" >> $odir/params.txt
echo "intensity_decay_fraction=$idf" >> $odir/params.txt
echo "num_vecs=$vecs" >> $odir/params.txt
echo "angle_thr=$angle" >> $odir/params.txt
echo "spline_order=$splo" >> $odir/params.txt
echo "smoothing_std=$sm" >> $odir/params.txt
echo "perlin_noise_res=$pres" >> $odir/params.txt
echo "perlin_noise_abs_max=$pabs" >> $odir/params.txt

python3 $scriptdir/cancer-displacement.py --ref $ref \
                               --tumormask $tmask \
                               --brainmask $bmask \
                               --displacement $disp \
                               --gaussian_range_one_sided $grange \
                               --intensity_decay_fraction $idf \
                               --num_vecs $vecs \
                               --angle_thr $angle \
                               --spline_order $splo \
                               --smoothing_std $sm \
                               --perlin_noise_res $pres \
                               --perlin_noise_abs_max $pabs \
                               --out $odir \
                               --minimal_output 1 \
                               --verbose 0

bash $scriptdir/convert-apply-transforms.sh $ref $ref $odir "$disp"
