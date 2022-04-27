#bash cancer-sim-fulloutput.sh <mris> <lesionmask> <lesionmaskval> <brainmask> <disp> <grange> <idf> <vecs> <angle> <splo> <sm> <pres> <pabs> <odir>

scriptdir=$(dirname $0)
mris=($1)
lmask=$2
lmaskval=$3
bmask=$4
disp=$5
grange=$6
idf=$7
vecs=$8
angle=$9
splo=${10}
sm=${11}
pres=${12}
pabs=${13}
odir=${14}
#: '
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
#'
python3 $scriptdir/cancer-displacement.py --lesionmask $lmask \
                               --lesionmask_value $lmaskval \
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
                               --minimal_output 0 \
                               --verbose 0

bash $scriptdir/convert-apply-transforms.sh "${mri[*]}" $lmask $odir "$disp"
