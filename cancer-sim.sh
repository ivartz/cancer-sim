#bash cancer-sim <ref> <tumormask> <brainmask> <disp> <grange> <mradb> <mrade> <vecs> <angle> <splits> <splo> <sm> <pres> <pabs> <odir>

ref=$1
tmask=$2
bmask=$3
disp=$4
grange=$5
mradb=$6
mrade=$7
vecs=$8
angle=$9
splits=${10}
splo=${11}
sm=${12}
pres=${13}
pabs=${14}
odir=${15}

python3 cancer-displacement.py --ref $ref \
                               --tumormask $tmask \
                               --brainmask $bmask \
                               --displacement $disp \
                               --gaussian_range_one_sided $grange \
                               --max_radial_displacement_to_brainmask_fraction $mradb \
                               --max_radial_displacement_to_outer_ellipsoid_mask_fraction $mrade \
                               --num_vecs $vecs \
                               --angle_thr $angle \
                               --num_splits $splits \
                               --spline_order $splo \
                               --smoothing_std $sm \
                               --perlin_noise_res $pres \
                               --perlin_noise_abs_max $pabs \
                               --out $odir \
                               --verbose 0

bash displace-3d.sh $odir $disp

bash copy-to-client.sh $odir $disp
