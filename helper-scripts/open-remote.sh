: '
bash open-remote.sh 0169 0 8.0
'
csimdir="Z:\code\cancer-sim"
mdir="Z:\derivatives\cancer-sim\\$1"
hasnoise=$2
disp=$3
if [ $hasnoise == 1 ]; then
"/cygdrive/c/Program Files/ITK-SNAP 3.8/bin/ITK-SNAP.exe" -g "$csimdir\2-T1c.nii.gz" \
      -o "$mdir\warped.nii.gz" \
      "$csimdir\3-T1c.nii.gz" \
      "$mdir\directional-binary-masks-max.nii.gz" \
      "$mdir\neg-field-${disp}mm-ants.nii.gz" \
      "$mdir\interp-neg-field-${disp}mm-ants.nii.gz" \
      "$csimdir\2-Tumormask.nii.gz" \
      "$mdir\gaussian.nii.gz" \
      "$mdir\perlin-noise.nii.gz" \
      "$mdir\interp-gaussian.nii.gz" \
      "$mdir\ellipsoid-mask.nii.gz" \
      "$mdir\outer-ellipsoid-mask.nii.gz" \
      "$mdir\interp-ellipsoid-mask.nii.gz" \
      "$mdir\interp-outer-ellipsoid-mask.nii.gz" \
      "$mdir\original-bounding-box-vector-max.nii.gz" \
      "$mdir\interp-bounding-box-vector-max.nii.gz" \
      "$csimdir\2-BrainExtractionMask.nii.gz" &
else
"/cygdrive/c/Program Files/ITK-SNAP 3.8/bin/ITK-SNAP.exe" -g "$csimdir\2-T1c.nii.gz" \
      -o "$mdir\warped.nii.gz" \
      "$csimdir\3-T1c.nii.gz" \
      "$mdir\directional-binary-masks-max.nii.gz" \
      "$mdir\neg-field-${disp}mm-ants.nii.gz" \
      "$mdir\interp-neg-field-${disp}mm-ants.nii.gz" \
      "$csimdir\2-Tumormask.nii.gz" \
      "$mdir\gaussian.nii.gz" \
      "$mdir\interp-gaussian.nii.gz" \
      "$mdir\ellipsoid-mask.nii.gz" \
      "$mdir\outer-ellipsoid-mask.nii.gz" \
      "$mdir\interp-ellipsoid-mask.nii.gz" \
      "$mdir\interp-outer-ellipsoid-mask.nii.gz" \
      "$mdir\original-bounding-box-vector-max.nii.gz" \
      "$mdir\interp-bounding-box-vector-max.nii.gz" \
      "$csimdir\2-BrainExtractionMask.nii.gz" &
fi
"/cygdrive/c/Users/Ivar/aliza_1.98.18/aliza.exe" \
      "$csimdir\2-T1c.nii.gz" \
      "$mdir\warped.nii.gz" \
      "$csimdir\3-T1c.nii.gz" \
      "$mdir\directional-binary-masks-max.nii.gz" \
      "$mdir\neg-field-${disp}mm-ants.nii.gz" \
      "$mdir\interp-neg-field-${disp}mm-ants.nii.gz" \
      "$csimdir\2-Tumormask.nii.gz" \
      "$mdir\gaussian.nii.gz" \
      "$mdir\perlin-noise.nii.gz" \
      "$mdir\interp-gaussian.nii.gz" \
      "$mdir\ellipsoid-mask.nii.gz" \
      "$mdir\outer-ellipsoid-mask.nii.gz" \
      "$mdir\interp-ellipsoid-mask.nii.gz" \
      "$mdir\interp-outer-ellipsoid-mask.nii.gz" \
      "$mdir\original-bounding-box-vector-max.nii.gz" \
      "$mdir\interp-bounding-box-vector-max.nii.gz" \
      "$csimdir\2-BrainExtractionMask.nii.gz" &
