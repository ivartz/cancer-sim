# bash copy-to-client.sh input_data results 3
indir=$1
outdir=$2
LC_ALL=C printf -v displacement %.1f $3 # Format the float input to string with one decimal and always use . as the decimal separator

# First, mount windows c drive with nomachine session

# Remove existing nifti files
# Achit
rm -v ~/Desktop/C/Users/Ivar/Desktop/temp/*.nii*
# Cube
#rm -v ~/Desktop/C/Users/ivar/Desktop/temp/*.nii*
# signatur
#rm -v ~/Desktop/root/home/ivar/Desktop/temp/*.nii*

echo "These files will be copied to Windows"

du -sh $indir/2-T1c.nii.gz \
      $outdir/warped.nii.gz \
      $input_data/3-T1c.nii.gz \
      $outdir/directional-binary-masks-max.nii.gz \
      $outdir/neg-field-${displacement}mm-ants.nii.gz \
      $outdir/interp-neg-field-${displacement}mm-ants.nii.gz \
      $indir/2-Tumormask.nii.gz \
      $outdir/gaussian.nii.gz \
      $outdir/perlin-noise.nii.gz \
      $outdir/interp-gaussian.nii.gz \
      $outdir/ellipsoid-mask.nii.gz \
      $outdir/outer-ellipsoid-mask.nii.gz \
      $outdir/interp-ellipsoid-mask.nii.gz \
      $outdir/interp-outer-ellipsoid-mask.nii.gz \
      $outdir/original-bounding-box-vector-max.nii.gz \
      $outdir/interp-bounding-box-vector-max.nii.gz \
      $indir/2-BrainExtractionMask.nii.gz

# Copy files

cp -v $indir/2-T1c.nii.gz \
      $outdir/warped.nii.gz \
      $indir/3-T1c.nii.gz \
      $outdir/directional-binary-masks-max.nii.gz \
      $outdir/neg-field-${displacement}mm-ants.nii.gz \
      $outdir/interp-neg-field-${displacement}mm-ants.nii.gz \
      $indir/2-Tumormask.nii.gz \
      $outdir/gaussian.nii.gz \
      $outdir/perlin-noise.nii.gz \
      $outdir/interp-gaussian.nii.gz \
      $outdir/ellipsoid-mask.nii.gz \
      $outdir/outer-ellipsoid-mask.nii.gz \
      $outdir/interp-ellipsoid-mask.nii.gz \
      $outdir/interp-outer-ellipsoid-mask.nii.gz \
      $outdir/original-bounding-box-vector-max.nii.gz \
      $outdir/interp-bounding-box-vector-max.nii.gz \
      $indir/2-BrainExtractionMask.nii.gz ~/Desktop/C/Users/Ivar/Desktop/temp      
      
      #$input_data/2-BrainExtractionMask.nii.gz ~/Desktop/root/home/ivar/Desktop/temp
      #2-BrainExtractionMask.nii.gz ~/Desktop/C/Users/ivar/Desktop/temp

#echo 'ITK-SNAP.exe -g 2-T1c.nii.gz \
#echo '/home/ivar/itksnap-3.8.0-20190612-Linux-gcc64/bin/itksnap -g 2-T1c.nii.gz \

echo '"/cygdrive/c/Program Files/ITK-SNAP 3.8/bin/ITK-SNAP.exe" -g 2-T1c.nii.gz \
      -o warped.nii.gz \
      3-T1c.nii.gz \
      directional-binary-masks-max.nii.gz \
      neg-field-'"${displacement}"'mm-ants.nii.gz \
      interp-neg-field-'"${displacement}"'mm-ants.nii.gz \
      2-Tumormask.nii.gz \
      gaussian.nii.gz \
      perlin-noise.nii.gz \
      interp-gaussian.nii.gz \
      ellipsoid-mask.nii.gz \
      outer-ellipsoid-mask.nii.gz \
      interp-ellipsoid-mask.nii.gz \
      interp-outer-ellipsoid-mask.nii.gz \
      original-bounding-box-vector-max.nii.gz \
      interp-bounding-box-vector-max.nii.gz \
      2-BrainExtractionMask.nii.gz &' > ~/Desktop/C/Users/Ivar/Desktop/temp/open.sh      

      #2-BrainExtractionMask.nii.gz &' > ~/Desktop/root/home/ivar/Desktop/temp/open.sh
      #2-BrainExtractionMask.nii.gz &' > ~/Desktop/C/Users/ivar/Desktop/temp/open.sh

#: '
#echo '"/cygdrive/c/Users/ivar/aliza_1.98.32/aliza.exe" \
#echo '"/cygdrive/c/Users/Ivar/aliza_1.98.18/aliza.exe" \
#echo 'aliza \

echo '"/cygdrive/c/Users/Ivar/aliza_1.98.18/aliza.exe" \
      2-T1c.nii.gz \
      warped.nii.gz \
      3-T1c.nii.gz \
      directional-binary-masks-max.nii.gz \
      neg-field-'"${displacement}"'mm-ants.nii.gz \
      interp-neg-field-'"${displacement}"'mm-ants.nii.gz \
      2-Tumormask.nii.gz \
      gaussian.nii.gz \
      perlin-noise.nii.gz \
      interp-gaussian.nii.gz \
      ellipsoid-mask.nii.gz \
      outer-ellipsoid-mask.nii.gz \
      interp-ellipsoid-mask.nii.gz \
      interp-outer-ellipsoid-mask.nii.gz \
      original-bounding-box-vector-max.nii.gz \
      interp-bounding-box-vector-max.nii.gz \
      2-BrainExtractionMask.nii.gz &' >> ~/Desktop/C/Users/Ivar/Desktop/temp/open.sh

      #2-BrainExtractionMask.nii.gz &' >> ~/Desktop/root/home/ivar/Desktop/temp/open.sh      
      #2-BrainExtractionMask.nii.gz &' >> ~/Desktop/C/Users/ivar/Desktop/temp/open.sh

#'

echo "run this command in cygwin to open the files:"
#echo "cd ~/Desktop/temp && bash open.sh"
echo "cd /cygdrive/c/Users/Ivar/Desktop/temp && bash /cygdrive/c/Users/Ivar/Desktop/temp/open.sh"
#echo "cd /cygdrive/c/Users/ivar/Desktop/temp && bash /cygdrive/c/Users/ivar/Desktop/temp/open.sh"
