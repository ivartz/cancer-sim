# bash copy-to-client.sh input_data results 3
indir=$1
outdir=$2
LC_ALL=C printf -v displacement %.1f $3 # Format the float input to string with one decimal and always use . as the decimal separator
client=$4

# First, mount windows c drive with nomachine session

if [ $client == "Achit" ]; then
    # Achit
    targetdir="/home/ivar/Desktop/C/Users/Ivar/Desktop/temp"
    itksnapbin="/cygdrive/c/Program Files/ITK-SNAP 3.8/bin/ITK-SNAP.exe"
    alizabin="/cygdrive/c/Users/Ivar/aliza_1.98.18/aliza.exe"
    runcmd="cd /cygdrive/c/Users/Ivar/Desktop/temp && bash /cygdrive/c/Users/Ivar/Desktop/temp/open.sh"
elif [ $client == "cube" ]; then
    # cube
    targetdir="/home/ivar/Desktop/C/Users/ivar/Desktop/temp"
    itksnapbin="ITK-SNAP.exe"
    alizabin="/cygdrive/c/Users/ivar/aliza_1.98.32/aliza.exe"
    runcmd="cd /cygdrive/c/Users/ivar/Desktop/temp && bash /cygdrive/c/Users/ivar/Desktop/temp/open.sh"
elif [ $client == "signatur" ]; then
    # signatur
    targetdir="/home/ivar/Desktop/root/home/ivar/Desktop/temp"
    itksnapbin="/home/ivar/itksnap-3.8.0-20190612-Linux-gcc64/bin/itksnap"
    alizabin="aliza"
    runcmd="cd ~/Desktop/temp && bash open.sh"
fi

# Remove existing nifti files
rm -v $targetdir/*.nii*


echo "These files will be copied to the client machine:"

du -sh $indir/T1c.nii.gz \
      $outdir/warped.nii.gz \
      $outdir/directional-binary-masks-max.nii.gz \
      $outdir/neg-field-${displacement}mm-ants.nii.gz \
      $outdir/interp-neg-field-${displacement}mm-ants.nii.gz \
      $indir/Segmentation.nii.gz \
      $outdir/gaussian.nii.gz \
      $outdir/perlin-noise.nii.gz \
      $outdir/interp-gaussian.nii.gz \
      $outdir/ellipsoid-mask.nii.gz \
      $outdir/outer-ellipsoid-mask.nii.gz \
      $outdir/interp-ellipsoid-mask.nii.gz \
      $outdir/interp-outer-ellipsoid-mask.nii.gz \
      $outdir/original-bounding-box-vector-max.nii.gz \
      $outdir/interp-bounding-box-vector-max.nii.gz \
      $indir/BrainExtractionMask.nii.gz

# Copy files
cp -v $indir/T1c.nii.gz \
      $outdir/warped.nii.gz \
      $outdir/directional-binary-masks-max.nii.gz \
      $outdir/neg-field-${displacement}mm-ants.nii.gz \
      $outdir/interp-neg-field-${displacement}mm-ants.nii.gz \
      $indir/Segmentation.nii.gz \
      $outdir/gaussian.nii.gz \
      $outdir/perlin-noise.nii.gz \
      $outdir/interp-gaussian.nii.gz \
      $outdir/ellipsoid-mask.nii.gz \
      $outdir/outer-ellipsoid-mask.nii.gz \
      $outdir/interp-ellipsoid-mask.nii.gz \
      $outdir/interp-outer-ellipsoid-mask.nii.gz \
      $outdir/original-bounding-box-vector-max.nii.gz \
      $outdir/interp-bounding-box-vector-max.nii.gz \
      $indir/BrainExtractionMask.nii.gz $targetdir

echo '"'"$itksnapbin"'" -g T1c.nii.gz \
      -o warped.nii.gz \
      directional-binary-masks-max.nii.gz \
      neg-field-'"${displacement}"'mm-ants.nii.gz \
      interp-neg-field-'"${displacement}"'mm-ants.nii.gz \
      Segmentation.nii.gz \
      gaussian.nii.gz \
      perlin-noise.nii.gz \
      interp-gaussian.nii.gz \
      ellipsoid-mask.nii.gz \
      outer-ellipsoid-mask.nii.gz \
      interp-ellipsoid-mask.nii.gz \
      interp-outer-ellipsoid-mask.nii.gz \
      original-bounding-box-vector-max.nii.gz \
      interp-bounding-box-vector-max.nii.gz \
      BrainExtractionMask.nii.gz &' > $targetdir/open.sh 

echo '"'"$alizabin"'" \
      T1c.nii.gz \
      warped.nii.gz \
      directional-binary-masks-max.nii.gz \
      neg-field-'"${displacement}"'mm-ants.nii.gz \
      interp-neg-field-'"${displacement}"'mm-ants.nii.gz \
      Segmentation.nii.gz \
      gaussian.nii.gz \
      perlin-noise.nii.gz \
      interp-gaussian.nii.gz \
      ellipsoid-mask.nii.gz \
      outer-ellipsoid-mask.nii.gz \
      interp-ellipsoid-mask.nii.gz \
      interp-outer-ellipsoid-mask.nii.gz \
      original-bounding-box-vector-max.nii.gz \
      interp-bounding-box-vector-max.nii.gz \
      BrainExtractionMask.nii.gz &' >> $targetdir/open.sh

echo "run this command in cygwin to open the files:"
echo $runcmd

