import argparse
import os
import numpy as np
import nibabel as nib
import sys
from scipy.ndimage import map_coordinates, gaussian_filter, binary_dilation
import gc
import time
# Perlin noise library: https://github.com/pvigier/perlin-numpy
from perlin_numpy import generate_perlin_noise_3d#, generate_perlin_noise_3d
import warnings
warnings.filterwarnings("ignore")

def max_spread_vectors_subset(vectors, positions, num_vecs=64):
    # Select a first vector and its position as the first
    # of the num_vecs selected vectors
    seli = np.array([0])
    vectors_sel = vectors[seli]
    positions_sel = positions[seli]
    # Delete selected vector and position from array
    vectors = np.delete(vectors, seli, axis=0)
    positions = np.delete(positions, seli, axis=0)
    # Find the other vectors
    for i in range(1, num_vecs):
        # Find the next vector based on the maximum of mean of norms
        # between a candate vector and already picked vectors
        # Calculate the difference between all vectors and picked vectors
        diffs = np.expand_dims(vectors, axis=-2)-vectors_sel
        # Compute the norm of these differences
        norms = np.linalg.norm(diffs, axis=-1)
        # Compute the mean of the norms across the picked vector axis
        mean_norms = np.mean(norms, axis=-1)
        # Select the next vector to be the vector with the maximum mean
        # of norms
        maxi = np.nanargmax(mean_norms)
        v = vectors[maxi]
        p = positions[maxi]
        # Store selected vector and its position
        vectors_sel = np.append(vectors_sel, [v], axis=0)
        positions_sel = np.append(positions_sel, [p], axis=0)
        # Delete selected vector and position from array
        vectors = np.delete(vectors, maxi, axis=0)
        positions = np.delete(positions, maxi, axis=0)
    return vectors_sel, positions_sel

def gaussian_norm(x):
    return np.exp(-(x**2)/2)

def bounding_box_mask(mask):
    """
    Returns the coodinates for the geometric center
    as well as x, y, and z widths of a binary mask,
    adjusted to an even number
    """
    
    """
    # Nonzero version, interpolation execution time: 11.519090 s
    x_nonzeros = np.nonzero(np.sum(mask, axis=(1,2)))[0]
    y_nonzeros = np.nonzero(np.sum(mask, axis=(0,2)))[0]
    z_nonzeros = np.nonzero(np.sum(mask, axis=(0,1)))[0]
    x_min, x_max = x_nonzeros[0], x_nonzeros[-1]
    y_min, y_max = y_nonzeros[0], y_nonzeros[-1]
    z_min, z_max = z_nonzeros[0], z_nonzeros[-1]
    """
    
    """
    # Argwhere version 1, interpolation execution time: 10.904181 s
    x_nonzeros = np.argwhere(np.sum(mask, axis=(1,2))).squeeze()
    y_nonzeros = np.argwhere(np.sum(mask, axis=(0,2))).squeeze()
    z_nonzeros = np.argwhere(np.sum(mask, axis=(0,1))).squeeze()
    x_min, x_max = x_nonzeros[0], x_nonzeros[-1]
    y_min, y_max = y_nonzeros[0], y_nonzeros[-1]
    z_min, z_max = z_nonzeros[0], z_nonzeros[-1]
    """
    
    """
    # Argwhere version 2, interpolation execution time: 10.689049 s BEST SO FAR
    x_nonzeros = np.argwhere(np.sum(mask, axis=(1,2)))[:,0]
    y_nonzeros = np.argwhere(np.sum(mask, axis=(0,2)))[:,0]
    z_nonzeros = np.argwhere(np.sum(mask, axis=(0,1)))[:,0]
    x_min, x_max = x_nonzeros[0], x_nonzeros[-1]
    y_min, y_max = y_nonzeros[0], y_nonzeros[-1]
    z_min, z_max = z_nonzeros[0], z_nonzeros[-1]
    """
    
    #"""
    # Sum and arange version. I think this is the fastest
    m1, m2, m3 = np.sum(mask, axis=(1,2)), np.sum(mask, axis=(0,2)), np.sum(mask, axis=(0,1))
    x_nonzeros = np.arange(m1.shape[0])[m1>0]
    y_nonzeros = np.arange(m2.shape[0])[m2>0]
    z_nonzeros = np.arange(m3.shape[0])[m3>0]
    x_min, x_max = x_nonzeros[[0, -1]]
    y_min, y_max = y_nonzeros[[0, -1]]
    z_min, z_max = z_nonzeros[[0, -1]]
    #"""
    
    # Find widths
    wx = x_max-x_min
    wy = y_max-y_min
    wz = z_max-z_min
    
    # If the widths are odd, add 1 to make them even
    if wx % 2:
        wx += 1 
    if wy % 2:
        wy += 1
    if wz % 2:
        wz += 1
    cx = x_min + wx//2
    cy = y_min + wy//2
    cz = z_min + wz//2
    return (cx, cy, cz), (wx, wy, wz)

if __name__ == "__main__":
    CLI=argparse.ArgumentParser()
    CLI.add_argument(
      "--ref",
      help="input 3D nifti for reference, can be a normal static scan",
      type=str,
      default="2-T1c.nii.gz",
    )
    CLI.add_argument(
      "--tumormask",
      help="A binary mask with 1=tumor tissue, 0=healthy (outside of tumor) tissue",
      type=str,
      default="2-Tumormask.nii.gz",
    )
    CLI.add_argument(
      "--brainmask",
      help="A binary mask with 1=brain tissue, 0=outside of the brain",
      type=str,
      default="2-BrainExtractionMask.nii.gz",
    )
    CLI.add_argument(
      "--displacement",
      help="<0,large] [mm]. The maximum amount of radial displacement (in isotropic units according to --ref) to add",
      type=float,
      default=4,
    )
    CLI.add_argument(
      "--gaussian_range_one_sided",
      help="[5,8] for working conditions. The one-sided range x used for x=[-x,x] on a normal (gaussian) symmetric distribution to model tumor expansion",
      type=float,
      default=5,
    )
    CLI.add_argument(
      "--brain_coverage_fraction",
      help="<0=brain coverage of displacements is local and the largest displacements are close to the tumor ellipsoid model inflection surface, 1=displacements will cover most of the brain. Maximum radial displacements will reach (or overreach, then scaled down) the end of the brain mask]",
      type=float,
      default=1,
    )
    CLI.add_argument(
      "--intensity_decay_fraction",
      help="[0=intensities of displacements decay slowly along radial axes from the tumor ellipsoid model, 1=intensities of displacements decay rapidly along radial axes from the tumor ellipsoid model and the largest displacements are close to the tumor ellipsoid model inflection surface]. This parameter is dependent on brain_coverage_fraction",
      type=float,
      default=0.5,
    )
    CLI.add_argument(
      "--num_vecs",
      help="The number of normal vectors used to simulate the explosive spread of tissue displacement",
      type=int,
      default=64,
    )
    CLI.add_argument(
      "--angle_thr",
      help="The maximum angle (in degrees) between a normal vector on the model ellipsoid surface and field vectors allowed when determining a directional binary mask",
      type=int,
      default=7,
    )
    CLI.add_argument(
      "--num_splits",
      help="How many splits to perform of the displacement vector array before calculating directional binary masks. Internal parameter, increase if memory error",
      type=int,
      default=4,
    )
    CLI.add_argument(
      "--spline_order",
      help="Order of intepolation in scipy.ndimage.map_coordinates. 0=kNN, 1=trilinear, 2=quadratic, 3=cubic, 4 and 5",
      type=int,
      default=1,
    )
    CLI.add_argument(
      "--smoothing_std",
      help="Standard deviation of smoothing of the final intepolated x, y and z displacement",
      type=float,
      default=4,
    )    
    CLI.add_argument(
      "--perlin_noise_res",
      #help="If Gaussian noise used, standard deviation (mean=0); if perlin noise; number of periods of noise to generate along each axis (mean ~0). The noise is added to the final intepolated Gaussian and x, y and z displacement (before scaling displacements to specified max displacement, and before scaling displacements that went oustide of the brain)",
      help="<0,1]. The number of periods of noise to generate along each axis for Perlin noise. The noise is added to the final intepolated Gaussian and x, y and z displacement (before scaling displacements to specified max displacement, and before scaling displacements that went oustide of the brain)",
      type=float,
      default=0.2,
    )    
    CLI.add_argument(
      "--perlin_noise_abs_max",
      help="[0,1]. The absolute value of maximum Perlin noise to add. The noise is added to the final intepolated Gaussian and x, y and z displacement (before scaling displacements to specified max displacement, and before scaling displacements that went oustide of the brain)",
      type=float,
      default=0.05,
    )    
    CLI.add_argument(
      "--out",
      help="Output directory of script for storing result files",
      type=str,
      default="results",
    )
    CLI.add_argument(
      "--verbose",
      help="Print all messages",
      type=int,
      default=0,
    )
    args = CLI.parse_args()
        
    # Store start time for the script
    script_start_time = time.time()
    
    # Set seed for random number generator used on Perlin noise generation
    random_seed = 0
    
    # Create output dir if not existing
    if not os.path.exists(args.out):
        os.makedirs(args.out)
    
    ref_img = nib.load(args.ref)
    tumormask_img = nib.load(args.tumormask)
    brainmask_img = nib.load(args.brainmask)
    
    xsize, ysize, zsize = ref_img.shape
    
    # Find the bounding box and geometric center
    # of the tumor based on the tumor mask
    #tumorbbox_geom_center, tumorbbox_widths = \
    #bounding_box_mask(tumormask_img.get_fdata().astype(np.bool)) # TODO
    tumorbbox_geom_center, tumorbbox_widths = \
    bounding_box_mask(tumormask_img.get_fdata().astype(np.int))
    cx, cy, cz = tumorbbox_geom_center
    wx, wy, wz = tumorbbox_widths
    
    print("Tumor bounding box geometric center: ", end='')
    print(tumorbbox_geom_center)
    print("Tumor bounding box widths: ", end='')
    print(tumorbbox_widths)
    
    # - Build 3D Gaussian and calculate its gradient
    # Using range [-args.gaussian_range_one_sided,args.gaussian_range_one_sided]
    # https://en.wikipedia.org/wiki/Normal_distribution
    endx = args.gaussian_range_one_sided
    gy = gaussian_norm(np.linspace(-endx, endx, wy)).reshape((wy, 1))
    gz = gaussian_norm(np.linspace(-endx, endx, wz)).reshape((wz, 1)).T
    gx = gaussian_norm(np.linspace(-endx, endx, wx)).reshape((wx, 1, 1))
    
    # Make 3D Gaussian by multiplying together the 3 1D gaussians,
    # and using broadcasting. Before multiplication, 
    # the 1D Gaussians are scaled so that the 3D gaussian retains a shape
    # with inflection surface analog to the inflection point of the 
    # unscaled 1D Gaussian
    # The 3D Gaussian is inverted in order to have partial derivatives that
    # resemble an explosion.
    g3d = -1*((gy**(1/3))*(gz**(1/3))*(gx**(1/3)))
    
    # Create the Gaussian in the reference image space
    gaussian_data = np.zeros(ref_img.shape, dtype=np.float32)
    gaussian_data[cx-wx//2:cx+wx//2, cy-wy//2:cy+wy//2, cz-wz//2:cz+wz//2] = g3d
    
    # Save the 3D Gaussian to nifti
    gaussian_img = \
    nib.spatialimages.SpatialImage(-gaussian_data, affine=ref_img.affine, header=ref_img.header)
    nib.save(gaussian_img, args.out+"/gaussian.nii.gz")
    
    # - 3D Gaussian gradients
    # The normalized partial derivatives of the 3D Gaussian are used 
    # to model the tumoral, peritumoral and healthy displacement 
    # caused by tumor growth.
    # The normalization is done so that all gradients (displacement vectors)
    # at the surface of the inflection ellipsoid have a magnitude of 1
    g3d_dx, g3d_dy, g3d_dz = np.gradient(g3d)
    dispx3d, dispy3d, dispz3d = g3d_dx/np.max(g3d_dx), \
                                g3d_dy/np.max(g3d_dy), \
                                -1*g3d_dz/np.max(g3d_dz) # Invert operation 1
    # (Invert operation in order to have correct ANTs transformation form)
    
    # - Ellipsoid mask
    # Create a mask of an ellipsoid where maximum displacement 
    # (= a gradient magnitude of 1) occurs normal to its surface.
    # 1 is the x coordinate of the inflection point in 1D normal distribution
    ellipsoid_threshold = gaussian_norm(1)
    #ellipsoid_mask = g3d <= -ellipsoid_threshold
    ellipsoid_mask = g3d < -ellipsoid_threshold # Since g3d is inverted
        
    # Can use the mask to remove increasing displacements overlapping 
    # with it (~= tumor)
    #dispx3d[ellipsoid_mask] = 0
    #dispy3d[ellipsoid_mask] = 0
    #dispz3d[ellipsoid_mask] = 0
    
    # Create the ellipsoid mask in the reference image space
    ellipsoid_data_bbox = np.zeros(g3d.shape, dtype=np.int)
    ellipsoid_data_bbox[ellipsoid_mask] = 1
    ellipsoid_data = np.zeros(ref_img.shape, dtype=np.int)
    ellipsoid_data[cx-wx//2:cx+wx//2, cy-wy//2:cy+wy//2, cz-wz//2:cz+wz//2] = ellipsoid_data_bbox
    
    # Save the mask to nifti
    ellipsoid_img = \
    nib.spatialimages.SpatialImage(ellipsoid_data, affine=ref_img.affine, header=ref_img.header)
    #nib.save(ellipsoid_img, args.eout)
    nib.save(ellipsoid_img, args.out+"/ellipsoid-mask.nii.gz")
    
    # - Outer ellipsoid mask
    # Create a mask of an ellipsoid where its
    # surface marks approximately the end of the 3D gaussian
    # Store it in the reference image space
    outer_ellipsoid_mask = g3d <= -0.05 # This was a good default
    outer_ellipsoid_data_bbox = np.zeros(g3d.shape, dtype=np.int)
    outer_ellipsoid_data_bbox[outer_ellipsoid_mask] = 1
    outer_ellipsoid_data = np.zeros(ref_img.shape, dtype=np.int)
    outer_ellipsoid_data[cx-wx//2:cx+wx//2, cy-wy//2:cy+wy//2, cz-wz//2:cz+wz//2] = outer_ellipsoid_data_bbox
    
    # Save the mask to nifti
    outer_ellipsoid_img = \
    nib.spatialimages.SpatialImage(outer_ellipsoid_data, affine=ref_img.affine, header=ref_img.header)
    #nib.save(outer_ellipsoid_img, "outer-"+args.eout)
    nib.save(outer_ellipsoid_img, args.out+"/outer-ellipsoid-mask.nii.gz")
    
    # - Displacement field
    field_data = np.zeros(ref_img.shape+(3,), dtype=np.float32)
    
    # Printing input parameters
    print("Displacement: %f" % args.displacement)
    print("Normal Gaussian x-range: [-%f,%f]" % (-args.gaussian_range_one_sided, args.gaussian_range_one_sided))
    print("Fraction of displacement coverage in the brain: %f" % args.brain_coverage_fraction)
    print("Fraction of displacement intensity decay within the displacement coverage in the brain: %f" % args.intensity_decay_fraction)
        
    # Insert the normalized gradients of the 3D Gaussian as displacement
    # field vectors
    field_data[cx-wx//2:cx+wx//2, cy-wy//2:cy+wy//2, cz-wz//2:cz+wz//2, 0] = dispx3d
    field_data[cx-wx//2:cx+wx//2, cy-wy//2:cy+wy//2, cz-wz//2:cz+wz//2, 1] = dispy3d
    field_data[cx-wx//2:cx+wx//2, cy-wy//2:cy+wy//2, cz-wz//2:cz+wz//2, 2] = dispz3d
    
    # Compute more realistic displacement field using various extra information
    # 1. Scale displacement field using the brian mask.
    brainmask_data = brainmask_img.get_fdata()
    brainmask_data[brainmask_data != 1] = 0 # Note this mask was not binary, making it binary
    #mask_img = nib.spatialimages.SpatialImage(brainmask_data, affine=ref_img.affine, header=ref_img.header)
    #nib.save(mask_img, args.out+"/brainmask-binary.nii.gz")
    
    # Calculate the absolute value of the displacement, for using later
    dnorm = np.linalg.norm(field_data, axis=-1)
    
    # - Normal ellipsoid mask
    # Calculate the norm of the current displacement vector field
    fabs = dnorm.copy()
    # Calculate the approxmate mask for of the ellipsoid surface
    fabs[ellipsoid_data == 1] = 0
    fabs[fabs <= 0.99] = 0
    #fabs[fabs < 1] = 0
    fabs_max_mask = fabs != 0
    
    # This normal ellipsoid is the same as the inner ellipsoid, 
    # but with its volume (contents) set to 0 and only a with a thin (~ voxel)
    # surface remaining.
    # Save the normal ellipsoid mask
    normal_ellipsoid_data = np.zeros(ref_img.shape, dtype=np.int)
    normal_ellipsoid_data[fabs_max_mask] = 1
    normal_ellipsoid_img = nib.spatialimages.SpatialImage(normal_ellipsoid_data, affine=ref_img.affine, header=ref_img.header)
    #nib.save(normal_ellipsoid_img, "normal-"+args.eout)
    nib.save(normal_ellipsoid_img, args.out+"/normal-ellipsoid-mask.nii.gz")
        
    # Find the (normal) displacement vector components and coordinates
    # for normal ellipsoid mask
    normal_displacement_vectors = field_data[fabs_max_mask]
    normal_displacement_vectors_coordinates = np.argwhere(fabs_max_mask)
    
    # Save normal vectors and positions for diagnostics
    #np.savez("jp/normal-displacement-vectors.npz", normal_displacement_vectors)
    #np.savez("jp/normal-displacement-vectors-coordinates.npz", normal_displacement_vectors_coordinates)
    
    # - Use a subset of the normal vectors 
    # (that together cover most of the field)
    # to interpolate and/or displace the field according to
    # the distance to the end of brain mask.
    
    # Continue using only the num_vecs subset of the vectors
    num_vecs = args.num_vecs
    
    # Make sure num_vecs is not larger then the available number of vectors
    if num_vecs > len(normal_displacement_vectors):
        if args.verbose == 1:
            print("Number of vectors specified is larger than the available number of vectors, setting num_vecs to the number of available vectors")
        num_vecs = len(normal_displacement_vectors)
    
    # Find the subset containing these number of vectors that are the most spread
    normal_displacement_vectors, \
    normal_displacement_vectors_coordinates = \
    max_spread_vectors_subset(normal_displacement_vectors, normal_displacement_vectors_coordinates, num_vecs=num_vecs)
    
    num_normal_displacement_vectors = len(normal_displacement_vectors)
    
    print("Number of normal vectors used: %i" % num_normal_displacement_vectors)
    #print(normal_displacement_vectors.shape)
    #print(normal_displacement_vectors_coordinates.shape)
    
    # Number of splits of the normal vector array,
    # to avoid large memory footprint when calculating dot products
    #num_splits = 1 # Good if num_vecs is low
    #num_splits = 4 # Good default
    num_splits = args.num_splits
    
    print("Maximum error angle allowed for directional binary masks [degrees]: %i" % args.angle_thr)
    print("Number of splits: %i" % num_splits)
    print("Spline order for intepolation in map_coordinates: %i" % args.spline_order)
    print("Gaussian smoothing standard deviation: %f" % args.smoothing_std)
    print("Additive Perlin noise number of periods along axis: %f" % args.perlin_noise_res)
    print("Additive Perlin noise absolute of maximum noise: %f" % args.perlin_noise_abs_max)
    
    # Calculate the split size and remainder
    split_size, remaining = divmod(num_normal_displacement_vectors, num_splits)
    
    print("Split size: %i" % split_size)
    print("Remaining: %i" % remaining)
    
    # Split normal vector array into these even splits and 
    # the last one containing the remaining ones if existing
    normal_displacement_vectors_to_split = normal_displacement_vectors[:num_splits*split_size]
    normal_displacement_vectors_remaining = normal_displacement_vectors[-remaining:]
    
    # Create array to hold all directional binary masks
    #bm = np.zeros(ref_img.shape+(num_normal_displacement_vectors,), dtype=np.bool) # TODO
    bm = np.zeros(ref_img.shape+(num_normal_displacement_vectors,), dtype=np.int)# np.int fastest?
    
    print("Processing splits")
    
    if args.verbose == 0:
        # Setup progress bar
        sys.stdout.write("[%s]" % (" " * (num_splits+1)))
        sys.stdout.flush()
        sys.stdout.write("\b" * (num_splits+2)) # return to start of line, after '['
    
    # Start timer to measure the time used on the for loop and remaining vectors
    start_time = time.time()
    
    # Iterate over even batches of the normal vectors
    for split_num, v in enumerate(np.split(normal_displacement_vectors_to_split, num_splits)):
        if args.verbose == 1:
            print("Processing split %i/%i" % (split_num+1, num_splits))
        # Calculate the dot product between the field vectors and the normal vectors
        d = np.dot(field_data, v.T)
        # The product of the norms of the field vectors and normal vectors
        norms = np.expand_dims(dnorm, axis=-1)*np.linalg.norm(v, axis=-1)
        # Calculate the deviation in degress between the field vectors and normal vectors
        deviation_degrees = np.rad2deg(np.arccos(d/norms))
        # Create a boolean mask of where there is less than or equal to 20 degrees 
        # difference between field vectors and normal vectors
        #m = deviation_degrees < 20
        m = deviation_degrees <= args.angle_thr
        # Save the boolean mask as a binary mask in existing array
        # NB! Using binary dilation with default structuring element
        bm[...,split_num*split_size:(split_num+1)*split_size][m] = 1
        if args.verbose == 1:
            print("Calculated directional masks")
        if args.verbose == 0:
            # Append to progress bar
            sys.stdout.write("-")
            sys.stdout.flush()
    # Iterate over the remaining normal vectors if existing
    if remaining:
        if args.verbose == 1:
            print("Processing remainder")
        # Calculate the dot product between the field vectors and the normal vectors
        d = np.dot(field_data, normal_displacement_vectors_remaining.T)
        # The product of the norms of the field vectors and normal vectors
        norms = np.expand_dims(dnorm, axis=-1)*np.linalg.norm(normal_displacement_vectors_remaining, axis=-1)
        # Calculate the deviation in degress between the field vectors and normal vectors
        deviation_degrees = np.rad2deg(np.arccos(d/norms))
        # Create a boolean mask of where there is less than or equal to 20 degrees 
        # difference between field and normal vectors
        #m = deviation_degrees < 20
        m = deviation_degrees <= args.angle_thr
        # Save the boolean mask as a binary mask in existing array
        # NB! Using binary dilation with default structuring element
        bm[...,-remaining:][m] = 1
        if args.verbose == 1:
            print("Calculated remaining directional masks")
        if args.verbose == 0:
            # Append to progress bar
            sys.stdout.write("-")
            sys.stdout.flush()
    
    if args.verbose == 0:
        sys.stdout.write("]\n") # this ends the progress bar
    
    print("Cone computation execution time: %f s" %(time.time()-start_time))

    print("Number of directional masks calculated: %i" % bm.shape[-1])
    print("Saving directional binary masks to disk")
    """
    # Save all directional binary masks as a 4D nifti
    bm_img = nib.spatialimages.SpatialImage(bm, affine=ref_img.affine, header=ref_img.header)
    nib.save(bm_img, "directional-binary-masks.nii.gz")
    """
    # As well as the max of the masks (=union)
    bm_max_img = nib.spatialimages.SpatialImage(np.max(bm, axis=-1), affine=ref_img.affine, header=ref_img.header)
    nib.save(bm_max_img, args.out+"/directional-binary-masks-max.nii.gz")
    
    # TODO: These arrays take up a lot of memory, and sparse arrays could be used
    # instead, like:
    # https://cmdlinetips.com/2018/03/sparse-matrices-in-python-with-scipy/
    
    # Create array to hold all original bounding boxes for directional binary masks
    print("Making array for holding all bounding boxes for directional masks")
    orig_bboxes_data = np.zeros(ref_img.shape+(num_normal_displacement_vectors,), dtype=np.int)
    
    # Create array to hold all bounding boxes for intepolated directional binary masks
    print("Making array for holding all bounding boxes for intepolated directional masks")
    interp_bboxes_data = np.zeros(ref_img.shape+(num_normal_displacement_vectors,), dtype=np.int)
    
    # Make the array for holding all intepolated displacement values
    print("Making array for holding all intepolated displacement values and initialize it with nan values")
    field_data_interp = np.empty(ref_img.shape+(3,num_normal_displacement_vectors), dtype=np.float32)
    field_data_interp[:] = np.nan
    
    # Make the array for holding all intepolated gaussian values
    print("Making array for holding all intepolated Gaussian values and initialize it with nan values")
    gaussian_data_interp = np.empty(ref_img.shape+(num_normal_displacement_vectors,), dtype=np.float32)
    gaussian_data_interp[:] = np.nan
    
    # Split the components of the displacement field into three 
    # scalar fields
    print("Splitting non-intepolated vector field array into three scalar arrays")
    dx, dy, dz = field_data[...,0], field_data[...,1], field_data[...,2]

    print("Processing vectors")
    
    if args.verbose == 0:
        # Setup progress bar
        sys.stdout.write("[%s]" % (" " * num_normal_displacement_vectors))
        sys.stdout.flush()
        sys.stdout.write("\b" * (num_normal_displacement_vectors+1)) # return to start of line, after '['
    
    # Start timer to measure the time used on the for loop
    start_time = time.time()
    # Iterate over each normal vector, that we just created
    # directional binary masks for
    #for i in range(2, num_normal_displacement_vectors-1): # when num_vecs = 4 (for debug)
    for i in range(num_normal_displacement_vectors):
        if args.verbose == 1:
            print("Processing vector: %i/%i" % (i+1, num_normal_displacement_vectors))
        # Get the coordinates of the normal vector
        nv_c = normal_displacement_vectors_coordinates[i]
        if args.verbose == 1:
            print("Position: ", end='')
            print(nv_c)
        # Get the normal vector
        nv_d = normal_displacement_vectors[i]
        nv_d[-1] *= -1 # Invert z component. This was necesary in order to 
        # find correct end of brain mask maximum displacement. The reason for -1 is
        # because dispz3d contains a -1 term from before 
        # (in order to have correct ANTs transformation form). Invert operation 2
        
        # Normalize the normal vector, to be ensure that it has exactly unit length
        # (int should already have approximately unit length)
        nv_d = nv_d/np.linalg.norm(nv_d)
        
        if args.verbose == 1:
            print("Displacement: ", end='')
            print(nv_d)
        
        # Get the directional binary mask corresponding to this
        # normal vector
        bmi = bm[...,i]
        
        # For debuging cone computations
        #np.savez("jp/cone-mask.npz", bmi)
        #np.savez("jp/cone-vectors.npz", field_data[bmi == 1])
        #np.savez("jp/cone-vectors-positions.npz", np.argwhere(bmi))
        
        # Find the maximum displacement possible along the nv_d vector 
        # between its starting position and the end of the original
        # directional binary mask
        n = 1
        # TODO: Can introduce step size in order to have more accurate displacements
        # as for now, this is equal to a step size of 1 (n*step_size = 1 here).
        # Start by displacing the vector coordinates
        # using the displacement vector
        p = (nv_c + n*nv_d).astype(np.int)
        # As long as the coordinates are within the directional binary
        # mask, displace the coordinates
        while bmi[p[0],p[1],p[2]] != 0: 
        #while brainmask_data[p[1],p[0],p[2]] != 0: # NB! y, x, z (c order)
            n += 1
            p = (nv_c + n*nv_d).astype(np.int)
        # The maximum displaced coordinates was found
        p_max_bmi = (nv_c + (n-1)*nv_d).astype(np.int)
        if args.verbose == 1:
            print("Max displaced coordinates within original directional mask: ", end='')
            print(p_max_bmi)
            
        # Calculate the magnitude of the maximum displacement
        disp_max_bmi = np.linalg.norm((p_max_bmi-nv_c).astype(np.float32))
        if args.verbose == 1:
            print("Max displacement within original directional mask along vector: %f" % disp_max_bmi)
        
        # Find the maximum displacement possible along the nv_d vector 
        # between its starting position and the end of the brain mask
        n = 1
        # TODO: Can introduce step size in order to have more accurate displacements
        # as for now, this is equal to a step size of 1 (n*step_size = 1 here).
        # Start by displacing the vector coordinates
        # using the displacement vector
        p = (nv_c + n*nv_d).astype(np.int)
        # As long as the coordinates are within the brain binary
        # mask, displace the coordinates
        while brainmask_data[p[0],p[1],p[2]] != 0: 
        #while brainmask_data[p[1],p[0],p[2]] != 0: # NB! y, x, z (c order)
            n += 1
            p = (nv_c + n*nv_d).astype(np.int)
        # The maximum displaced coordinates was found
        p_max_brain = (nv_c + (n-1)*nv_d).astype(np.int)
        if args.verbose == 1:
            print("Max displaced coordinates within brain mask: ", end='')
            print(p_max_brain)
        # Calculate the magnitude of the maximum displacement
        disp_max_brain = np.linalg.norm((p_max_brain-nv_c).astype(np.float32))
        if args.verbose == 1:
            print("Max displacement within brain mask along vector: %f" % disp_max_brain)
        
        # Scale the extent (by fraction) of the displacements reaching the end of the brain mask
        disp_max_brain = args.brain_coverage_fraction*disp_max_brain
        if args.verbose == 1:
            print("Scaled max displacement within brain mask along vector: %f" % disp_max_brain)
        
        p_max_brain = (nv_c + disp_max_brain*nv_d).astype(np.int)
        if args.verbose == 1:
            print("Scaled max displaced coordinates within brain mask: ", end='')
            print(p_max_brain)
        
        # Get the absolute values of original displacement field
        # within the original directionary mask
        dnormcone = dnorm[bmi == 1]
        # .. as well as the actual displacements
        dxcone = dx[bmi == 1]
        dycone = dy[bmi == 1]
        dzcone = -dz[bmi == 1] # For the same reson for negating nv_d[-1]. Invert operation 3
        # Normalize the displacement coordinates within the mask
        # so that each displacement vector is unit length
        dxcone = dxcone/dnormcone
        dycone = dycone/dnormcone
        dzcone = dzcone/dnormcone
        
        if args.intensity_decay_fraction > 0:
            
            # Making a copy of bmi to work on
            bmi_copy = bmi.copy()
            
            if args.verbose == 1:
                print("Extending directional mask")
            
            # Get all the positions within the original directionary mask
            # as float32
            mask_pts = np.argwhere(bmi).astype(np.float32)
            
            # Displace the positions of the original directional
            # binary mask according to the normalized displacement vectors
            # ,the difference between maximum displacement
            # and maximum displacement within original directional
            # binary mask, then scaled with an intensity decay fraction.
            
            extension = args.intensity_decay_fraction*(disp_max_brain-disp_max_bmi)
            # If extension is negative, the the stretched cone
            # is identical to the original cone (not stretched and bounding bounding box
            # remains the same)
            if extension > 0:
                mask_pts[:,0] += extension*dxcone
                mask_pts[:,1] += extension*dycone
                mask_pts[:,2] += extension*dzcone
            # Fill in the points of the diplaced binary mask. TODO: improve speed if found slow
            if args.verbose == 1:
                print("Storing displaced positions for extended directional binary mask")
            for tofilli in np.unique(mask_pts.astype(np.int), axis=0):
                xp, yp, zp = tofilli[0], tofilli[1], tofilli[2]
                if xp < 0 or yp < 0 or zp < 0:
                    if args.verbose == 1:
                        print("Negative index of stretched binary mask, Skipping")
                        print(xp)
                        print(yp)
                        print(zp)
                elif xp < xsize and yp < ysize and zp < zsize and brainmask_data[xp,yp,zp] != 0:
                    bmi_copy[xp, yp, zp] = 1
            
            # Find the geometric center of the directional
            # binary mask extended for determining coordinates before interpolation
            if args.verbose == 1:
                print("Finding bounding box for binary dilated directional mask extended for determining coordinates before interpolation")
            bm_geom_center, bm_widths = bounding_box_mask(binary_dilation(bmi_copy)) # NB! Binary dilation is performed to ensure complete coverage of the field
            #bm_geom_center, bm_widths = bounding_box_mask(bmi_copy)
        else:
            # Find the geometric center of the directional
            # binary mask extended for determining coordinates before interpolation
            if args.verbose == 1:
                print("Finding bounding box for binary dilated directional mask for determining coordinates before interpolation")
            bm_geom_center, bm_widths = bounding_box_mask(binary_dilation(bmi)) # NB! Binary dilation is performed to ensure complete coverage of the field
            #bm_geom_center, bm_widths = bounding_box_mask(bmi)
        
        bmcx, bmcy, bmcz = bm_geom_center
        bmwx, bmwy, bmwz = bm_widths
        
        # Calculate a fraction used for scaling the interpolated 
        # displacement accrong to how good the bounding box of 
        # the cone correspons to the minimal bounding box of an elliptic cone.
        # https://mathworld.wolfram.com/EllipticCone.html
        # Better fit, fraction closer to 1.
        # The reason for doing this is that the volume of a bounding box
        # for an original or extended cone varies depending on how minimal 
        # the bounding box convers the cone
        # (how optimal the bounding box is enclosing the cone in the fixed axis reference space).
        # A = volume of an elliptic cone
        # B = volume of the minimal boundinb box
        # Then B/A = 3/pi
        # The fraction is calculated as 
        # scale_fr = 1-(3/pi)/(B_real/A_real)
        # Where A_real is the volume of the original elliptic cone
        # and B_real is the volume of its bounding box
        #scale_fr = 1-(3/np.pi)/(np.prod(bm_widths)/np.sum(bmi))
        #scale_fr = 1
        #print("Will scale interpolated displacement for the extended cone with: %f" % scale_fr)
        
        # Stretch the displacement field towards the skull,
        # using the directional binary mask (bmi), disp_max_brain,
        # disp_max_bmi, dx, dy, dz, field_data
        # and interpolation. To do this,
        # find the geometric center of the directional
        # binary mask AFTER inteprolation
        
        # Not used start
        # Set voxel to 1 at maximum displaced voxel
        # indicating interpolation (stretched binary mask)
        #bmi[p_max[0],p_max[1],p_max[2]] = 1
        #print("Finding bounding box for directional mask with maximum displaced position added")
        # Not used end
        
        # Making a copy of bmi to work on
        bmi_copy = bmi.copy()
        
        if args.verbose == 1:
            print("Extending directional mask")

        # Get all the positions within the original directionary mask
        # as float32
        mask_pts = np.argwhere(bmi).astype(np.float32)
        
        # Displace the positions of the original directional
        # binary mask according to these displacement vectors
        # and the difference between maximum displacement
        # and maximum displacement within original directional
        # binary mask
        extension = (disp_max_brain-disp_max_bmi)
        # If extension is negative, the the stretched cone
        # is identical to the original cone (not stretched and bounding bounding box
        # remains the same)
        if extension > 0:
            mask_pts[:,0] += extension*dxcone
            mask_pts[:,1] += extension*dycone
            mask_pts[:,2] += extension*dzcone
        # Fill in the points of the diplaced binary mask. TODO: improve speed if found slow
        if args.verbose == 1:
            print("Storing displaced positions for extended directional binary mask")
        for tofilli in np.unique(mask_pts.astype(np.int), axis=0):
            xp, yp, zp = tofilli[0], tofilli[1], tofilli[2]
            if xp < 0 or yp < 0 or zp < 0:
                if args.verbose == 1:
                    print("Negative index of stretched binary mask, Skipping")
                    print(xp)
                    print(yp)
                    print(zp)
            elif xp < xsize and yp < ysize and zp < zsize and brainmask_data[xp,yp,zp] != 0:
                bmi_copy[xp, yp, zp] = 1
        
        # Now the new bounding box can be found
        if args.verbose == 1:
            print("Finding bounding box for binary dilated directional mask extended for determining coordinates after interpolation")
        bm_geom_center_interp, bm_widths_interp = bounding_box_mask(binary_dilation(bmi_copy)) # NB! Binary dilation is performed to ensure complete coverage of the field
        #bm_geom_center_interp, bm_widths_interp = bounding_box_mask(bmi_copy)
        
        bmcx_interp, bmcy_interp, bmcz_interp = bm_geom_center_interp
        bmwx_interp, bmwy_interp, bmwz_interp = bm_widths_interp
        
        # For diagnostics, store old and intepolated (stretched) bounding box
        if args.verbose == 1:
            print("Storing original and extended bounding boxes")
        orig_bboxes_data[bmcx-bmwx//2:bmcx+bmwx//2, \
                         bmcy-bmwy//2:bmcy+bmwy//2, \
                         bmcz-bmwz//2:bmcz+bmwz//2, \
                         i] = 1
        
        interp_bboxes_data[bmcx_interp-bmwx_interp//2:bmcx_interp+bmwx_interp//2, \
                           bmcy_interp-bmwy_interp//2:bmcy_interp+bmwy_interp//2, \
                           bmcz_interp-bmwz_interp//2:bmcz_interp+bmwz_interp//2, \
                           i] = 1
        
        """
        # Find the maximum displacement possible along the nv_d vector 
        # between its starting position and the end of the outer ellipsoid mask
        n = 1
        # Start by displacing the vector coordinates
        # using the displacement vector
        p = (nv_c + n*nv_d).astype(np.int)
        # As long as the coordinates are within the outer ellipsoid mask
        # mask, displace the coordinates
        while outer_ellipsoid_data[p[0],p[1],p[2]] != 0: 
        #while brainmask_data[p[1],p[0],p[2]] != 0: # NB! y, x, z (c order)
            n += 1
            p = (nv_c + n*nv_d).astype(np.int)
        # The maximum displaced coordinates was found
        p_max_oel = (nv_c + (n-1)*nv_d).astype(np.int)
        if args.verbose == 1:
            print("Max displaced coordinates within outer ellipsoid mask: ", end='')
            print(p_max_oel)
        
        # Calculate the absolute max difference between p_max_oel and nv_c
        diff_max_oel_abs = np.abs((p_max_oel-nv_c).astype(np.float32))
        if args.verbose == 1:
            print("Absolute max difference between ellipsoid mask surface and outer ellipsoid mask surface along vector: " , end='')
            print(diff_max_oel_abs)
        """
        
        # Extract the components of the vector position (inflection point)
        #nv_cx, nv_cy, nv_cz = nv_c

        # Extract the components of the vector (displacement point). This remembers invert operation 2
        #nv_dx, nv_dy, nv_dz = nv_d
        
        # Extract the components of the maximum displacement along 
        # nv_d within the outer ellipsoid mask
        #diff_max_abs_x, diff_max_abs_y, diff_max_abs_z = diff_max_oel_abs
        
        # Scaling parameter <0,1] : max_radial_displacement_to_outer_ellipsoid_mask_fraction
        # 0.1: Only using the value nearest inflection surface to intepolate.
        # Small value will lead to most rigid displacements.
        # 1: The entire radial range from inflection surface to outer ellipsoid
        # surface will be used for intepolation. Large value will lead to least
        # rigid / most elastic displacements.
        #scale_param = 0.5
        #scale_param = args.max_radial_displacement_to_outer_ellipsoid_mask_fraction
        """
        # Find the correct start and end points for intepolation
        # inside the original bounding box
        if bmcx > nv_cx:
            cstartx = bmcx-bmwx//2
            #cstartx = nv_cx
            #cendx = bmcx
            cendx = nv_cx + scale_param*diff_max_abs_x*nv_dx
        else:
            #cstartx = bmcx
            cstartx = nv_cx + scale_param*diff_max_abs_x*nv_dx
            cendx = bmcx+bmwx//2
            #cendx = nv_cx
        if bmcy > nv_cy:
            cstarty = bmcy-bmwy//2
            #cstarty = nv_cy
            #cendy = bmcy
            cendy = nv_cy + scale_param*diff_max_abs_y*nv_dy
        else:
            #cstarty = bmcy
            cstarty = nv_cy + scale_param*diff_max_abs_y*nv_dy
            cendy = bmcy+bmwy//2
            #cendy = nv_cy
        if bmcz > nv_cz:
            cstartz = bmcz-bmwz//2
            #cstartz = nv_cz
            #cendz = bmcz
            cendz = nv_cz + scale_param*diff_max_abs_z*nv_dz
        else:
            #cstartz = bmcz
            cstartz = nv_cz + scale_param*diff_max_abs_z*nv_dz
            cendz = bmcz+bmwz//2
            #cendz = nv_cz
        
        # Define a new grid for interpolation of points
        
        xi, yi, zi = np.mgrid[cstartx:cendx:bmwx_interp*1j, \
                              cstarty:cendy:bmwy_interp*1j, \
                              cstartz:cendz:bmwz_interp*1j]
        # Old version        
        #xi, yi, zi = np.mgrid[bmcx-bmwx//2:bmcx:bmwx_interp*1j, \
        #                      bmcy-bmwy//2:bmcy:bmwy_interp*1j, \
        #                      bmcz-bmwz//2:bmcz:bmwz_interp*1j]
        """
        
        # Define a new grid for interpolation of points        
        xi, yi, zi = np.mgrid[bmcx-bmwx//2:bmcx+bmwx//2:bmwx_interp*1j, \
                              bmcy-bmwy//2:bmcy+bmwy//2:bmwy_interp*1j, \
                              bmcz-bmwz//2:bmcz+bmwz//2:bmwz_interp*1j]
        
        # Interpolate
        if args.verbose == 1:
            print("Interpolating x displacement")
        dxi = map_coordinates(dx, [xi.ravel(), yi.ravel(), zi.ravel()], order=args.spline_order)\
                                 .reshape(bmwx_interp, bmwy_interp, bmwz_interp)
        if args.verbose == 1:
            print("Interpolating y displacement")
        dyi = map_coordinates(dy, [xi.ravel(), yi.ravel(), zi.ravel()], order=args.spline_order)\
                                 .reshape(bmwx_interp, bmwy_interp, bmwz_interp)
        if args.verbose == 1:
            print("Interpolating z displacement")
        dzi = map_coordinates(dz, [xi.ravel(), yi.ravel(), zi.ravel()], order=args.spline_order)\
                                 .reshape(bmwx_interp, bmwy_interp, bmwz_interp)
        if args.verbose == 1:
            print("Interpolating Gaussian")
        gaussian_data_interp_part = map_coordinates(gaussian_data, [xi.ravel(), yi.ravel(), zi.ravel()], order=args.spline_order)\
                                                                  .reshape(bmwx_interp, bmwy_interp, bmwz_interp)
        
        if args.verbose == 1:
            print("Inserting interpolated displacements and Gaussian into existing arrays")
        field_data_interp[bmcx_interp-bmwx_interp//2:bmcx_interp+bmwx_interp//2, \
                          bmcy_interp-bmwy_interp//2:bmcy_interp+bmwy_interp//2, \
                          bmcz_interp-bmwz_interp//2:bmcz_interp+bmwz_interp//2, \
                          0, i] = dxi
        field_data_interp[bmcx_interp-bmwx_interp//2:bmcx_interp+bmwx_interp//2, \
                          bmcy_interp-bmwy_interp//2:bmcy_interp+bmwy_interp//2, \
                          bmcz_interp-bmwz_interp//2:bmcz_interp+bmwz_interp//2, \
                          1, i] = dyi
        field_data_interp[bmcx_interp-bmwx_interp//2:bmcx_interp+bmwx_interp//2, \
                          bmcy_interp-bmwy_interp//2:bmcy_interp+bmwy_interp//2, \
                          bmcz_interp-bmwz_interp//2:bmcz_interp+bmwz_interp//2, \
                          2, i] = dzi
        gaussian_data_interp[bmcx_interp-bmwx_interp//2:bmcx_interp+bmwx_interp//2, \
                             bmcy_interp-bmwy_interp//2:bmcy_interp+bmwy_interp//2, \
                             bmcz_interp-bmwz_interp//2:bmcz_interp+bmwz_interp//2, \
                             i] = gaussian_data_interp_part
        if args.verbose == 1:
            print("Inserting done")
        """
        print("Scalig interpolated displacements and Gaussian using dot products") # TODO: This resulted in triangle shaped displacements
        # Invert operation 6
        # https://falstad.com/dotproduct/, scaled according to |A|cos(theta)
        nv_d[-1] *= -1
        interpfield = np.stack((dxi, dyi, dzi), axis=-1)
        interpfieldnorm = np.linalg.norm(interpfield, axis=-1)
        fprod = np.dot(interpfield, nv_d)
        interp_directional_scaling = fprod/interpfieldnorm
        field_data_interp[bmcx_interp-bmwx_interp//2:bmcx_interp+bmwx_interp//2, \
                          bmcy_interp-bmwy_interp//2:bmcy_interp+bmwy_interp//2, \
                          bmcz_interp-bmwz_interp//2:bmcz_interp+bmwz_interp//2, \
                          0, i] *= interp_directional_scaling
        field_data_interp[bmcx_interp-bmwx_interp//2:bmcx_interp+bmwx_interp//2, \
                          bmcy_interp-bmwy_interp//2:bmcy_interp+bmwy_interp//2, \
                          bmcz_interp-bmwz_interp//2:bmcz_interp+bmwz_interp//2, \
                          1, i] *= interp_directional_scaling
        field_data_interp[bmcx_interp-bmwx_interp//2:bmcx_interp+bmwx_interp//2, \
                          bmcy_interp-bmwy_interp//2:bmcy_interp+bmwy_interp//2, \
                          bmcz_interp-bmwz_interp//2:bmcz_interp+bmwz_interp//2, \
                          2, i] *= interp_directional_scaling
        gaussian_data_interp[bmcx_interp-bmwx_interp//2:bmcx_interp+bmwx_interp//2, \
                             bmcy_interp-bmwy_interp//2:bmcy_interp+bmwy_interp//2, \
                             bmcz_interp-bmwz_interp//2:bmcz_interp+bmwz_interp//2, \
                             i] *= interp_directional_scaling
        """
        """
        # Garbage collect
        print("Garbage collect")
        del zi, yi, xi, bmi
        gc.collect()
        print("Garbage collect done")
        """
        if args.verbose == 0:
            # Append to progress bar
            sys.stdout.write("-")
            sys.stdout.flush()
    
    if args.verbose == 0:
        sys.stdout.write("]\n") # this ends the progress bar
    
    print("Interpolation execution time: %f s" %(time.time()-start_time))
    
    """
    print("Garbage collect")
    del zi, yi, xi, bmi
    print("Garbage collect done")
    """
    # Save old and intepolated (stretched) bounding box to disk
    print("Saving bounding boxes for original directional binary masks to disk")
    orig_bboxes_img = nib.spatialimages.SpatialImage(np.max(orig_bboxes_data, axis=-1), affine=ref_img.affine, header=ref_img.header)
    nib.save(orig_bboxes_img, args.out+"/original-bounding-box-vector-max.nii.gz")
    
    print("Saving bounding boxes for intepolated directional binary masks to disk")
    interp_bboxes_img = nib.spatialimages.SpatialImage(np.max(interp_bboxes_data, axis=-1), affine=ref_img.affine, header=ref_img.header)
    nib.save(interp_bboxes_img, args.out+"/interp-bounding-box-vector-max.nii.gz")
    
    """
    print("Garbage collect again")
    del orig_bboxes_data
    del orig_bboxes_img 
    del interp_bboxes_data
    del interp_bboxes_img 
    gc.collect() # TODO
    print("Garbage collect again done")
    """
    
    # Bounding box interpolation is now done.
    # Aggregate all interpolated data into a single vector field
    # using three final interpolations
    print("Building total intepolated field and Gaussian")
    
    # Take the mean or max over the last axis, excluding nan values
    print("Splitting interpolated displacement values before computing mean")
    # For debug
    #np.savez("field_data_interp.npz", field_data_interp)
    
    # Split interpolated field data, to be able to compute means without
    # exceeding memory limits
    field_data_interp_x, field_data_interp_y, field_data_interp_z = \
    field_data_interp[:,:,:,0,:], field_data_interp[:,:,:,1,:], field_data_interp[:,:,:,2,:]
    
    #"""
    print("Computing mean of non-nan x displacement values")
    field_data_interp_x = np.nanmean(field_data_interp_x, axis=-1, dtype=np.float32)
    #field_data_interp_x = np.nansum(field_data_interp_x, axis=-1)
    #field_data_interp_x = np.nanmedian(field_data_interp_x, axis=-1)
    #field_data_interp_x = np.nanmin(field_data_interp_x, axis=-1)
    #field_data_interp_x = np.nanmax(field_data_interp_x, axis=-1)
    print("Computing mean of non-nan y displacement values")
    field_data_interp_y = np.nanmean(field_data_interp_y, axis=-1, dtype=np.float32)
    #field_data_interp_y = np.nansum(field_data_interp_y, axis=-1)
    #field_data_interp_y = np.nanmedian(field_data_interp_y, axis=-1)
    #field_data_interp_y = np.nanmin(field_data_interp_y, axis=-1)
    #field_data_interp_y = np.nanmax(field_data_interp_y, axis=-1)
    print("Computing mean of non-nan z displacement values")
    field_data_interp_z = np.nanmean(field_data_interp_z, axis=-1, dtype=np.float32)
    #field_data_interp_z = np.nansum(field_data_interp_z, axis=-1)
    #field_data_interp_z = np.nanmedian(field_data_interp_z, axis=-1)
    #field_data_interp_z = np.nanmin(field_data_interp_z, axis=-1)
    #field_data_interp_z = np.nanmax(field_data_interp_z, axis=-1)
    
    # Stack the displacements together to make a field again
    field_data_interp = np.stack((field_data_interp_x, field_data_interp_y, field_data_interp_z), axis=-1)
    #"""
    
    #print("Computing max of non-nan displacement values")
    #field_data_interp_max = np.nanmax(field_data_interp, axis=-1)
    #print("Computing min of non-nan displacement values")
    #field_data_interp_min = np.nanmin(field_data_interp, axis=-1)
    #print("Computing mean of these non-nan max and min values")
    #field_data_interp = np.nanmean(np.stack((field_data_interp_max, field_data_interp_min), axis=-1), axis=-1, dtype=np.float32)
    #print("Combining max and min of non-nan values to get the minimum of absolute length vectors")
    #field_data_interp = field_data_interp_min
    #field_data_interp[-field_data_interp_max < field_data_interp_min] = field_data_interp_max[-field_data_interp_max < field_data_interp_min]
    #field_data_interp = field_data_interp_max
    #field_data_interp[-field_data_interp_min > field_data_interp_max] = field_data_interp_min[-field_data_interp_min > field_data_interp_max]

    print("Computing mean of non-nan Gaussian values")
    gaussian_data_interp = np.nanmean(gaussian_data_interp, axis=-1, dtype=np.float32)
    #gaussian_data_interp = np.nansum(gaussian_data_interp, axis=-1)
    #gaussian_data_interp = np.nanmedian(gaussian_data_interp, axis=-1)
    #gaussian_data_interp = np.nanmin(gaussian_data_interp, axis=-1)
    #gaussian_data_interp = np.nanmax(gaussian_data_interp, axis=-1)
    
    """
    print("Garbage collect again")
    gc.collect()
    print("Garbage collect again done")
    """
    
    # Set nan values to 0
    field_data_interp[np.isnan(field_data_interp)] = 0
    #print(np.unique(field_data_interp))
    #d = np.nanmean(field_data_interp, axis=-1, dtype=np.float32)
    #d = np.nanmean(field_data_interp, axis=-1)
    
    gaussian_data_interp[np.isnan(gaussian_data_interp)] = 0
    
    # Smooth
    # NB! Gaussian smoothing like this will lower the extreme values (maximum absolute displacement)
    # a little.
    print("Smoothing interpolated x displacement")
    field_data_interp[...,0] = gaussian_filter(field_data_interp[...,0], sigma=args.smoothing_std)
    print("Smoothing interpolated y displacement")
    field_data_interp[...,1] = gaussian_filter(field_data_interp[...,1], sigma=args.smoothing_std)
    print("Smoothing interpolated z displacement")
    field_data_interp[...,2] = gaussian_filter(field_data_interp[...,2], sigma=args.smoothing_std)
    print("Smoothing interpolated Gaussian")
    gaussian_data_interp = gaussian_filter(gaussian_data_interp, sigma=args.smoothing_std)
    
    # Add noise to the interpolated field and Gaussian BEFORE scaling with specified displacement    
    """
    # Gaussian noise
    field_data += \
    np.random.normal(loc=0,scale=args.noise_param,size=field_data.shape)
    field_data_interp += \
    np.random.normal(loc=0,scale=args.noise_param,size=field_data_interp.shape)
    gaussian_data_interp += \
    np.random.normal(loc=0,scale=args.noise_param,size=gaussian_data_interp.shape)
    """
    
    if args.perlin_noise_abs_max > 0:
        # Perlin noise
        # https://github.com/pvigier/perlin-numpy
        #"""
        # Find the number of periods of noise to generate along each axis, based on args.perlin_noise_res
        resx, resy, resz = xsize//np.int(xsize*args.perlin_noise_res),\
                           ysize//np.int(ysize*args.perlin_noise_res),\
                           zsize//np.int(zsize*args.perlin_noise_res)
        # Round xsize, ysize, zsize integers down to nearest multiple of resx, resy, resz
        # to create xsize_perlin, ysize_perlin, zsize_perlin
        xsize_perlin = xsize - (xsize%resx)
        ysize_perlin = ysize - (ysize%resy)
        zsize_perlin = zsize - (zsize%resz)
        # Generate perlin noise
        print("Generating Perlin noise for x displacement")
        np.random.seed(random_seed)
        noisex = \
        args.perlin_noise_abs_max*generate_perlin_noise_3d((xsize_perlin, ysize_perlin, zsize_perlin), (resx, resy, resz)).astype(np.float32)
        print("Generating Perlin noise for y displacement")
        np.random.seed(random_seed)
        noisey = \
        args.perlin_noise_abs_max*generate_perlin_noise_3d((xsize_perlin, ysize_perlin, zsize_perlin), (resx, resy, resz)).astype(np.float32)
        print("Generating Perlin noise for z displacement")
        np.random.seed(random_seed)
        noisez = \
        args.perlin_noise_abs_max*generate_perlin_noise_3d((xsize_perlin, ysize_perlin, zsize_perlin), (resx, resy, resz)).astype(np.float32)
        print("Adding perlin noise to original displacement field")
        field_data[:xsize_perlin,:ysize_perlin,:zsize_perlin,0] += noisex
        field_data[:xsize_perlin,:ysize_perlin,:zsize_perlin,1] += noisey
        field_data[:xsize_perlin,:ysize_perlin,:zsize_perlin,2] += noisez
        print("Adding perlin noise to interpolated displacement field")
        field_data_interp[:xsize_perlin,:ysize_perlin,:zsize_perlin,0] += noisex
        field_data_interp[:xsize_perlin,:ysize_perlin,:zsize_perlin,1] += noisey
        field_data_interp[:xsize_perlin,:ysize_perlin,:zsize_perlin,2] += noisez
        print("Computing absolute value of Perlin noise field")
        noisenorm = np.linalg.norm(np.stack((noisex, noisey, noisez), axis=-1), axis=-1)
        print("Adding perlin noise to interpolated Gaussian")
        gaussian_data_interp[:xsize_perlin,:ysize_perlin,:zsize_perlin] += noisenorm
        #"""
        
        # Save Perlin field
        print("Saving Perlin noise field")
        perlin_img = \
        nib.spatialimages.SpatialImage(np.stack((noisex, noisey, noisez), axis=-1), affine=ref_img.affine, header=ref_img.header)
        nib.save(perlin_img, args.out+"/perlin-noise.nii.gz")
    
    # Finally, scale displacement fields with the specified intensity
    print("Scaling displacement fields")
    field_data *= args.displacement
    field_data_interp *= args.displacement
    
    # Make the interpolated ellipsoid and outer ellipsoid masks
    print("Computing interpolated ellipsoid masks")
    
    # - Interpolated ellipsoid mask    
    ellipsoid_mask_interp = gaussian_data_interp < -ellipsoid_threshold
    ellipsoid_data_interp = np.zeros(gaussian_data_interp.shape, dtype=np.int)
    ellipsoid_data_interp[ellipsoid_mask_interp] = 1
    # Save the mask to nifti
    print("Saving interpolated ellipsoid mask")
    ellipsoid_img_interp = \
    nib.spatialimages.SpatialImage(ellipsoid_data_interp, affine=ref_img.affine, header=ref_img.header)
    #nib.save(ellipsoid_img_interp, "interp-"+args.eout)
    nib.save(ellipsoid_img_interp, args.out+"/interp-ellipsoid-mask.nii.gz")
    
    # - Interpolated outer ellipsoid mask
    # Create a mask of an ellipsoid where its
    # surface marks approximately the end of the 3D gaussian
    # Store it in the reference image space
    outer_ellipsoid_mask_interp = gaussian_data_interp <= -0.05 # This was a good default
    outer_ellipsoid_data_interp = np.zeros(gaussian_data_interp.shape, dtype=np.int)
    outer_ellipsoid_data_interp[outer_ellipsoid_mask_interp] = 1
    
    # Save the mask to nifti
    print("Saving interpolated outer ellipsoid mask")
    outer_ellipsoid_img_interp = \
    nib.spatialimages.SpatialImage(outer_ellipsoid_data_interp, affine=ref_img.affine, header=ref_img.header)
    #nib.save(outer_ellipsoid_img_interp, "interp-outer-"+args.eout)
    nib.save(outer_ellipsoid_img_interp, args.out+"/interp-outer-ellipsoid-mask.nii.gz")
        
    # Remove displacements from bounding boxes starting from outside of the brain
    field_data_interp[brainmask_data != 1] = 0
    gaussian_data_interp[brainmask_data != 1] = 0
    
    # Restrict displacements leaving the brain START
    # TODO: This section of code could be further improved and optimzed:
    # 1. Improvements: Better handling of edge cases. 
    # 2. Optimization: Abstraction.
    # Avoid displacing outside of the brain mask, NB! Might crash if Gaussian smoothng is turned off
    print("Scaling interpolated displacement field to not displace outside of the brain mask")
    # Get all the positions (points) within the brian mask
    mask_pts = np.argwhere(brainmask_data).astype(np.float32)
    # Get the interpolated displacement field
    # within the brain mask
    dinterpmask = field_data_interp[brainmask_data == 1]
    dinterpmask[:,-1] *= -1 # NB! Invert operation 4
    # Displace the mask_pts with the interpolated field
    mask_pts_displaced = mask_pts + dinterpmask
    # Create mask of the points that went outside of the brain mask
    mask_pts_x, mask_pts_y, mask_pts_z = mask_pts[:,0], mask_pts[:,1], mask_pts[:,2]
    mask_pts_displaced_x, mask_pts_displaced_y, mask_pts_displaced_z = \
    mask_pts_displaced[:,0], mask_pts_displaced[:,1], mask_pts_displaced[:,2]
    points_outside_mask_x = ~np.isin(mask_pts_displaced_x.astype(np.int), mask_pts_x.astype(np.int))
    points_outside_mask_y = ~np.isin(mask_pts_displaced_y.astype(np.int), mask_pts_y.astype(np.int))
    points_outside_mask_z = ~np.isin(mask_pts_displaced_z.astype(np.int), mask_pts_z.astype(np.int))
    # Stack together the coordinates of the points that were displaced to outside for the brain mask
    points_outside_mask = \
    np.stack((points_outside_mask_x, points_outside_mask_y, points_outside_mask_z), axis=0).any(axis=0)
    
    # Continue only if some points actually went outside of the mask    
    if points_outside_mask.size == 0:
        # Contiue working only with points and displacements that went outside of the brain mask
        mask_pts_went_outside = mask_pts[points_outside_mask]
        dinterpmask_went_outside = dinterpmask[points_outside_mask]
        # Find the absolute value of displacements that went outside of the brain mask
        dinterpmasknorm_went_outside = np.linalg.norm(dinterpmask_went_outside, axis=-1)
        # Find the maximum absolute displacement
        maxdisp_went_outside = np.max(dinterpmasknorm_went_outside)
        # Calculate all candidate displacements that are restricted to within the brain mask
        dnorm_went_outside = dinterpmask_went_outside/np.expand_dims(dinterpmasknorm_went_outside, axis=-1)
        # q = p + n*np.arange(maxdisp_went_outside.astype(np.int)). Where n is the normalized displacement vector starting from point p
        allpts_displacements = \
        np.expand_dims(mask_pts_went_outside, axis=-1) + np.expand_dims(dnorm_went_outside, axis=-1)*np.arange(maxdisp_went_outside.astype(np.int))
        allpts_displacements_x, allpts_displacements_y, allpts_displacements_z = \
        allpts_displacements[:,0,:], allpts_displacements[:,1,:], allpts_displacements[:,2,:]
        # Create mask of the candidate displacements that went oustie of the brain
        points_outside_mask_x = ~np.isin(allpts_displacements_x.astype(np.int), mask_pts_x.astype(np.int))
        points_outside_mask_y = ~np.isin(allpts_displacements_y.astype(np.int), mask_pts_y.astype(np.int))
        points_outside_mask_z = ~np.isin(allpts_displacements_z.astype(np.int), mask_pts_z.astype(np.int))
        #points_outside_mask = np.stack((points_outside_mask_x, points_outside_mask_y, points_outside_mask_z), axis=0).any(axis=0)
        # Find the furthest candidate displacement that are still within the brain mask
        # If no candadita displacement was found, the returned candidate displacement will be negative.
        # Set these negative values to 0, indicating no displacement as the candidate displacement.
        # Continue only if some points actually went outside of the mask
        
        #if points_outside_mask_x.size: TODO, investigate if necessary        
        
        furthest_point_within_mask_x = np.argmax(points_outside_mask_x, axis=-1).astype(np.float32)-1
        furthest_point_within_mask_x[furthest_point_within_mask_x < 0] = 0
        furthest_point_within_mask_y = np.argmax(points_outside_mask_y, axis=-1).astype(np.float32)-1
        furthest_point_within_mask_y[furthest_point_within_mask_y < 0] = 0
        furthest_point_within_mask_z = np.argmax(points_outside_mask_z, axis=-1).astype(np.float32)-1
        furthest_point_within_mask_z[furthest_point_within_mask_z < 0] = 0
        # Stack together
        furthest_point_within_mask = \
        np.stack((furthest_point_within_mask_x, furthest_point_within_mask_y, furthest_point_within_mask_z), axis=-1)
        # Actually find the furthest points displaced within the brain mask
        furthest_point_within_mask = mask_pts_went_outside + dnorm_went_outside*furthest_point_within_mask
        # Find the corresponding max displacement
        dinterpmask_restricted = furthest_point_within_mask-mask_pts_went_outside
        # Paste these new restricted displacement into the array of interpolated displacements
        dinterpmask[points_outside_mask] = dinterpmask_restricted
        dinterpmask[:,-1] *= -1 # NB! Invert operation 5 (invert z component back, for ANTs)
        field_data_interp[brainmask_data == 1] = dinterpmask
    # Restrict displacements leaving the brain END
    
    # Save original (non-intepolated) field
    print("Saving original fields")
    field_img = nib.spatialimages.SpatialImage(field_data, affine=ref_img.affine, header=ref_img.header)
    #nib.save(field_img, args.fout)
    nib.save(field_img, args.out+"/field-"+str(args.displacement)+"mm.nii.gz")
    
    # Also save the negative of the field (since ITK-SNAP needs the negative visualize correctly)
    field_oppos_img = nib.spatialimages.SpatialImage(-field_data, affine=ref_img.affine, header=ref_img.header)
    #nib.save(field_oppos_img, "neg-"+args.fout)
    nib.save(field_oppos_img, args.out+"/neg-field-"+str(args.displacement)+"mm.nii.gz")
    
    # Save intepolated field
    print("Saving interpolated fields and Gaussian")
    #np.savez("field_data_interp.npz", field_data_interp)
    field_img_interp = nib.spatialimages.SpatialImage(field_data_interp, affine=ref_img.affine, header=ref_img.header)
    #nib.save(field_img_interp, "interp-"+args.fout)
    nib.save(field_img_interp, args.out+"/interp-field-"+str(args.displacement)+"mm.nii.gz")
    
    # Also save the negative of the field (since ITK-SNAP needs the negative visualize correctly)
    field_oppos_img_interp = nib.spatialimages.SpatialImage(-field_data_interp, affine=ref_img.affine, header=ref_img.header)
    #nib.save(field_oppos_img_interp, "interp-neg-"+args.fout)
    nib.save(field_oppos_img_interp, args.out+"/interp-neg-field-"+str(args.displacement)+"mm.nii.gz")
    
    gaussian_img_interp = \
    nib.spatialimages.SpatialImage(-gaussian_data_interp, affine=ref_img.affine, header=ref_img.header)
    nib.save(gaussian_img_interp, args.out+"/interp-gaussian.nii.gz")

    print("Script execution time: %f s" %(time.time()-script_start_time))
    print(sys.argv[0] + " done")
