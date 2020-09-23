import argparse
import numpy as np
import nibabel as nib
import sys
from scipy.interpolate import Rbf, griddata
from scipy.ndimage import map_coordinates, gaussian_filter
import gc

def max_spread_vectors_subset(vectors, positions, num_vecs=100):
    # Select a first vector and its position as the first
    # of the num_vecs selected vectors
    #randi = np.random.choice(len(vectors), 1)
    seli = np.array([0])
    vectors_sel = vectors[seli]
    positions_sel = positions[seli]
    #vectors_sel = vectors[randi]
    #positions_sel = positions[randi] # TODO
    #vectors_sel = np.array([-0.3057756, -0.6702055, -0.6794647], dtype=np.float32).reshape(1, 3) # TODO debug
    #positions_sel = np.array([43, 106,  64], dtype=np.int).reshape(1, 3)
    """
    print("Selected vector and position")
    print(vectors_sel.shape)
    print(vectors_sel)
    print(positions_sel.shape)
    print(positions_sel)
    """
    # Delete selected vector and position from array
    vectors = np.delete(vectors, seli, axis=0)
    positions = np.delete(positions, seli, axis=0)
    #vectors = np.delete(vectors, randi, axis=0) # TODO
    #positions = np.delete(positions, randi, axis=0)
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
        #print("Selected vector and position %i/%i" % (i+1,num_vecs))
        # Store selected vector and its position
        vectors_sel = np.append(vectors_sel, [v], axis=0)
        positions_sel = np.append(positions_sel, [p], axis=0)
        # Delete selected vector and position from array
        vectors = np.delete(vectors, maxi, axis=0)
        positions = np.delete(positions, maxi, axis=0)
    return vectors_sel, positions_sel

def restrict_to_max_displacement(disp_magn, disp_max):
    """
    NB: Used after the interpolation of stretched version
    NB: Used after the displacement intensity scaling
    disp_magn: Array of magnitudes of equidistant radial displacement vectors
    disp_max: The maximum radial displacement starting from the ellipsoid surface
              before hitting the end of the brain
    """
    disp_max_arr = disp_max - np.arange(len(disp_magn))
    disp_magn_restricted = np.minimum(disp_magn, disp_max_arr)
    disp_magn_restricted[disp_magn_restricted<0] = 0
    return disp_magn_restricted    

def scale_vector_positions(dx_flat, \
                           dy_flat, \
                           dz_flat, \
                           tumorbbox_geom_center, \
                           nv_d, \
                           nv_c, \
                           p_max):
    """
    tumorbbox_geom_center: x, y, z coordinates of the geometric center of the
                           tumor bounding box (center of ellipsoid)
    nv_d: x, y, z displacement of the normal vector at ellipsoid surface
    nv_c: x, y, z coordinates of the normal vector at ellipsoid surface
    p_max: x, y, z coodinates of the point furthest away from nv_c along nv_d within the brain
    """
    #print(tumorbbox_geom_center)
    #print(type(tumorbbox_geom_center))
    
    # Find the vector with coodinates furthest away from tumorbbox_geom_center
    l = lambda x: np.linalg.norm((x[1:]-tumorbbox_geom_center).astype(np.float32))
    mp = dx_flat[np.argmax(np.apply_along_axis(l, 1, dx_flat))]
    mv_c = mp[1:]
    print("furthest point away from ellipsoid center is")
    print(mv_c)
    
    #print(dx_flat.shape)
    #print(np.apply_along_axis(l, 1, dx_flat).shape)
    #sys.exit()

def gaussian_norm(x):
    return np.exp(-(x**2)/2)

def flatten_tensor(T):
    """
    Flattens the M x N x ..., x D tensor T while preserving 
    original indices to the output shape (MxNx...xD, F, M, N, ..., D)
    where F is a vector of the values in T, and M, N, ..., D vectors of 
    original indices along their dimension in T (for the values in F).
    https://stackoverflow.com/questions/46135070/generalise-slicing-operation-in-a-numpy-array/46135084#46135084
    """
    n = T.ndim
    grid = np.ogrid[tuple(map(slice, T.shape))]
    out = np.empty(T.shape + (n+1,), dtype=T.dtype)
    for i in range(n):
        out[...,i+1] = grid[i]
    out[...,0] = T
    out.shape = (-1,n+1)
    return out

def angle_between(v1, v2):
    dot_pr = v1.dot(v2)
    norms = np.linalg.norm(v1) * np.linalg.norm(v2)
 
    return np.rad2deg(np.arccos(dot_pr / norms))    

def bounding_box_mask(mask):
    """
    Returns the coodinates for the geometric center
    as well as x, y, and z widths of a binary mask,
    adjusted to an even number
    """
    mask_flat = flatten_tensor(mask)
    # Extract tumor
    mask_flat = mask_flat[mask_flat[:,0] == 1]
    x, y, z = mask_flat[:,1], mask_flat[:,2], mask_flat[:,3]
    #z, y, x = mask_flat[:,1], mask_flat[:,2], mask_flat[:,3]
    xmin, ymin, zmin = np.min(x), np.min(y), np.min(z)
    xmax, ymax, zmax = np.max(x), np.max(y), np.max(z)
    wx = xmax-xmin
    wy = ymax-ymin
    wz = zmax-zmin
    # If the widths are odd, add 1 to make them even
    if wx % 2:
        wx += 1
    if wy % 2:
        wy += 1
    if wz % 2:
        wz += 1
    cx = xmin + wx/2
    cy = ymin + wy/2
    cz = zmin + wz/2
    return (np.int(cx), np.int(cy), np.int(cz)), (np.int(wx), np.int(wy), np.int(wz))

if __name__ == "__main__":
    CLI=argparse.ArgumentParser()
    CLI.add_argument(
      "--ref",
      help="input 3D nifti for reference, can be a normal static scan",
      type=str,
      default=["2-T1c.nii"],
    )
    CLI.add_argument(
      "--tumormask",
      help="A binary mask with 1=tumor tissue, 0=healthy (outside of tumor) tissue",
      type=str,
      default=["2-Tumormask.nii.gz"],
    )
    CLI.add_argument(
      "--brainmask",
      help="A binary mask with 1=brain tissue, 0=outside of the brain",
      type=str,
      default=["2-BrainExtractionMask.nii.gz"],
    )
    CLI.add_argument(
      "--displacement",
      help="The maximum amount of radial displacement (in isotropic units according to --ref) to add",
      type=float,
      default=1,
    )
    CLI.add_argument(
      "--gaussian_range_one_sided",
      help="The one-sided range x used for x=[-x,x] on a normal (gaussian) symmetric distribution to model tumor expansion",
      type=float,
      default=5,
    )
    CLI.add_argument(
      "--max_radial_displacement_to_brainmask_fraction",
      help="<0=intepolated radial displacement is very local close to the tumor ellipsoid model inflection surface, 1=the maximum intepolated radial displacement will reach or overreach the end of the brain mask along its path. Displacements will cover most of the brain]",
      type=float,
      default=1,
    )
    CLI.add_argument(
      "--max_radial_displacement_to_outer_ellipsoid_mask_fraction",
      help="<0=almost no radial data to intepolate, stretching out first (largest) values near inflection surface (0), 1=no cut, entire gaussian_range_one_sided will be used for intepolation, starting from inflection surface]",
      type=float,
      default=1,
    )
    CLI.add_argument(
      "--num_vecs",
      help="The number of normal vectors used to simulate the explosive spread of tissue displacement",
      type=int,
      default=12,
    )
    CLI.add_argument(
      "--angle_thr",
      help="The maximum angle (in degrees) between a normal vector on the model ellipsoid surface and field vectors allowed when determining a directional binary mask",
      type=int,
      default=20,
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
      "--smoothing",
      help="Standard deviation for smoothing of the final intepolated x, y and z displacement",
      type=float,
      default=1,
    )    
    CLI.add_argument(
      "--eout",
      help="Base name of files saved containing binary masks of model ellipsoids",
      type=str,
      default=["2-ellipsoid-mask.nii.gz"],
    )
    CLI.add_argument(
      "--fout",
      help="Base name of generated 4D output fields, needs to be converted to ANTs transform before used in ANTs antsApplyTransforms",
      type=str,
      default=["field.nii.gz"],
    )
    args = CLI.parse_args()
    
    ref_img = nib.load(args.ref)
    tumormask_img = nib.load(args.tumormask)
    brainmask_img = nib.load(args.brainmask)
    
    xsize, ysize, zsize = ref_img.shape
    
    # Find the bounding box and geometric center
    # of the tumor based on the tumor mask
    tumorbbox_geom_center, tumorbbox_widths = \
    bounding_box_mask(tumormask_img.get_fdata())
    cx, cy, cz = tumorbbox_geom_center
    wx, wy, wz = tumorbbox_widths
    
    print("Tumor bounding box geometric center: ", end='')
    print(tumorbbox_geom_center)
    print("Tumor bounding box widths: ", end='')
    print(tumorbbox_widths)
    
    # - Build 3D Gaussian and calculate its gradient
    # Using range [-5,5] as attempt to include most of the function
    # https://en.wikipedia.org/wiki/Normal_distribution
    #endx = 5
    endx = args.gaussian_range_one_sided
    gy = gaussian_norm(np.linspace(-endx, endx, wy)).reshape((wy, 1))
    gz = gaussian_norm(np.linspace(-endx, endx, wz)).reshape((wz, 1)).T
    gx = gaussian_norm(np.linspace(-endx, endx, wx)).reshape((wx, 1, 1))
    
    # Make 3D Gaussian by multiplying together the 3 1D gaussians,
    # and using broadcasting. Before multiplication, 
    # the 1D Gaussians are scaled so that the 3D gaussian retains a shape
    # with inflection surface analog to the inflection point of the 
    # unscaled 1D Gaussian
    g3d = -1*((gy**(1/3))*(gz**(1/3))*(gx**(1/3)))
    
    # Create the Gaussian in the reference image space
    gaussian_data = np.zeros(ref_img.shape, dtype=np.float32)
    gaussian_data[cx-wx//2:cx+wx//2, cy-wy//2:cy+wy//2, cz-wz//2:cz+wz//2] = g3d
    
    # Save the 3D Gaussian to nifti
    gaussian_img = \
    nib.spatialimages.SpatialImage(-gaussian_data, affine=ref_img.affine, header=ref_img.header)
    nib.save(gaussian_img, "gaussian.nii.gz")
    
    # - 3D Gaussian gradients
    # The normalized partial derivatives of the 3D Gaussian are used 
    # to model the healthy and peritumoral displacement caused by nodal tumor growth.
    # The normalization is done so that all gradients (displacement vectors)
    # at the surface of the inflection ellipsoid have a magnitude of 1
    g3d_dx, g3d_dy, g3d_dz = np.gradient(g3d)
    dispx3d, dispy3d, dispz3d = g3d_dx/np.max(g3d_dx), \
                                g3d_dy/np.max(g3d_dy), \
                                -1*g3d_dz/np.max(g3d_dz)
    
    # - Ellipsoid mask
    # Create a mask of an ellipsoid where maximum displacement 
    # (= a gradient magnitude of 1) occurs normal to its surface
    ellipsoid_threshold = gaussian_norm(1)
    #ellipsoid_mask = g3d <= -ellipsoid_threshold
    ellipsoid_mask = g3d < -ellipsoid_threshold
        
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
    nib.save(ellipsoid_img, args.eout)
    
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
    nib.save(outer_ellipsoid_img, "outer-"+args.eout)
    
    # - Displacement field
    field_data = np.zeros(ref_img.shape+(3,), dtype=np.float32)
    
    print("Displacement: %f" % args.displacement)

    print("Normal Gaussian x-range: [-%f,%f]" % (-args.gaussian_range_one_sided, args.gaussian_range_one_sided))
    
    print("Fraction of maximum radial displacement distance to end of brain mask: %f" % args.max_radial_displacement_to_brainmask_fraction)
    
    print("Fraction of maximum radial displacement distance to end of outer ellipsoid mask: %f" % args.max_radial_displacement_to_outer_ellipsoid_mask_fraction)
        
    # Insert the normalized gradients of the 3D Gaussian as displacement
    # field vectors
    field_data[cx-wx//2:cx+wx//2, cy-wy//2:cy+wy//2, cz-wz//2:cz+wz//2, 0] = dispx3d
    field_data[cx-wx//2:cx+wx//2, cy-wy//2:cy+wy//2, cz-wz//2:cz+wz//2, 1] = dispy3d
    field_data[cx-wx//2:cx+wx//2, cy-wy//2:cy+wy//2, cz-wz//2:cz+wz//2, 2] = dispz3d
    
    # Compute more realistic displacement field using various extra information
    brainmask_data = brainmask_img.get_fdata()
    brainmask_data[brainmask_data != 1] = 0 # Note this mask was not binary, making it binary
    #mask_img = nib.spatialimages.SpatialImage(brainmask_data, affine=ref_img.affine, header=ref_img.header)
    #nib.save(mask_img, "brainmask-binary.nii.gz")
    
    # - Normal ellipsoid mask
    # Calculate the norm of the current displacement vector field
    fabs = np.linalg.norm(field_data, axis=-1)
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
    nib.save(normal_ellipsoid_img, "normal-"+args.eout)
        
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
    
    print("Maximum error angle allowed for directional binary masks: %i" % args.angle_thr)
        
    # Continue using only the num_vecs subset of the vectors
    # that are most spread
    #num_vecs = 16
    #num_vecs = 4
    #num_vecs = 2
    num_vecs = args.num_vecs
    
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
    #num_splits = 10
    num_splits = args.num_splits
    
    print("Number of splits: %i" % num_splits)

    print("Gaussian smoothing standard deviation: %f" % args.smoothing)

    print("Spline order for intepolation in map_coordinates: %i" % args.spline_order)
    
    # Calculate the split size and remainder
    split_size, remaining = divmod(num_normal_displacement_vectors, num_splits)
    
    print("Split size: %i" % split_size)
    print("Remaining: %i" % remaining)
    
    # Split normal vector array into these even splits and 
    # the last one containing the remaining ones if existing
    normal_displacement_vectors_to_split = normal_displacement_vectors[:num_splits*split_size]
    normal_displacement_vectors_remaining = normal_displacement_vectors[-remaining:]
    
    # Create array to hold all directional binary masks
    bm = np.zeros(ref_img.shape+(num_normal_displacement_vectors,), dtype=np.int)
        
    # Iterate over even batches of the normal vectors
    for split_num, v in enumerate(np.split(normal_displacement_vectors_to_split, num_splits)):
        print("Processing split %i/%i" % (split_num+1, num_splits))
        # Calculate the dot product between the field vectors and the normal vectors
        d = np.dot(field_data, v.T)
        # The product of the norms of the field vectors and normal vectors
        norms = np.expand_dims(np.linalg.norm(field_data, axis=-1), axis=-1)*np.linalg.norm(v, axis=-1)
        # Calculate the deviation in degress between the field vectors and normal vectors
        deviation_degrees = np.rad2deg(np.arccos(d/norms))
        # Create a boolean mask of where there is less than or equal to 20 degrees 
        # difference between field vectors and normal vectors
        #m = deviation_degrees < 20
        m = deviation_degrees <= args.angle_thr
        # Save the boolean mask as a binary mask in existing array
        bm[...,split_num*split_size:(split_num+1)*split_size][m] = 1
        print("Calculated directional masks")
    # Iterate over the remaining normal vectors if existing
    if remaining:
        print("Processing remainder")
        # Calculate the dot product between the field vectors and the normal vectors
        d = np.dot(field_data, normal_displacement_vectors_remaining.T)
        # The product of the norms of the field vectors and normal vectors
        norms = np.expand_dims(np.linalg.norm(field_data, axis=-1), axis=-1)*np.linalg.norm(normal_displacement_vectors_remaining, axis=-1)
        # Calculate the deviation in degress between the field vectors and normal vectors
        deviation_degrees = np.rad2deg(np.arccos(d/norms))
        # Create a boolean mask of where there is less than or equal to 20 degrees 
        # difference between field and normal vectors
        #m = deviation_degrees < 20
        m = deviation_degrees <= args.angle_thr
        # Save the boolean mask as a binary mask in existing array
        bm[...,-remaining:][m] = 1
        print("Calculated remaining directional masks")
    print("Number of directional masks calculated: %i" % bm.shape[-1])
    print("Saving directional binary masks to disk")
    """
    # Save all directional binary masks as a 4D nifti
    bm_img = nib.spatialimages.SpatialImage(bm, affine=ref_img.affine, header=ref_img.header)
    nib.save(bm_img, "directional-binary-masks.nii.gz")
    """
    # As well as the sum of the masks
    bm_max_img = nib.spatialimages.SpatialImage(np.max(bm, axis=-1), affine=ref_img.affine, header=ref_img.header)
    nib.save(bm_max_img, "directional-binary-masks-max.nii.gz")
    #sys.exit()
    #"""
    
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
    
    # Iterate over each normal vector, that we just created
    # directional binary masks for
    #for i in range(2, num_normal_displacement_vectors-1): # when num_vecs = 4 (for debug)
    for i in range(num_normal_displacement_vectors):
        print("Processing vector: %i/%i" % (i+1, num_normal_displacement_vectors))
        # Get the coordinates of the normal vector
        nv_c = normal_displacement_vectors_coordinates[i]
        print("Position: ", end='')
        print(nv_c)
        # Get the normal vector
        nv_d = normal_displacement_vectors[i]
        nv_d[-1] *= -1 # Invert z component. This was necesary in order to 
        # find correct end of brain mask maximum displacement. The reason for -1 is
        # because dispz3d contains a -1 term from before 
        # (in order to have correct of ANTs form later)
        print("Displacement: ", end='')
        print(nv_d)
        # Get the directional binary mask corresponding to this
        # normal vector
        bmi = bm[...,i]
        
        # TODO
        #np.savez("jp/cone-mask.npz", bmi)
        #np.savez("jp/cone-vectors.npz", field_data[bmi == 1])
        #np.savez("jp/cone-vectors-positions.npz", np.argwhere(bmi))
        
        # Find the maximum displacement possible along the nv_d vector 
        # between its starting position and the end of the brain mask
        n = 1
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
        p_max = (nv_c + (n-1)*nv_d).astype(np.int)
        print("Max displaced coordinates within brain mask: ", end='')
        print(p_max)
        # Calculate the magnitude of the maximum displacement
        disp_max = np.linalg.norm((p_max-nv_c).astype(np.float32))
        print("Max displacement within brain mask along vector: %f" % disp_max)
        
        # Scale the extent (by fraction) of the displacements reaching the end of the brain mask
        p_max = (nv_c + args.max_radial_displacement_to_brainmask_fraction*disp_max*nv_d).astype(np.int)
        print("Scaled max displaced coordinates within brain mask: ", end='')
        print(p_max)
        
        # Find the geometric center of the directional
        # binary mask BEFORE interpolation
        print("Finding bounding box for directional mask")
        bm_geom_center, bm_widths = bounding_box_mask(bmi)
        bmcx, bmcy, bmcz = bm_geom_center
        bmwx, bmwy, bmwz = bm_widths
        
        # Stretch the displacement field towards the skull,
        # using the directional binary mask (bmi), disp_max
        # and interpolation. To do this,
        # find the geometric center of the directional
        # binary mask AFTER inteprolation
        # Set voxel to 1 at maximum displaced voxel
        # indicating interpolation (stretched binary mask)
        bmi[p_max[0],p_max[1],p_max[2]] = 1
        print("Finding bounding box for directional mask with maximum displaced position added")
        bm_geom_center_interp, bm_widths_interp = bounding_box_mask(bmi)
        bmcx_interp, bmcy_interp, bmcz_interp = bm_geom_center_interp
        bmwx_interp, bmwy_interp, bmwz_interp = bm_widths_interp
        
        # For diagnostics, store old and intepolated (stretched) bounding box
        orig_bboxes_data[bmcx-bmwx//2:bmcx+bmwx//2, \
                         bmcy-bmwy//2:bmcy+bmwy//2, \
                         bmcz-bmwz//2:bmcz+bmwz//2, \
                         i] = 1
        
        interp_bboxes_data[bmcx_interp-bmwx_interp//2:bmcx_interp+bmwx_interp//2, \
                           bmcy_interp-bmwy_interp//2:bmcy_interp+bmwy_interp//2, \
                           bmcz_interp-bmwz_interp//2:bmcz_interp+bmwz_interp//2, \
                           i] = 1
        
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
        p_max = (nv_c + (n-1)*nv_d).astype(np.int)
        print("Max displaced coordinates within outer ellipsoid mask: ", end='')
        print(p_max)
        
        # Calculate the absolute max difference between p_max and nv_c
        diff_max_abs = np.abs((p_max-nv_c).astype(np.float32))
        print("Absolute max difference between intersection of vector on outer ellipsoid mask surface, and vector position: " , end='')
        print(diff_max_abs)
        
        # Extract the components of the vector position (inflection point)
        nv_cx, nv_cy, nv_cz = nv_c

        # Extract the components of the vector (displacement point)
        nv_dx, nv_dy, nv_dz = nv_d
        
        # Extract the components of the maximum displacement along 
        # nv_d within the outer ellipsoid mask
        diff_max_abs_x, diff_max_abs_y, diff_max_abs_z = diff_max_abs
        
        # Scaling parameter <0,1] : max_radial_displacement_to_outer_ellipsoid_mask_fraction
        # 0.1: Only using the value nearest inflection surface to intepolate.
        # Small value will lead to most rigid displacements.
        # 1: The entire radial range from inflection surface to outer ellipsoid
        # surface will be used for intepolation. Large value will lead to least
        # rigid / most elastic displacements.
        #scale_param = 0.5
        scale_param = args.max_radial_displacement_to_outer_ellipsoid_mask_fraction
        
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
        
        # Interpolate
        print("Interpolating x displacement")
        dxi = map_coordinates(dx, [xi.ravel(), yi.ravel(), zi.ravel()], order=args.spline_order)\
                                 .reshape(bmwx_interp, bmwy_interp, bmwz_interp)
        print("Interpolating y displacement")
        dyi = map_coordinates(dy, [xi.ravel(), yi.ravel(), zi.ravel()], order=args.spline_order)\
                                 .reshape(bmwx_interp, bmwy_interp, bmwz_interp)
        print("Interpolating z displacement")
        dzi = map_coordinates(dz, [xi.ravel(), yi.ravel(), zi.ravel()], order=args.spline_order)\
                                 .reshape(bmwx_interp, bmwy_interp, bmwz_interp)
        print("Interpolating Gaussian")
        gaussian_data_interp_part = map_coordinates(gaussian_data, [xi.ravel(), yi.ravel(), zi.ravel()], order=args.spline_order)\
                                                                  .reshape(bmwx_interp, bmwy_interp, bmwz_interp)
        
        print("Inserting into existing array")
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
        print("Inserting done")
        
        # Garbage collect
        """
        print("Garbage collect")
        del zi, yi, xi, bmi
        gc.collect()
        print("Garbage collect done")
        """
        #if i == 1:
        #    sys.exit() # TODO
    """
    print("Garbage collect again")
    gc.collect()
    print("Garbage collect again done")
    """
    # Save old and intepolated (stretched) bounding box to disk
    print("Saving bounding boxes for original directional binary masks to disk")
    orig_bboxes_img = nib.spatialimages.SpatialImage(np.max(orig_bboxes_data, axis=-1), affine=ref_img.affine, header=ref_img.header)
    nib.save(orig_bboxes_img, "original-bounding-box-vector-max.nii.gz")

    print("Saving bounding boxes for intepolated directional binary masks to disk")
    interp_bboxes_img = nib.spatialimages.SpatialImage(np.max(interp_bboxes_data, axis=-1), affine=ref_img.affine, header=ref_img.header)
    nib.save(interp_bboxes_img, "interp-bounding-box-vector-max.nii.gz")
    
    # Bounding box interpolation is now done.
    # Aggregate all interpolated data into a single vector field
    # using three final intepolations
    print("Building total intepolated field and Gaussian")
    
    # Take the mean or max over the last axis, excluding nan values
    print("Computing mean of non-nan displacement values")
    field_data_interp = np.nanmean(field_data_interp, axis=-1, dtype=np.float32)
    
    #print("Computing max of non-nan values")
    #field_data_interp_max = np.nanmax(field_data_interp, axis=-1)
    #print("Computing min of non-nan values")
    #field_data_interp_min = np.nanmin(field_data_interp, axis=-1)
    #print("Computing mean of these non-nan max and min values")
    #field_data_interp = np.nanmean(np.stack((field_data_interp_max, field_data_interp_min), axis=-1), axis=-1, dtype=np.float32)
    #print("Combining max and min of non-nan values to get the minimum of absolute length vectors")
    #field_data_interp = field_data_interp_min
    #field_data_interp[-field_data_interp_max < field_data_interp_min] = field_data_interp_max[-field_data_interp_max < field_data_interp_min]

    print("Computing mean of non-nan Gaussian values")
    gaussian_data_interp = np.nanmean(gaussian_data_interp, axis=-1, dtype=np.float32)

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
    #"""
    print("Smoothing interpolated x displacement")
    field_data_interp[...,0] = gaussian_filter(field_data_interp[...,0], sigma=args.smoothing)
    print("Smoothing interpolated y displacement")
    field_data_interp[...,1] = gaussian_filter(field_data_interp[...,1], sigma=args.smoothing)
    print("Smoothing interpolated z displacement")
    field_data_interp[...,2] = gaussian_filter(field_data_interp[...,2], sigma=args.smoothing)
    print("Smoothing interpolated Gaussian")
    gaussian_data_interp = gaussian_filter(gaussian_data_interp, sigma=args.smoothing)
    #"""
        
    # Finally, scale displacement fields with the specified intensity
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
    nib.save(ellipsoid_img_interp, "interp-"+args.eout)
    
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
    nib.save(outer_ellipsoid_img_interp, "interp-outer-"+args.eout)
    
    # Remove displacements from edges of bounding boxes that went outside of the brain
    field_data_interp[brainmask_data != 1] = 0
    gaussian_data_interp[brainmask_data != 1] = 0
    
    # Save original (non-intepolated) field
    print("Saving original fields")
    field_img = nib.spatialimages.SpatialImage(field_data, affine=ref_img.affine, header=ref_img.header)
    nib.save(field_img, args.fout)
    
    # Also save the negative of the field (since ITK-SNAP needs the negative visualize correctly)
    field_oppos_img = nib.spatialimages.SpatialImage(-field_data, affine=ref_img.affine, header=ref_img.header)
    nib.save(field_oppos_img, "neg-"+args.fout)
    
    # Save intepolated field
    print("Saving interpolated fields and Gaussian")
    field_img_interp = nib.spatialimages.SpatialImage(field_data_interp, affine=ref_img.affine, header=ref_img.header)
    nib.save(field_img_interp, "interp-"+args.fout)
    
    # Also save the negative of the field (since ITK-SNAP needs the negative visualize correctly)
    field_oppos_img_interp = nib.spatialimages.SpatialImage(-field_data_interp, affine=ref_img.affine, header=ref_img.header)
    nib.save(field_oppos_img_interp, "interp-neg-"+args.fout)

    gaussian_img_interp = \
    nib.spatialimages.SpatialImage(-gaussian_data_interp, affine=ref_img.affine, header=ref_img.header)
    nib.save(gaussian_img_interp, "interp-gaussian.nii.gz")

    print(sys.argv[0] + " done")
    
