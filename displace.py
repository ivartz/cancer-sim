import argparse
import numpy as np
import sys
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt

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

"""
def rbf_interpolate_1d(data_flat, positions):
            
    # Get the resulting dimensions of the stacked data
    # for later use in the grid defition
    tdim, zdim, ydim, xdim = data.shape
    
    # Flatten the stacked data, for use in Rbf
    data_flattened = flatten_tensor(data)
    
    # Get the colums in the flattened data
    # The voxel values
    f = data_flattened[:,0]
    # Time coordinates of the voxel values
    t = data_flattened[:,1]
    # Z coordinates of the voxel values
    z = data_flattened[:,2]
    # Y coordinates of the voxel values
    y = data_flattened[:,3]
    # X coordinates of the voxel values
    x = data_flattened[:,4]
    
    # Make grids of indices with resolutions we want after the interpolation
    grids = [np.mgrid[time_idx:time_idx+1:1/interval_duration, 0:zdim, 0:ydim, 0:xdim] \
    for time_idx, interval_duration in enumerate(intervals_between_volumes_t)]
    
    # Stack all grids
    TI, ZI, YI, XI = np.hstack(tuple(grids))
    
    # Create radial basis functions
    #rbf_clinst = Rbf(t, z, y, x, f, function="multiquadric", norm='euclidean')
    rbf = Rbf(t, z, y, x, f, function='multiquadric') # If scipy 1.1.0 , only euclidean, default
    
    # Interpolate the voxel values f to have values for the indices in the grids,
    # resulting in interpolated voxel values FI
    # This uses the Rbfs
    FI = rbf(TI, ZI, YI, XI)
    
    data_interpolated = FI
    
    return data_interpolated
"""
if __name__ == "__main__":
    CLI=argparse.ArgumentParser()
    CLI.add_argument(
      "--of",
      help="Output folder",
      type=str,
      default="dispout",
    )
    args = CLI.parse_args()
    
    # The cross-sectional width of the 1D displacement
    w = 300
    
    # The cross-sectional width/2 of a Gaussian
    ghw = 30
    
    # The cross-section spatial data points
    resolution = w/(2*ghw)
    x = np.linspace(-ghw, ghw, w)
    
    # A Gaussian normal distributioun function evaluated 
    # at the cross-section spatial points
    g = gaussian_norm(x)
    
    # The gradient (or partial derivative with respect to x) of the
    # Gaussian function
    gx = np.gradient(g)
    
    # Scale the gradient such that the largest value is 1
    gx = gx/np.max(gx)
    
    # Make a normalized version of the (1D) gradient
    #gx_norm = gx/gx
    
    #for n in range(2):
    
    # Flatten the gradient data
    f = flatten_tensor(gx)
    
    # Insert correct spatial points 
    f[:,1] = x
    
    # f is now the columnar form of an 1D vector field
    
    # Extract the number of evaluation points
    l = len(f)
    #print(l)
    # Determine the number of steps to simulate the field
    num_steps = ghw-1

    # Make a copy of the field that is to be displaced according to
    # itself
    sim = f.copy()

    inflectionpoint = 1
    
    interpolation_cut = 2
    
    sim_interpol = sim.copy()
    
    rbfi = Rbf(np.arange(l)+1, sim_interpol[:,0])
        
    xi1 = np.linspace((l//2)-(interpolation_cut+inflectionpoint)*resolution, l//2, np.int(l//2))
    xi2 = np.linspace(1+l//2, 1+(l//2)+(interpolation_cut+inflectionpoint)*resolution, np.int(l//2))
    
    di1 = rbfi(xi1)
    
    di2 = rbfi(xi2)
        
    d = np.concatenate((di1, di2))
    #print(d.shape)

    # Save plot of the original field
    xlim = ghw
    
    plt.figure()
    plt.xlim(-xlim,xlim)
    plt.ylim(-1.2,1.2)
    #plt.plot(sim[:,1], sim[:,0])
    plt.plot(f[:,1], d)
    plt.savefig(args.of+"/1.png")
    
    for n in range(num_steps):
        print("Adding displacement")
        # Subtracting to have the correct sign of adding outward displacement
        # The spatial location of each field vector is displaced
        # according to 
        # the normalized vector at that point
        #sim[:,1] -= sim[:,0]/np.abs(sim[:,0])
        # the vector at that point
        sim[:,1] -= sim[:,0]
        
        sim_interpol = sim.copy()
        
        inflectionpoint = 2+n
    
        rbfi = Rbf(np.arange(l)+1, sim_interpol[:,0])
            
        #xi1 = np.linspace((l//2)-(3+ipointcord)*resolution, (l//2)-ipointcord*resolution, (l//2)-np.int(ipointcord*resolution))
        #xi1 = np.linspace((l//2)-(1+ipointcord)*resolution, l//2, (l//2)-np.int(ipointcord*resolution))
        #xi1 = np.linspace((l//2)-(1+ipointcord)*resolution, (l//2)+(1+ipointcord)*resolution, l-2*np.int(ipointcord*resolution))
        #xi1 = np.linspace((l//2)-(1+ipointcord)*resolution, (l//2)+(1+ipointcord)*resolution, l)
        xi1 = np.linspace((l//2)-(interpolation_cut+inflectionpoint)*resolution, l//2, np.int((l//2)-(n+1)*resolution))
        xi2 = np.linspace(1+l//2, 1+(l//2)+(interpolation_cut+inflectionpoint)*resolution, np.int((l//2)-(n+1)*resolution))
        #xi1 = np.linspace((l//2)-(3+ipointcord)*resolution, l//2, np.int(l//2))
        #xi1 = np.mgrid[(l//2)-(5+ipointcord)*resolution:l//2]
        
        
        di1 = rbfi(xi1)
        
        di2 = rbfi(xi2)
        
        dmiddle = np.zeros(np.int(2*(n+1)*resolution))
        
        print(di1.shape)
        print(di2.shape)
        print(dmiddle.shape)
        
        d = np.concatenate((di1, dmiddle, di2))

        #print(d.shape)
        # Save plot of the displaced field
        plt.figure()
        plt.xlim(-xlim,xlim)
        plt.ylim(-1.2,1.2)
        #plt.xlim(1,l)
        #plt.xlim(1,l//2)
        #plt.plot(np.arange(l//2), sim_interpol[:,0])
        #plt.plot(sim_interpol[:,1], sim_interpol[:,0])
        #plt.plot(sim_interpol[:,0])
        print(f[:,1].shape)
        print(d.shape)
        plt.plot(f[:,1],d)
        #plt.plot(sim[:,1], sim[:,0])
        plt.savefig(args.of+"/"+str(n+2)+".png")
    #plt.show()
    #print(dx_f.shape)
    
    
    #dx_new = np.interp(x, dx_f[:,1], dx_f[:,0])
    
    #XI, YI = np.meshgrid(x, np.linspace(-1,1,100))
    
    #xx, yy = dx_f[:,1], dx_f[:,0]

    # Create radial basis functions
    #rbf = Rbf(xx, yy, function='gaussian', epsilon=0.1)
    #rbf = Rbf(xx, yy, function='gaussian')
    #rbf = Rbf(xx, yy, epsilon=1)
    #"""
    #rbf = Rbf(f[:,1], f[:,0], epsilon=0.02)
    #rbf = Rbf(f[:,1], f[:,0])
        
    # Interpolate
    #xi = np.linspace(-ghw, ghw, 100)
    #FI = rbf(xi)
    #"""
    #print(FI.shape)

    #plt.figure()
    
    #plt.plot(x, g)
    #plt.plot(x, dx)
    #plt.plot(f[-l:,1], f[-l:,0])
    #plt.plot(f[:,1], f[:,0])
    
    #plt.figure()
    #plt.plot(x, dx_sim)
    #plt.plot(xi, FI)
    #plt.plot(x, FI)
    #plt.plot(x, dx_norm)
    
    #plt.show()
    
