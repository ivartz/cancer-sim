import numpy as np

vectors = np.load("normal_displacement_vectors.npz")["arr_0"]
positions = np.load("normal_displacement_vectors_coordinates.npz")["arr_0"]

#print(vectors.shape)
#print(positions.shape)

# Number of vectors to select
num_sel = 100

# Select a random vector and its position as the first
# of the num_sel selected vectors
randi = np.random.choice(len(vectors), 1)
vectors_sel = vectors[randi]
positions_sel = positions[randi]

vectors_sel = np.append(vectors_sel, [np.array([0.2,0.3,0.1])], axis=0)
    

print("Selected vector and position %i/%i" % (1,num_sel))

# Store first selected
#vectors_sel = np.expand_dims(v, axis=0)
#positions_sel = np.expand_dims(p, axis=0)

for i in range(1, num_sel):
    
    
    # Calculate the dot product between all normal vectors and selected normal vectors
    d = np.dot(vectors, vectors_sel.T)
    # The product of the norms of all normal vectors and selected normal vectors
    norms = np.expand_dims(np.linalg.norm(vectors, axis=-1), axis=-1)*np.linalg.norm(vectors_sel, axis=-1)
    # Calculate the deviation in degress between all normal vectors vectors and selected normal vectors
    deviation_degrees = np.rad2deg(np.arccos(d/norms))
    
    mean_deviation_degrees = np.mean(deviation_degrees, axis=-1)
    
    maxi = np.nanargmax(mean_deviation_degrees)
    maxdev = mean_deviation_degrees[maxi]
    v = vectors[maxi]
    p = positions[maxi]

    print("Selected vector and position %i/%i" % (i+1,num_sel))
    
    #print(maxi)
    #print(maxdev)
    #print(v)
    #print(p)
    
    # Store selected
    vectors_sel = np.append(vectors_sel, [v], axis=0)
    positions_sel = np.append(positions_sel, [p], axis=0)
