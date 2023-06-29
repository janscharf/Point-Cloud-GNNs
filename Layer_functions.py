import tensorflow as tf

#Implementation of DGCNN, GravNet, and GarNet Layers for use with the tensorflow keras functional api made by Jan Scharf
#This code includes a reworked code of https://github.com/jkiesele/caloGraphNN and https://github.com/WangYueFt/dgcnn

#Potentialfunctions

def gauss(x):
    return tf.math.exp(-1* x*x)

def gauss_of_lin(x):
    return tf.math.exp(-1*(tf.math.abs(x)))





#Calculate Euclidean Distances:

def euclidean_squared(A, B):
    """
    Returns the euclidean distance (a-b)^2=a^2+b^2-2ab for every element in A, B. The input Dimensions are A=(BS,V,F) and B=(BS,V',F), with BS being the batch_size, V and V' are the number of nodes
    in the specific Datapoint and F being the dimensions of spatial features. The input should be of the dtype float64 or float32. The output Dimensions are (BS, V, V') with every entry [i,j] being
    the euclidean suqared distance of node i from V and node j from V' over all spatial features F, so =sum_i(a_i-b_i)^2. This is mostly used for A=B so V=V' and we get the distances from all nodes
    in a Datapoint, with the diagonal being the selfdistance (so 0)."
    """
    shape_A=A.shape.as_list()
    shape_B=B.shape.as_list()

    assert (A.dtype == tf.float32 or A.dtype == tf.float64) and (B.dtype == tf.float32 or B.dtype == tf.float64)
    assert len(shape_A) == 3 and len(shape_B) == 3
    assert shape_A[0] == shape_B[0] #same amount of data in a Batch


    #Calculate the euclidean squared distance:
    two_factor  = -2* tf.linalg.matmul(A, tf.transpose(B, perm=[0, 2, 1])) #-2ab term, (B,V,F)*(B,F,V')->(B,V,V')
    a_squared   = tf.expand_dims( tf.math.reduce_sum(A * A, axis=2), axis=2) #a^2 term, (B,V,F)^2->(B,V)^2->(B,V,1)^2 , ^2 means every value is squared, so every entry of (B,V,1) is sum_i(a_i^2)
    b_squared   = tf.expand_dims( tf.math.reduce_sum(B * B, axis=2), axis=1) #b^2 term, (B,V',F)^2->(B,V')^2->(B,1,V')^2
    return tf.math.abs(two_factor + a_squared + b_squared) #Add all the terms, (B,V,V')+(B,V,1)^2+(B,1,V')^2->(B,V,V')




#Calculate kNN:

def nearest_neighbor_matrix(spatial_features, k=10):
    """
    Returns the indices and values of the kNN to all nodes in the shape (B,V,k). The input are the spatial_features of every node (B,V,F), B being the batch_size, V the number of nodes in the specific
    Datapoint and F being the dimensions of the spatial features. This function gets all the distances between all nodes of a Datapoint from euclidean_squared and cuts the k shortest distances. It
    returns the indices and values of these shortest connections of every node.
    """

    shape   = spatial_features.shape.as_list()

    assert spatial_features.dtype == tf.float32 or spatial_features.dtype == tf.float64
    assert len(shape)==3

    D = euclidean_squared(spatial_features, spatial_features) #Calculate distances of nodes (B,V,F),(B,V,F)->(B,V,V)
    D, N = tf.math.top_k(-D, k+1, sorted=True) #Calculate kNN and self distance 0, so the first entry, returns tensor of Distances D and indices N in shape (B,V,k+1), -D so that longer distances are cut off
    return N[:,:,1:], -D[:,:,1:] #-D so distance are not negative, dimensions are (B,V,k), cuts of the self distance





#Expand indexing tensor for GravNet:

def indexing_tensor(spatial_features, k=10, n_batch=-1):
    """
    Returns an indexing tensor that combines the kNN index from nearest_neighbor_matrix with a index describing the position of the Datapoint in the batch and the kNN distance tensor,
    in shapes (B,V,k,2) and (B,V,k). As input it takes the spatial_features in (B,V,F) and a amount of neighbors k that is used, set to 10 when not provided. Idk what n_batch does. This
    is only needed for the GravNet Layer.
    """

    shape_spatial_features = tf.shape(spatial_features) #This is writen so long, so it does not rais an error for mixing up the data types
    n_batch = shape_spatial_features[0] #gets the amount of Datapoints in a Batch
    n_max_entries = shape_spatial_features[1] #gets the max amount of nodes in a Datapoint in the specific Batch

    assert len(spatial_features.shape.as_list()) == 3
    assert spatial_features.dtype == tf.float32 or spatial_features.dtype == tf.float64

    neighbor_matrix, distance_matrix = nearest_neighbor_matrix(spatial_features, k) #gets the indices and values of kNN of every node, (B,V,F),k->(B,V,k)

    batch_range = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.range(0, n_batch, 1), axis=1), axis=1), axis=1) # creats batch index tensor (B,1,1,1), represents the indexing of each Datapoint in the Batch, is needed for calculations collapse_to_vertex tf.gather_nd in GravNet
    batch_range = tf.tile(batch_range, [1, n_max_entries, k, 1]) #Copy entries so often that (B,1,1,1)->(B,V,k,1) to get same size as expanded_neighbor_matrix
    expanded_neighbor_matrix = tf.expand_dims(neighbor_matrix, axis=3) #Add dim (B,V,k)->(B,V,k,1), so that we are able to concatonate with batch_range

    indexing_tensor = tf.concat([batch_range, expanded_neighbor_matrix], axis=3) #Combine index tensors (B,V,k,1)+(B,V,k,1)->(B,V,k,2), so that every entry is (Datapoint_index, kNN_index)

    return tf.cast(indexing_tensor, tf.int64), distance_matrix #returns the combine index tensor as int64 and the tensor of the kNN distances, (B,V,k,2),(B,V,k)





#Calculate the new aggregated and distance weighted spatical features for the GarNet:

def apply_edges(vertices, edges, reduce_sum=True, flatten=True, expand_first_vertex_dim=True, aggregation_function=tf.math.reduce_max):
    """
    Applies the update function as described in the paper f=f*V(d). Inputes spatial features of the vertices (B,V',F') and edges (B,V,V',F) they are attached to. Edges can contain more then one
    feature F, but in practice of the GarNet only consist of one feature F=1 the distance of node V and V'. It returns the new aggregated and distance weighted spatial features
    for the nodes V (B,V,F''=F'+F). It is only used in GarNet. Specificly the input vertices only describe the connceted vertices v_j and not the node v_k we are trying to learn.
    This node v_k is only provided by the edges, as it is part of V and v_j is part of V'. So the input vertices are only the aggregators.
    """

    edges = tf.expand_dims(edges, axis=3) #(B,V,V',F)->(B,V,V',1,F)
    if expand_first_vertex_dim:
        vertices = tf.expand_dims(vertices, axis=1) #(B,V',F')->(B,1,V',F')
    vertices = tf.expand_dims(vertices, axis=4) #(B,1,V',F')->(B,1,V',F',1), so same rank as edges

    out = edges*vertices #(B,V,V',1,F)*(B,1,V',F',F)->(B,V,V',F',F), calculate f=f*V(d) as described in the paper

    if edges.shape[1]==None: #Define the second dimension for the final output, otherwise this causes problems if the second dimension is None
        F_2 = -1 #None would cause a problem
    else:
        F_2 = edges.shape[1] #if the dimensions has a defined number

    if reduce_sum:
        out = aggregation_function(out, axis=2) #(B,V,V',F',F)->(B,V,F',F), aggregate over all the connected nodes with a specific aggregation_function
    if flatten:
        if reduce_sum: #edges (B, V, V', 1, F) and vertices (B, 1, V', F', 1)
            F_out = edges.shape[-1] * vertices.shape[-2] #Define the thrid dimension for the final output, needs to be defined and can't use -1 since the second dimension is possibly already using -1
            assert (F_out == out.shape[-1]* out.shape[-2]) #Check if F_out is right
        else:
            F_out = edges.shape[-1] * vertices.shape[-2] * vertices.shape[-3] #same as prior for the case reduce_sum=False
            assert (F_out == out.shape[-1] * out.shape[-2] * out.shape[-3]) #Check if F_out is right
        out = tf.reshape(out, shape=[tf.shape(out)[0], F_2 , F_out]) #reshape the output to fit spatial_features shape of (B,V,F), (B,V,F',F)->(B,V,F'*F)
    return out


# Define high_dim_dense as ConvLayer:

def high_dim_dense(inputs, nodes, **kwargs):
    """
    This provides Conv Layers as high dim dense for the specific len of the input.
    """
    if len(inputs.shape.as_list())==3:
        conv_layer1d = tf.keras.layers.Conv1D(nodes, kernel_size=(1), strides=(1), padding='valid', **kwargs)
        outputs = conv_layer1d(inputs)
        outputs.set_shape([None, None, nodes]) #To specify the output demensions, so it does not return (None,None,None) which will cause a problem a later on
        return outputs
    elif len(inputs.shape.as_list())==4:
        conv_layer2d = tf.keras.layers.Conv2D(nodes, kernel_size=(1,1), strides=(1,1), padding='valid', **kwargs)
        outputs = conv_layer2d(inputs)
        outputs.set_shape([None, None, None, nodes])
        return outputs
    elif len(inputs.shape.as_list())==5:
        conv_layer3d = tf.keras.layers.Conv3D(nodes, kernel_size=(1,1,1), strides=(1,1,1), padding='valid', **kwargs)
        outputs = conv_layer3d(inputs)
        outputs.set_shape([None, None, None, None, nodes])
        return outputs
    else:
        print("Warning! Inputs is too long to be turned into a high_dim_dense.")










#The actuall layers based on the Layers API of keras:


def layer_GarNet(vertices_in, n_aggregators, n_filters, n_propagate, plus_mean=True):
    """
    The implementation of a GarNet-Layer with inputs vertices_in (B,V,F), n_aggregators=dim(S), n_filters=dim(F_OUT) and n_propagate=dim(F_LR). It returns the new spatial features (B,V,F_OUT).
    For a specifc description of how a GarNet-Layer works, see ???.
    """
    #Calculate learned embeding S and F_LR:
    vertices_in_orig = vertices_in #save input for the self-loop later on
    vertices_in = tf.keras.layers.Dense(n_propagate, activation=None)(vertices_in) #Calculates F_LR (B,V,F_IN)->(B,V,n_propagate=F_LR)

    agg_nodes = tf.keras.layers.Dense(n_aggregators, activation=None)(vertices_in_orig) #Calulates S (B,V,F_IN)->(B,V,n_aggregators=S), calculate the distance from every node to S dummy nodes
    agg_nodes = gauss_of_lin(agg_nodes) #apply GarNet potential on distances between nodes and dummy nodes (B,V,S)->(B,V,S)
    vertices_in = tf.concat([vertices_in, agg_nodes], axis=-1) #combine F_LR with S (B,V,F_LR)+(B,V,S)->(B,V,F_LR+S)

    #Create spatial features of dummy nodes

    edges = tf.expand_dims(agg_nodes, axis=3) #rehsape S (distances) to match shape of edges in apply_edges (B,V,S)->(B,V,S,1)
    edges = tf.transpose(edges, perm=[0, 2, 1, 3]) #permuate edges so it fits the order of apply_edges (B,V,S,1)->(B,S,V,1)

    vertices_in_collapsed = apply_edges(vertices_in, edges, reduce_sum=True, flatten=True) #create spatial features tensor for the dummy nodes by providing the connected edges and their aggregator vertices_in, (B,V,F_LR+S),(B,S,V,1)->(B,S,V,F_LR+S,1)->(B,S,F_LR+S,1)->(B,S,F_LR+S*1)
    vertices_in_mean_collapsed = apply_edges(vertices_in, edges, reduce_sum=True, flatten=True, aggregation_function=tf.math.reduce_mean) #same as line prior, just uses mean as aggregation method instead of max
    vertices_in_collapsed = tf.concat([vertices_in_collapsed, vertices_in_mean_collapsed], axis=-1) #Combine aggregation methods (B,S,(F_LR+S)_max)+(B,S,(F_LR+S)_mean)->(B,S,F'=2*(F_LR+S))

    #Get new spatial features for node while using the dummy nodes as aggregators

    edges = tf.transpose(edges, perm=[0, 2, 1, 3]) #perm edgdes so they fit apply_edges to use dummy nodes as aggregators (B,S,V,1)->(B,V,S,1)

    expanded_collapsed = apply_edges(vertices_in_collapsed, edges, reduce_sum=False, flatten=True) #create new spatial feature tensor for nodes while using the dummy nodes as aggregators, no aggregation is performed rather all information is saved and flattend into the spatial features (B,S,F'),(B,V,S,1)->(B,V,S,F',1)->(B,V,S*F'=S*2*(F_LR+S))
    expanded_collapsed = tf.concat([vertices_in_orig, expanded_collapsed, agg_nodes], axis=-1) #combine all F'_LRs and self-loop, scaled S (B,V,F)+(B,V,S+F')+(B,V,S)->(B,V,F+2S*(F_LR+S)+S)

    merged_out = high_dim_dense(expanded_collapsed, n_filters, activation='gelu') #Calculates the final new spatial feature tensor (B,V,n_filters=F_OUT), uses high_dim_dense/Conv as F+2S*(F_LR+S)+S=F+2S²+2SF_LR+S can be high dim and a normal dense layer will be computational ineffectiv

    return merged_out


def layer_GravNet(vertices_in, n_neighbours, n_dimensions, n_filters, n_propagate, activation_function_for_F_OUT='tanh'):
    """
    The implementation of a GravNet-Layer with inputs vertices_in (B,V,F), n_neighbours=dim(k), n_dimensions=dim(S), n_filters=dim(F_OUT), n_propagate=dim(F_LR) and the definition
    of the activation_function_for_F_OUT, that is used for the last step. When not provided then it uses tanh as used in the original code. It returns the new spatial feature tensor (B,V,F_OUT).
    """

    #Calculate F_LR and S:
    vertices_prop = high_dim_dense(vertices_in, n_propagate, activation=None) #Calculates F_LR (B,V,F)->(B,V,n_propagate=F_LR)
    neighb_dimensions = high_dim_dense(vertices_in, n_dimensions, activation=None) #Calculates S (B,V,F)->(B,V,n_dimensions=S), new spatial representation for distance calculation


    #Define function that calculates the aggregated distance weighted new spatial features from F_LR, the distances to the neighbours and the indices of neighbours
    def collapse_to_vertex(indexing, distance, vertices):
        """
        This function takes the indexing (B,V,k,2), distances (B,V,k) and vertices (B,V,F) as inputs and computes f=f*V(d). It returns the new learned and distance weighted features for the aggregation
        method of max and mean (B,V,F_LR'+F_LR'').
        """
        neighbours = tf.gather_nd(vertices, indexing) #get neighbour features through the kNN indexing (B,V,F) X (B,V,k,2) -> (B,V,k,F), needs Datapoint indexing for the gather_nd function
        distance = tf.expand_dims(distance, axis=3) #(B,V,k)->(B,V,k,1)
        distance = distance*10 #spread distances, allows for a better spread for the activation function. Proposed for the usage of tanh, needs to be checked for other activation functions. Should maybe be batch normed after high_dim_layer
        edges = gauss(distance) #scales the distances to be the edge features with the proposed gauss potential
        scaled_feat = edges * neighbours #Calculates f=f*V(d), (B,V,k,1)*(B,V,k,F)->(B,V,k,F)
        collapsed = tf.math.reduce_max(scaled_feat, axis=2) #Aggregat max over neighbours (B,V,k,F)->(B,V,F)
        collapsed_mean = tf.math.reduce_mean(scaled_feat, axis=2) #same as line prior, just with mean as aggregation method
        collapsed = tf.concat([collapsed, collapsed_mean], axis=-1) # combine aggregation methods (B,V,F_max)+(B,V,F_mean)->(B,V,F_max+F_mean)
        return collapsed


    #Get kNN and calculate new aggregated distance weighted spatial features
    indexing, distance = indexing_tensor(neighb_dimensions, n_neighbours) #get indices and distances of kNN ->(B,V,k,2),(B,V,k)
    collapsed = collapse_to_vertex(indexing, distance, vertices_prop) #get new aggregated distance weighted spatial features
    updated_vertices = tf.concat([vertices_in, collapsed], axis=-1) #add self-loop

    return high_dim_dense(updated_vertices, n_filters, activation=activation_function_for_F_OUT) #calculate F_OUT -> (B,V,n_filters=F_OUT)





def layer_DGCNN(vertices_in, n_filters, k=10,  activation_function='relu', add_mean=False, add_x_j_info=False):
    """
    The implementation of a DFCNN-Layer with inputs vertices_in (B,V,F), n_filters=dim(F_OUT), the activation function that should be used, if the mean to all k neighbours
    should be calculated as well (then F_OUT will be 2*n_filters) and if h(x_i,x_j-x_i,x_j) should be used with the added x_j. It returns the new spatial features (B,V,F_OUT).
    """

    def get_graph_feature(vertices, k=10, add_x_j_info=False):
        """
        This function gets the collapsed tensor x_i, x_j-x_i. As input it uses the original vertices (B,V,F), the amount of neighbours and if a x_j component should be added. It starts by
        getting the indexing for the kNN (B,V,k), then gets all the neighbour features (B,V,k,F') and expands the original vertices so that they can be subtractet to the neighbour features,
        so (B,V,F)->(B,V,k,F). Then it combines the information of x_i with the distance x_j-x_i, here a x_j component can be added as well. It returns the collapsed tensor x_i, x_j-x_i so
        (B,V,k,F+(F'-F) /+F').
        """
        indexing, _ = indexing_tensor(vertices, k) #Get kNN indexing. (B,V,F)+k->(B,V,k,2)
        neighbours = tf.gather_nd(vertices, indexing) #(B,V,F) X (B,V,k,2) -> (B,V,k,F'), retrievs neighbour features by getting the neighbours provided by indexing form the original vertices
        vertices_i = tf.expand_dims(vertices, axis=2) #(B,V,F)->(B,V,1,F), expand vertices features to fit neighbours shape
        vertices_i = tf.tile(vertices_i, [1,1,k,1]) #(B,V,1,F)->(B,V,k,F), copy the vertices eatures for every neighbour
        collapsed = tf.concat([vertices_i, (neighbours-vertices_i)], axis=-1) #(B,V,k,F)+(B,V,k,F'-F) -> (B,V,k,F+(F'-F)), calculate the collapsed tensor by combining the original vertices features with the feature distances to their neighbours
        if add_x_j_info: #this adds an extra x_j component
            collapsed = tf.concat([collapsed, neighbours], axis=-1)
        return collapsed



    collapsed = get_graph_feature(vertices_in, k, add_x_j_info) #(B,V,F) -> (B,V,k,2/3F), get the collapsed tensor from the input vertices by using the get_graph_feature function


    edges = high_dim_dense(collapsed, n_filters, activation=None)
    edges = tf.keras.layers.BatchNormalization()(edges)
    edges = tf.keras.layers.Activation(activation_function)(edges)#(B,V,k,2/3F) -> (B,V,k,n_filter), this is an equivalent to the MLP propsed to calculate the edge features in the paper, conv2d layer without activation + batch_norm + activation_function


    reduced = tf.math.reduce_max(edges, axis=2) #(B,V,k,n_filters) -> (B,V,n_filters), aggregates over the edges by using max

    if add_mean: #adds another aggregator using the mean function
        reduced_mean = tf.math.reduce_mean(edges, axis=2)
        reduced = tf.concat([reduced, reduced_mean], axis=-1) #the mean aggregation is concat to the max aggregation ->(B,V,n_filters*2)


    return reduced #returns the aggregated edges that correspond to the new spatial features for the next layer, shape (B,V,n_filters*1/2)
