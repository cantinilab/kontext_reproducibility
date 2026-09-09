# from sklearn.cluster import KMeans
# from sklearn.mixture import GaussianMixture
# import scanpy as sc 
import numpy as np 
# import pandas as pd


def mclust_clustering(n_clusters,seed,matrix):
    import os
        # os.environ['R_HOME'] = '/scbio4/tools/R/R-4.0.3_openblas/R-4.0.3'
    os.environ['R_HOME'] = '/opt/gensoft/exe/R/4.5.0/lib64/R'
    if n_clusters < 50:
        modelNames = 'EEE'    
    else:
        modelNames = 'EEV'
    np.random.seed(seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")
    r_random_seed = robjects.r['set.seed']
    r_random_seed(seed)
    rmclust = robjects.r['Mclust']
    from rpy2.robjects import default_converter, conversion, numpy2ri
    with conversion.localconverter(default_converter + numpy2ri.converter):
        r_res = rmclust(matrix,n_clusters, modelNames)
        mclust_res = np.array(r_res[-2])#.rx2('classification'))
    return(mclust_res)    

def leiden_clustering(resolution,seed,matrix,num_nn=50,num_iterations=-1):
    from banksy.main import LeidenPartition
    import leidenalg
    partitioner = LeidenPartition(
                        matrix,
                        num_nn = num_nn,
                        nns_have_weights = True,
                        compute_shared_nn = True,
                        filter_shared_nn = True,
                        # shared_nn_max_rank = options['shared_nn_max_rank '],
                        # shared_nn_min_shared_nbrs = options['shared_nn_min_shared_nbrs'],
                        # verbose = options['verbose'], 
                            )   

    label, modularity = partitioner.partition(
                                    resolution = resolution,
                                    partition_metric = leidenalg.RBConfigurationVertexPartition,
                                    n_iterations = num_iterations,
                                    seed = seed,)

    return(label.dense)


    


# def compute_kmeans_clustering(adata,n_clusters,seed,slot):
#     X = adata.obsm[slot]
#     kmeans = KMeans(n_clusters=n_clusters,random_state=seed).fit(X)
#     return(kmeans.labels_)

# def compute_gmm_clustering(adata,n_clusters,seed,slot,reg_covar=1e-5):
#     X = adata.obsm[slot]
#     gmm = GaussianMixture(n_components=n_clusters,random_state=seed,reg_covar=reg_covar).fit(X)
#     return(gmm.predict(X))

# def compute_louvain_clustering(adata,n_neighbors,seed,slot):
#     sc.pp.neighbors(adata,n_neighbors=n_neighbors,random_state=seed)
#     sc.tl.louvain(adata)
#     return(adata.obs['louvain'])

# def compute_leiden_clustering(adata,n_neighbors,slot,resolution=0.1,flavor='igraph',n_iterations=2,seed=None):
#     sc.pp.neighbors(adata,n_neighbors=n_neighbors,random_state=seed)
#     sc.tl.leiden(adata,resolution=resolution,flavor=flavor,n_iterations=n_iterations)
#     return(adata.obs['leiden'])

# def original_banksy_clustering(banksy_dict_concat,
#                                resolution,
#                                 num_nn=50,
#                                 num_iterations = -1,
#                                 partition_seed=1234,
#                                 match_labels=True):
#     resolutions = [resolution]
#     from banksy.cluster_methods import run_Leiden_partition

#     results_df, max_num_labels = run_Leiden_partition(
#         banksy_dict_concat,
#         resolutions,
#         num_nn = num_nn,
#         num_iterations = num_iterations,
#         partition_seed = partition_seed,
#         match_labels = match_labels,
#     )
#     return(results_df)

# def SEDR_mclust_R(adata, n_clusters,seed,slot):
#     """\
#     Clustering using the mclust algorithm.
#     The parameters are the same as those in the R package mclust.
#     """
#     import os
#     # os.environ['R_HOME'] = '/scbio4/tools/R/R-4.0.3_openblas/R-4.0.3'
#     os.environ['R_HOME'] = '/opt/gensoft/exe/R/4.5.0/lib64/R'
#     modelNames = 'EEE'
#     np.random.seed(seed)
#     import rpy2.robjects as robjects
#     robjects.r.library("mclust")
#     r_random_seed = robjects.r['set.seed']
#     r_random_seed(seed)
#     rmclust = robjects.r['Mclust']

#     from rpy2.robjects import default_converter, conversion, numpy2ri
#     with conversion.localconverter(default_converter + numpy2ri.converter):
#         r_res = rmclust(adata.obsm[slot], n_clusters, modelNames)
#         mclust_res = np.array(r_res[-2])#.rx2('classification'))
    
#     return(mclust_res)



    # adata.obs[key_added] = mclust_res
    # adata.obs[key_added] = adata.obs[key_added].astype('int')
    # adata.obs[key_added] = adata.obs[key_added].astype('category')

    # return adata

# def my_banksy_clustering(adata,resolution,num_nn=50,num_iterations=-1,partition_seed=1234,match_labels=True,slot=None):
#     if slot is not None:
#         pca_dim = adata.obsm[slot].shape[1]
#         adata.obsm[f'reduced_pc_{pca_dim}'] = adata.obsm[slot]    
#     banksy_dict = {'scaled_gaussian':{0.2:{'adata':adata}}}
#     results_df = original_banksy_clustering(banksy_dict,resolution=resolution,num_nn=num_nn,num_iterations=num_iterations,partition_seed=partition_seed,match_labels=match_labels)
#     return(results_df['labels'].values[0].dense)

# def get_clustering_params_from_cfg(cfg):
#     clusterings = []
#     for cl_key in cfg['clustering_keys'].split(','):
#         for seed in get_range_from_minmaxstep(cfg['clustering_params'][cl_key]['seeds']):
#             if cl_key == 'BANKSY_leiden':            
#                 for resolution in get_range_from_minmaxstep(cfg['clustering_params'][cl_key]['resolutions']):
#                     num_nn = cfg['clustering_params'][cl_key]['num_nn']
#                     num_iterations = cfg['clustering_params'][cl_key]['num_iterations']
#                     clusterings += [{'clustering_key':cl_key,
#                                     'clustering_params':{'partition_seed':int(seed),
#                                                          'resolution':np.float64(resolution),
#                                                          'num_nn':num_nn,
#                                                          'num_iterations':num_iterations,},
#                                     'need_n_clusters':False}]
#             else:
#                 clusterings += [{'clustering_key':cl_key,
#                                 'clustering_params':{'seed':int(seed)},
#                                 'need_n_clusters':True}]
#     return(clusterings)

# def get_clustering_function_from_key(key):
#     # clusterings 
#     if key == 'kmeans':
#         return(compute_kmeans_clustering)
#     if key == 'gmm':
#         return(compute_gmm_clustering)
#     if key == 'louvain':
#         return(compute_louvain_clustering)
#     if key == 'leiden':
#         return(compute_leiden_clustering)
#     if key == 'BANKSY_leiden':
#         return(my_banksy_clustering)
#     if key == 'mclust':
#         return(SEDR_mclust_R)

# def get_clustering_str(clustering_key,clustering_params):


#                 # clustering='kmeans',
#                 # n_clusters=None,
#                 # n_neighbors=None,
#                 # flavor='igraph',
#                 # n_iterations=None,
#                 # seed=0,):
#     clustering_str = 'bleiden' if clustering_key=='BANKSY_leiden' else clustering_key 
#     clustering_params_str = f'{clustering_str}'
#     if clustering_key in ['kmeans','gmm','mclust']:
#         seed = clustering_params['seed']
#         n_clusters = clustering_params['n_clusters']
#         clustering_params_str += f'_s{seed}_nc{n_clusters}'
#     if clustering_key in ['louvain','leiden']:
#         seed = clustering_params['seed']
#         n_neighbors = clustering_params['n_neighbors']
#         clustering_params_str += f'_s{seed}_nn{n_neighbors}'
#     if clustering_key == 'BANKSY_leiden':
#         resolution = clustering_params['resolution']
#         num_nn = clustering_params['num_nn']
#         num_iterations = clustering_params['num_iterations']
#         partition_seed = clustering_params['partition_seed']
#         clustering_params_str += f'_s{partition_seed}_res{resolution:.2f}_nn{num_nn}'
#     return(clustering_str,clustering_params_str)

# def get_clustering_param_from_str(clustering_str,params_str):     
#     try: 
#         if clustering_str in ['gmm','mclust']:
#             clustering_key = 'gmm' if clustering_str=='gmm' else 'mclust'
        
#             seed = params_str.split(clustering_str)[1].split('_s')[1].split('_')[0]
#             nc = params_str.split('_nc')[1].split('.csv')[0]
            
#             clustering_params = {
#                 'seed':int(seed),
#                 'n_clusters':int(nc)}
        
#         if clustering_str == 'bleiden':
#             clustering_key = 'BANKSY_leiden'

#             cl_params = params_str.split(clustering_str)[1]
#             seed = cl_params.split('_s')[1].split('_')[0]
#             resolution = cl_params.split('_res')[1].split('_')[0]
#             num_nn = cl_params.split('_nn')[1].split('.csv')[0]

#             clustering_params = {
#                     'resolution':resolution,
#                     'num_nn':num_nn,
#                     'num_iterations':-1,
#                     'partition_seed':seed,}

#         return({'clustering_key':clustering_key,
#                     'clustering_params':clustering_params})
#     except Exception as e:
#         print("Error in get clustering params")
#         print(e)
#         print('clustering_str',clustering_str,'params_str',params_str)
#         raise

# def add_clustering_parameters_to_df_results(df):
#     if len(df)>0:

#         dfs = []
#         for clustering in df['clustering_key'].unique():
#             df_ = df.loc[df['clustering_key']==clustering]
#             if clustering == 'BANKSY_leiden':    
#                 df_['resolution'] = df_['clustering_params'].apply(lambda x: get_clustering_param_from_str('bleiden',x)['clustering_params']['resolution'])
#                 df_['partition_seed'] = df_['clustering_params'].apply(lambda x: get_clustering_param_from_str('bleiden',x)['clustering_params']['partition_seed'])
#             if clustering in ['gmm','mclust']:
#                 df_['nc'] = df_['clustering_params'].apply(lambda x: get_clustering_param_from_str(clustering,x)['clustering_params']['n_clusters'])
#                 df_['seed'] = df_['clustering_params'].apply(lambda x: get_clustering_param_from_str(clustering,x)['clustering_params']['seed'])
#             dfs += [df_]
#         df = pd.concat(dfs)

#     return(df)


