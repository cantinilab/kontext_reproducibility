from pathlib import Path
from time import time
import tracemalloc
import json
import pandas as pd 
import scanpy as sc 
import torch

from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score,homogeneity_score,completeness_score
from kontext.clustering import gmm_clustering,kmeans_clustering
from utils.clustering import mclust_clustering,leiden_clustering


def compute_SEDR_embedding(adata,seed,n_components=200,n_neighbors=12):
    import SEDR
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEDR.fix_seed(seed)

    from sklearn.decomposition import PCA  # sklearn PCA is used because PCA in scanpy is not stable.
    adata_X = PCA(n_components=n_components, random_state=seed).fit_transform(adata.X)
    adata.obsm['X_pca'] = adata_X

    graph_dict = SEDR.graph_construction(adata, n_neighbors)
    # print(graph_dict)
    sedr_net = SEDR.Sedr(adata.obsm['X_pca'], graph_dict, mode='clustering', device=device)
    using_dec = True
    if using_dec:
        sedr_net.train_with_dec(N=1)
    else:
        sedr_net.train_without_dec(N=1)
    sedr_feat, _, _, _ = sedr_net.process()
    adata.obsm['E_pca'] = sedr_feat
    return(adata)


def compute_GraphST_embedding(adata,seed):
    
    from GraphST import GraphST
    import torch

    ref_datatypes = {'DLPFC':'10X',
                     'MOSTA':'Stereo',
                     'ZESTA':'Stereo',
                     'ARTISTA':'Stereo',
                     'MBA':'Slide'#Actually MERFish but it is based on GraphST options
            }
    dataset = adata.uns['data']['dataset']
    datatype = ref_datatypes[dataset]
    sc.pp.filter_genes(adata, min_cells=3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = GraphST.GraphST(adata, device=device,random_seed=seed,datatype=datatype,)


    # train model
    adata = model.train()
    print('pca')
    pca_of_mat(adata, adata.obsm['emb'], n_comps=50, key="E")
    print(adata)
    return(adata)


def original_banksy_matrix(adata,k_geom = 18,nbr_weight_decay='scaled_gaussian',m=0,param_lambda = 0.2):

    lambda_list = [param_lambda]
    xy_coord = 'spatial'
    x_coord = 'x_array'
    y_coord = 'y_array'
    coord_keys = (x_coord,y_coord,xy_coord)
    from banksy.initialize_banksy import initialize_banksy
    from banksy.embed_banksy import generate_banksy_matrix
    banksy_dict = initialize_banksy(adata,
                                    coord_keys,
                                    k_geom,
                                    nbr_weight_decay=nbr_weight_decay,
                                    max_m=m,
                                    plt_edge_hist= False,
                                    plt_nbr_weights= False,
                                    plt_agf_angles=False,
                                    plt_theta=False,
                                    )
    banksy_dict, banksy_matrix = generate_banksy_matrix(adata,
                                                        banksy_dict,
                                                        lambda_list,
                                                        max_m=m,
                                                        verbose=False)
    
    banksy_dict[nbr_weight_decay][param_lambda]['adata'].obsm = adata.obsm.copy()
    return(banksy_dict)

def my_banksy_matrix(adata,k_geom=18,nbr_weight_decay='scaled_gaussian',m=0,param_lambda=0.2):
    banksy_dict = original_banksy_matrix(adata,k_geom=k_geom,nbr_weight_decay=nbr_weight_decay,m=m,param_lambda=param_lambda)
    adata = banksy_dict[nbr_weight_decay][param_lambda]['adata']
    return(adata)

def compute_BANKSY_embedding(adata,k_geom=18,nbr_weight_decay='scaled_gaussian',m=0,param_lambda=0.2,n_comps=50):
    from kontext.utils import pca_on_adata
    adata_ = my_banksy_matrix(adata,k_geom=k_geom,nbr_weight_decay=nbr_weight_decay,m=m,param_lambda=param_lambda)
    adata_ = pca_on_adata(adata_,n_comps=n_comps)
    adata.obsm['E_pca'] = adata_.obsm['X_pca']
    return(adata)



def benchmark_path(data):
    from kontext.utils import format_value
    signature_dataset = "_".join(f"{format_value(v)}" for k, v in data.items() if v is not None)
    path = Path(f"/pasteur/helix/projects/ml4ig_hot/Users/aozierla/kontext_project/results/benchmark_2026_07/{signature_dataset}/")
    path.mkdir(parents=True, exist_ok=True)
    return(path)

def compute_Kontext_embedding(adata,sigma_e=1, snn=20, alpha=1):
    from kontext.kontext_class import Kontext
    context = Kontext(sigma_e=sigma_e, snn=snn, alpha=alpha,verbose=True)
    context.fit(adata,add_layer=True)
    return(adata)

def embedding_benchmark(adata,matrix,param):

    adata = {'SEDR':compute_SEDR_embedding,
           'BANKSY':compute_BANKSY_embedding,
           'GraphST':compute_GraphST_embedding,
           'Kontext':compute_Kontext_embedding
                           }[matrix](adata,**param)
    return adata


def clustering_benchmark(adata,clustering,seed,matrix_key='E_pca'):
    fn = {
    "gmm": gmm_clustering,
    "kmeans": kmeans_clustering,
    "SEDR_mclust": mclust_clustering,
    "BANKSY_leiden": leiden_clustering,
    }[clustering]

    params0 = {'clustering':clustering,"seed":seed, "n_domains":adata.uns['data']['n_domains']} 
    params =  {k: v for k, v in params0.items() if k != "clustering"}
    matrix = adata.obsm[matrix_key]
    labels = fn(matrix=adata.obsm[matrix_key], **params)
    adata.obs["spatial_domains"] = pd.Categorical(labels)


def scores_benchmark(adata,ground_truth='annotation',spatial_domains='spatial_domains',spatial='spatial'):

        out = {}
        print('ARI',end=' ')
        out['ARI'] = adjusted_rand_score(adata.obs[ground_truth],adata.obs[spatial_domains])
        print(out['ARI'])
        print('NMI',end=' ')
        out['NMI'] = normalized_mutual_info_score(adata.obs[ground_truth],adata.obs[spatial_domains])
        print(out['NMI'])
        print('HOM',end=' ')
        out['HOM'] = homogeneity_score(adata.obs[ground_truth],adata.obs[spatial_domains])
        print(out['HOM'])
        print('COM',end=' ')
        out['COM'] = completeness_score(adata.obs[ground_truth],adata.obs[spatial_domains])
        print(out['COM'])
    
        return(out)


def run_benchmark(data,matrix,embedding,spatial_domains,preprocessing=None,additional_info=None,save_clustering=False):
    from utils.load_data import load_and_preprocess_adata
    from kontext.utils import format_time,format_size,Profiler,format_value,format_key
    if additional_info is None:     
        additional_info = {} 

    print('data',data)
    print('preprocessing',preprocessing)
    print('matrix',matrix)
    print('embedding',embedding)
    print('spatial_domains',spatial_domains)
    print('additional_info',additional_info)
    print('benchmark_path')
    signature = "_".join(f"{format_key(k)}{format_value(v)}" for k, v in {'matrix':matrix,**preprocessing,**embedding,**spatial_domains}.items() if v is not None)
    path = benchmark_path(data)
    p = path / f'{signature}.json'
    pclust = path / f'{signature}_clustering.csv'
    print(p)
    print(signature)

    signature_dataset = "_".join(f"{format_value(v)}" for k, v in data.items() if v is not None)
    signature_preprocessing = "_".join(f"{format_key(k)}{format_value(v)}" for k, v in {'matrix':matrix,**preprocessing}.items() if v is not None)
    signature_embedding = "_".join(f"{format_key(k)}{format_value(v)}" for k, v in {'matrix':matrix,**embedding}.items() if v is not None)
    signature_spatial_domains = "_".join(f"{format_key(k)}{format_value(v)}" for k, v in {'matrix':matrix,**spatial_domains}.items() if v is not None)
    
    
    
    print('load_and_preprocess_adata')
    print(data,matrix)
    start = time()
    tracemalloc.start()
    adata = load_and_preprocess_adata(data,preprocessing_params=preprocessing,ref_method=matrix)
    loading_time = time() - start 
    mem = tracemalloc.get_traced_memory()
    a3,b3 = mem[0],mem[1]
    tracemalloc.stop()
    print(f'{format_time(loading_time)}  {format_size(b3)}')    
    print(len(adata))
    print(adata)


    print('embedding')
    start = time()
    tracemalloc.start()



    gpu = matrix in ('GraphST', 'SEDR')          # these run on GPU
    with Profiler(gpu=gpu) as pr_emb:
        adata = embedding_benchmark(adata, matrix=matrix, param=embedding)
    embedding_time   = pr_emb.time
    emb_peak_rss     = pr_emb.peak_rss
    emb_delta_rss    = pr_emb.delta_rss
    emb_gpu_peak     = pr_emb.gpu_peak
    
    new_time = {
        'embedding_time_n': embedding_time,
        'embedding_peak_rss': emb_peak_rss,
        'embedding_delta_rss': emb_delta_rss,
        'embedding_gpu_peak': emb_gpu_peak,
        'f_embedding_time_n': format_time(embedding_time),
        'f_embedding_peak_rss': format_size(emb_peak_rss),
        'f_embedding_gpu_peak': format_size(emb_gpu_peak),
    }
    print(new_time)

    embedding_time = time() - start 
    mem = tracemalloc.get_traced_memory()
    a1,b1 = mem[0],mem[1]
    tracemalloc.stop()
    print(f'{format_time(embedding_time)}  {format_size(b1)}')


    print('spatial_domains')
    start_2 = time()
    tracemalloc.start()
    clustering_benchmark(adata,**spatial_domains)
    clustering_time = time() - start_2
    mem = tracemalloc.get_traced_memory()
    a2,b2 = mem[0],mem[1]
    tracemalloc.stop()
    print(f'{format_time(clustering_time)}  {format_size(b2)}')
    
    print('scores')
    start_2 = time()
    tracemalloc.start()
    scores = scores_benchmark(adata)
    score_time = time() - start_2
    mem = tracemalloc.get_traced_memory()
    a4,b4 = mem[0],mem[1]
    tracemalloc.stop()
    print(f'{format_time(score_time)}  {format_size(b4)}')
    

    print('output')
    out = {'signature':signature,
            'f_embedding_time':format_time(embedding_time),
            'f_clustering_time':format_time(clustering_time),
            'f_loading_time':format_time(loading_time),
            'f_embedding_memory_a':format_size(a1),
            'f_embedding_memory_b':format_size(b1),
            'f_clustering_memory_a':format_size(a2),
            'f_clustering_memory_b':format_size(b2),
            'f_loading_memory_a':format_size(a3),
            'f_loading_memory_b':format_size(b3),
            'loading_time':loading_time,
            'embedding_time':embedding_time,
            'clustering_time':clustering_time,
            'embedding_memory_a':a1,
            'embedding_memory_b':b1,
            'clustering_memory_a':a2,
            'clustering_memory_b':b2,
            'loading_memory_a':a3,
            'loading_memory_b':b3,
            **adata.uns['data'],
            'embedding_params':embedding,
            'spatial_domains_params':spatial_domains,
            'matrix':matrix,
            **embedding,
            **spatial_domains,
            **scores,
            **preprocessing,
            **additional_info,
            'signature_dataset':signature_dataset,
            'signature_preprocessing':signature_preprocessing,
            'signature_embedding':signature_embedding,
            'signature_spatial_domains':signature_spatial_domains,
            **new_time
            }
    
    print('save')
    print(scores)
    print(out)
    with open(p, "w") as f:
        json.dump(out, f, indent=2)

    if spatial_domains['seed']==0:
        adata.obs["spatial_domains"].to_csv(pclust)


def run_benchmark_wrapper(param):
     run_benchmark(**param)


