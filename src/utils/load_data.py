import json
from anndata import AnnData 
from scipy.io import mmread
from scipy.sparse import csr_matrix
import pandas as pd
import numpy as np
import scanpy as sc 
import matplotlib.pyplot as plt


from pathlib import Path
import json

from kontext.data import preprocess_adata,get_pp_str
from kontext.utils import format_key, format_value


# _BENCHMARK_DIR = Path(__file__).parent
# _DATASET_INFO_PATH = _BENCHMARK_DIR / "datasets_info.json"


def get_dataset_info():
    return {
        "MOSTA":{
            "E9.5_E1S1": {"n_domains": 12, "n_obs": 5913, "n_var": 25568},
            "E10.5_E1S1": {"n_domains": 13, "n_obs": 18408, "n_var": 25201}, 
            "E11.5_E1S1": {"n_domains": 19, "n_obs": 30124, "n_var": 26854}, 
            "E12.5_E1S1": {"n_domains": 23, "n_obs": 51365, "n_var": 27810}, 
            "E13.5_E1S1": {"n_domains": 19, "n_obs": 77369, "n_var": 28408}, 
            "E13.5_E1S2": {"n_domains": 19, "n_obs": 64584, "n_var": 28208}, 
            "E14.5_E1S1": {"n_domains": 26, "n_obs": 102519, "n_var": 28463}, 
            "E14.5_E1S2": {"n_domains": 21, "n_obs": 102977, "n_var": 28766}, 
            "E15.5_E1S1": {"n_domains": 26, "n_obs": 113350, "n_var": 28798}, 
            "E15.5_E1S2": {"n_domains": 22, "n_obs": 109811, "n_var": 28154}, 
            "E16.5_E1S1": {"n_domains": 25, "n_obs": 121767, "n_var": 28204}},
        "ZESTA":{
           "zf3_1":{"n_domains":3,"n_obs":1167,"n_var":6261},
            "zf5_10":{"n_domains":7,"n_obs":563,"n_var":1962},
            "zf10_14":{"n_domains":8,"n_obs":1231,"n_var":2756},    
            "zf12_8":{"n_domains":11,"n_obs":2081,"n_var":4963},     
           "zf12_9":{"n_domains":11,"n_obs":2588,"n_var":5802},     
           "zf18_7":{"n_domains":14,"n_obs":3475,"n_var":7522},     
           "zf18_8":{"n_domains":14,"n_obs":3048,"n_var":9217},
           "zf18_9":{"n_domains":14,"n_obs":5935,"n_var":12485},     
           "zf24_4":{"n_domains":19,"n_obs":5271,"n_var":12680},      
           "zf24_7":{"n_domains":18,"n_obs":6318,"n_var":13786},      
           "zf24_9":{"n_domains":17,"n_obs":4708,"n_var":13042},
           "zf24_10":{"n_domains":18,"n_obs":5968,"n_var":13527}},
          "ARTISTA":{
            "2DPI_1": {"n_domains": 18, "n_obs": 7668, "n_var": 27324},      
            "2DPI_2": {"n_domains": 17, "n_obs": 8192, "n_var": 28365},
            "2DPI_3": {"n_domains": 18, "n_obs": 8738, "n_var": 28371},
            "5DPI_1": {"n_domains": 19, "n_obs": 8106, "n_var": 28741},
            "5DPI_2": {"n_domains": 17, "n_obs": 8178, "n_var": 29442},     
            "5DPI_3": {"n_domains": 17, "n_obs": 8372, "n_var": 29341},
            "10DPI_1": {"n_domains": 18, "n_obs": 9440, "n_var": 27600},
            "10DPI_2": {"n_domains": 18, "n_obs": 9600, "n_var": 28602},
            "10DPI_3": {"n_domains": 18, "n_obs": 10002, "n_var": 27398},    
            "15DPI_1": {"n_domains": 18, "n_obs": 8996, "n_var": 23331},    
            "15DPI_2": {"n_domains": 20, "n_obs": 9639, "n_var": 23885},
            "15DPI_3": {"n_domains": 21, "n_obs": 9676, "n_var": 22297},
            "15DPI_4": {"n_domains": 18, "n_obs": 10792, "n_var": 23271},
            "20DPI_1": {"n_domains": 17, "n_obs": 10462, "n_var": 28205},
            "20DPI_2": {"n_domains": 20, "n_obs": 11319, "n_var": 28987},    
            "20DPI_3": {"n_domains": 18, "n_obs": 11048, "n_var": 28069},
            "30DPI": {"n_domains": 17, "n_obs": 9252, "n_var": 19909},
            "60DPI": {"n_domains": 16, "n_obs": 10964, "n_var": 20038} },
        "MBA":{
           "C57BL6J_1_082": {"n_domains":{"ccf_region_name": 170, "major_brain_region": 10, "tissue": 10, "cell_type": 22 },"n_obs": 42117, "n_var": 1120},
            "C57BL6J_1_053": {"n_domains":{"ccf_region_name": 77, "major_brain_region": 9, "tissue": 9, "cell_type": 19 },"n_obs": 34126, "n_var": 1120},
            "C57BL6J_1_117": {"n_domains":{"ccf_region_name": 59, "major_brain_region": 6, "tissue": 6, "cell_type": 21 },"n_obs": 22862, "n_var": 1120},
            "C57BL6J_1_111": {"n_domains":{"ccf_region_name": 89, "major_brain_region": 8, "tissue": 8, "cell_type": 20 },"n_obs": 35171, "n_var": 1120},
            "C57BL6J_1_113": {"n_domains":{"ccf_region_name": 84, "major_brain_region": 8, "tissue": 8, "cell_type": 21 },"n_obs": 30864, "n_var": 1120},
               },
     }


    # with open(_DATASET_INFO_PATH, "r") as f:
    #     return json.load(f)



def get_n_domains(dataset,sample,annotation=None):
    n_domains = get_dataset_info()[dataset][sample]['n_domains']
    if dataset == 'MBA':
        n_domains = n_domains[annotation]
    return(n_domains)

def load_adata(dataset,sample,keep_counts=False,annotation='tissue'):

    if dataset == 'MOSTA':
        adata = load_MOSTA(sample)
    elif dataset == 'ZESTA':
        adata = load_ZESTA(sample)
    elif dataset == 'ARTISTA':
        adata = load_ARTISTA(sample)
    elif dataset == 'MBA':
        adata = load_MBA(sample,annotation=annotation)
    elif dataset == 'CARDIO':
        adata = load_CARDIO(sample)
    elif dataset == 'LYMPH':
        adata = load_LYMPH(sample)
    else:
        print(f'{dataset} unknown')
    if keep_counts:
        adata.layers['counts'] = adata.X
    if adata.is_view:
        adata = adata.copy()
    adata.uns['data'] = {'dataset':dataset,'sample':sample,'annotation':annotation,'n_domains':get_n_domains(dataset,sample,annotation)}
    return(adata)










def load_ARTISTA(sample='2DPI_1'):
    # path = "/pasteur/zeus/projets/p02/ml4ig_hot/Users/ghuizing/data/artista/"
    path = "/pasteur/appa/scratch/aozierla/data/artista/"
    adata = sc.read_h5ad(f'{path}{sample}.h5ad')

    adata.X = adata.layers['counts'] 
    adata.layers.pop('counts')

    adata.obs['sample'] = sample
    adata.obs['annotation'] = adata.obs['Annotation'].astype('category')
    sc.pl._utils._set_colors_for_categorical_obs(adata, 'annotation',sc.pl.palettes.default_28)
    adata.obsm["spatial"] = adata.obsm["spatial"] * np.array([-1, -1])
    adata.uns['data'] = {'dataset':'ARTISTA','sample':sample}
    if sample == '10DPI_1':
        adata = adata[np.arange(adata.n_obs) != np.argmin(adata.obsm["spatial"][:, 0])]
        adata = adata[np.arange(adata.n_obs) != np.argmin(adata.obsm["spatial"][:, 0])]
        adata = adata[np.arange(adata.n_obs) != np.argmin(adata.obsm["spatial"][:, 0])]
        adata = adata[np.arange(adata.n_obs) != np.argmin(adata.obsm["spatial"][:, 0])]

    if sample == '20DPI_2':
        adata = adata[np.arange(adata.n_obs) != np.argmin(adata.obsm["spatial"][:, 0])]
        adata = adata[np.arange(adata.n_obs) != np.argmin(adata.obsm["spatial"][:, 0])]       
        adata = adata[np.arange(adata.n_obs) != np.argmax(adata.obsm["spatial"][:, 1])]    
    return(adata)



def load_ZESTA(sample):
    # path = "/pasteur/helix/projects//ml4ig_hot/Users/ghuizing/data/zesta/"
    path = "/pasteur/appa/scratch/aozierla/data/zesta/"
    sampl,slic = sample.split('_')
    adata = sc.read_h5ad(f'{path}{sampl}_stereoseq.h5ad')
    adata.X = adata.layers['counts']
    adata.layers.pop('counts')

    adata.obsm['spatial'] = adata.obs[['spatial_x','spatial_y']].to_numpy()
    adata.obs['annotation'] = adata.obs['bin_annotation'].astype('category')
    adata.uns['data'] = {'dataset':'ZESTA','sample':sample}
    adata = adata[adata.obs['slice']==int(slic)]

    return(adata)


def load_MOSTA(sample='E9.5_E1S1'):
    path = "/pasteur/helix/projects//ml4ig_hot/Users/ghuizing/data/mosta/"
    adata = sc.read_h5ad(f'{path}{sample}.MOSTA.h5ad')

    adata.X = adata.layers['count']
    adata.layers.pop('count')
    adata.obsm["spatial"] = adata.obsm["spatial"] * np.array([1, -1])
    adata.obs['layer_guess_reordered_short'] = adata.obs['annotation']
    adata.uns['data'] = {'dataset':'MOSTA','sample':sample}
    adata.obs['sample'] = [sample]*len(adata.obs)
    
    return(adata)


def load_MBA(sample,annotation='tissue'): # MERFISH BRAIN ATLAS
    # Possible annotations : ['ccf_region_name','major_brain_region','cell_type']
    adata = sc.read_h5ad(f'/pasteur/helix/projects/ml4ig_hot/Users/aozierla/data/mouse_brain_atlas/WB_MERFISH_animal1_coronal_{sample}.h5ad')
    adata.obs['annotation'] = adata.obs[annotation]
    adata.uns['data'] = {'dataset':'MBA','sample':sample,'annotation':annotation}
    adata = adata[~adata.obs['annotation'].isna()]
    return(adata)


def create_mba_files(adata):
    adata.obs['sample'] = adata.obs['brain_section_label'].apply(lambda x: x.replace('-','_').replace('.','_')) 

    for s in adata.obs['sample'].unique():
        a_ = adata[adata.obs['sample']==s]
        
        a_.obsm['spatial'] = a_.obsm['X_spatial_coords'] 
        a_.obsm.pop('X_spatial_coords')
        a_.obs['annotation'] = a_.obs['tissue'].astype('category')
        a_.uns['dataset'] = 'MMBA1C'
        a_.uns['sample'] = s

        print(s,len(a_))

        fig,axes = get_figs(3,3,3)
        n_obs,n_var = a_.shape
        nccf = a_.obs['ccf_region_name'].nunique()
        nmbr = a_.obs['major_brain_region'].nunique()
        nct = a_.obs['cell_type'].nunique()
        nostr = f"nobs:{n_obs} "
        nvstr = f"ng:{n_var} "
        ncstr = f"ccf:{nccf} "
        nmstr = f"mbr:{nmbr} "
        nctstr = f"ct:{nct} "
        fig.suptitle(f'{s} {nostr} {nvstr} {ncstr} {nmstr} {nctstr}',y=1.02)
        ax=axes[0]
        sc.pl.embedding(a_,'spatial',color='ccf_region_name',ax=ax,show=False)
        ax.legend([])
        ax=axes[1]
        sc.pl.embedding(a_,'spatial',color='major_brain_region',ax=ax,show=False)
        ax=axes[2]
        sc.pl.embedding(a_,'spatial',color='cell_type',ax=ax,show=False)
        plt.show()
        a_.write_h5ad(f'/pasteur/helix/projects/ml4ig_hot/Users/aozierla/data/mouse_brain_atlas/WB_MERFISH_animal1_coronal_{s}.h5ad')
        

def load_LYMPH(sample):
    path = '/pasteur/helix/projects/ml4ig_hot/Users/aozierla/data/metastatic_lymph_node/'
    
    datasets = {'primary':'GSM8193670_primary_HNSCC_S2_',
            'metastatic':'GSM8193671_metastatic_lymph_node_S37_',
            '3d':'GSE251926_metastatic_lymph_node_3d.h5ad'}
    
    full_name = datasets[sample]
    if sample == '3d':
        adata = sc.read_h5ad(f'{path}{full_name}')
        # print('load_LYMPH',adata.obsm.keys)
        adata.obsm['spatial'] = adata.obsm['spatial_3d_aligned']


        annot_to_new ={
        'unknown':'other',
        'Tumor':'Tumor cell',
        'Plasma_IgG':'Plasma IgG',
        'CAF':'CAF',
        'Fibroblasts':'Fibroblast',
        'T_cell':'T cells',
        'CAM':'CAM',
        'Macrophages':'Macrophage',
        'High_Endothelial_Venules':'other',
        'M1_macrophages':'Macrophage',
        'Plasma_IgA':'Plasma IgA',
        'Germinal_Center_Plasma_IgM_B_cell':'Plasma IgM',
        'Tumor_Keratin_Pearl':'Keratin Pearl',
        'Dendritic_cells':'other',
        'Cytotoxic_IFN_signaling':'other',
        'Mast_cells':'other',
        'Cortex_CCL21':'other'}
        adata.obs['annotation'] = adata.obs['annotation'].map(annot_to_new)


        sections = adata.obs['n_section'].sort_values().unique()
        next_ = {sections[i]: sections[i+1] if i<len(sections)-1 else sections[i] for i in range(len(sections)) }
        prev_ = {sections[i]: sections[i-1] if i>0 else sections[i] for i in range(len(sections))}



    else :
        path = '/pasteur/helix/projects/ml4ig_hot/Users/aozierla/data/metastatic_lymph_node/'
        adata = sc.read_10x_h5(f"{path}{full_name}xenium_cell_feature_matrix.h5")
        obs = pd.read_csv(f"{path}{full_name}xenium_cells.csv",index_col=0)
        adata.obs = obs
        adata.obsm['spatial'] = adata.obs[["x_centroid", "y_centroid"]].values
    if adata.is_view:
        adata = adata.copy()
    adata.uns['data'] = {'dataset':'LYMPH','sample':sample}
    return(adata)



def load_CARDIO(sample):
    full_names = {
        'sham_S1':'GSM8595803_sham_day7_',
        'sham_S2':"GSM9460459_sham_day7_slide2_",
        'cryo7D_S1':'GSM8595804_MI_day7_',
        'cryo7D_S2':"GSM9460460_MI_day7_slide2_",
        'cryo7D_S3':"GSM9460462_MI_day7_slide3_",
        'cryo28D_S1':'GSM8595805_MI_day28_',
        'cryo28D_S2':"GSM9460461_MI_day28_slide2_",
        'cryo28D_S3':"GSM9460463_MI_day28_slide3_",}
    full_name = full_names[sample]
    radical = f"/pasteur/helix/projects/ml4ig_hot/Users/aozierla/data/cardioimmune/{full_name}"

    # Obs 
    p = pd.read_csv(f'{radical}cells.csv')
    p['barcode'] = p['cell_id']
    p = p.set_index('barcode')

    # # Cell types 
    if sample == 'cryo7D_S2':
        clusters = pd.read_csv('/pasteur/helix/projects/ml4ig_hot/Users/aozierla/data/cardioimmune/resegment_Day7slide2_Xenium_groups_detail.csv',index_col=0)
        clusters['barcode'] = clusters.index
        clusters = clusters.set_index('barcode')
        cluster_col = clusters.columns[0]
        p['cell_type'] = clusters[cluster_col]
        p['cell_type'] = p['cell_type'].apply(lambda x: x.replace('Fbmyo','FBmyo') if not (isinstance(x,float) and np.isnan(x)) else x)
        p['global_cell_type'] = p['cell_type'].map({v:v.split(sep='_')[0] for v in p['cell_type'].unique() if not (isinstance(v,float) and np.isnan(v))})
        
    else:
        cell_types = pd.read_csv(f'{radical}nico_annotation_ct_name.csv',index_col=0)     
        ct_map = cell_types.iloc[:, 0].to_dict()
        ct_map = {k:v.replace('Fbmyo','FBmyo') for k,v in ct_map.items()}
        global_ct_map = {k:v.split(sep='_')[0] for k,v in ct_map.items()}
        clusters  = pd.read_csv(f'{radical}nico_annotation_cluster.csv',index_col=0)
        cluster_col = clusters.columns[0]                         # first (only) data column
        p['cell_type'] = clusters[cluster_col].map(ct_map)
        p['global_cell_type'] = clusters[cluster_col].map(global_ct_map)

    p['cell_type'] = p['cell_type'].astype('category')
    p['global_cell_type'] = p['global_cell_type'].astype('category')
    
    
    
    # # Count
    m = mmread(f'{radical}matrix.mtx')
    X = csr_matrix(m.T)


    genes = pd.read_csv(f'{radical}features.tsv', sep='\t', header=None, names=['id', 'gene', 'type'])
    adata = AnnData(X=X,obs =p,var = genes.set_index('gene'))
    sc.pl._utils._set_colors_for_categorical_obs(adata, 'cell_type',sc.pl.palettes.default_102)
    sc.pl._utils._set_colors_for_categorical_obs(adata, 'global_cell_type',sc.pl.palettes.default_28)


    adata.obsm['spatial'] = p[["x_centroid", "y_centroid"]].values
    adata.uns['data'] = {'dataset':'CARDIO','sample':sample}    
    adata = adata[~adata.obs['cell_type'].isna()].copy()

    adata.obs['time'] = adata.uns['data']['sample'].split('_')[0]
    adata.obs['time'] =adata.obs['time'].astype('category')

    ct = {
    'CM':'Cardiomyocyte',
    'FB':'Fibroblast',
    'FBmyo':'Myofibroblast',
    'Macro':'Macrophage',
    'mo':'Myeloid_other',
    'DC':'Myeloid_other',
    'Mast':'Myeloid_other',
    'Neutrophil':'Myeloid_other',
    'B':'Lymphoid',
    'ILC2':'Lymphoid',
    'MAIT':'Lymphoid',
    'NK':'Lymphoid',
    'T':'Lymphoid',
    'Treg':'Lymphoid',
    'Fibrocytes':'Fibrocyte',
    'Erythrocyte':'Other',
    'NM':'Other',
    'Schwann':'Other',
    'vEC':'vEC',
    'vEpicardial':'vEpicardial',
    'vPericyte':'vPericyte',
    'vSMC':'vSMC',
    'vSMC1':'vSMC',
    'vSMC2':'vSMC'}

    adata.obs['main_cell_type'] = adata.obs['global_cell_type'].map(ct).astype("category")


    if sample == 'cryo28D_S2':
        # This cell was behavioring weirdly when I computed its context so I discraded it. 
        adata = adata[adata.obs.index != 'hmonjnan-1']
    if sample == 'cryo28D_S3':
        # This cell was behavioring weirdly when I computed its context so I discraded it. 
        adata = adata[adata.obs.index != 'mifmgime-1']
    return(adata)



def load_and_preprocess_adata(data_info,preprocessing_params=None,ref_method=None,keep_counts=False):
    if ref_method is not None:
        keep_counts = keep_counts or ref_method in ['SEDR','STAGATE','BANKSY','GraphST'] or preprocessing_params['preprocessing'] in ['sparse_hvg','large3svg']
    if 'hvg' in str(preprocessing_params['preprocessing']) or preprocessing_params['preprocessing'] in ['image']:
        keep_counts=True

    # if preprocessing_params is None:
    #     preprocessing_params = {}
    #     if ref_method is not None:
    #         preprocessing_params = get_standard_preprocessing_params(ref_method)


    adata = load_adata(keep_counts=keep_counts,**data_info)
    pp_str = get_pp_str(preprocessing_params)
    adata.uns['data']['pp_str'] = pp_str
    adata = preprocess_adata(adata,**preprocessing_params)

    if adata.is_view:
        adata = adata.copy()
    
    return(adata)