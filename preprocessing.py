from itertools import count
import pandas as pd

def analyse_data(anno, exp):
    merged_df = pd.merge(anno, exp.rename(columns={'Unnamed: 0':'sample'}), on='sample',  how='left')
    
    to_drop = ['sample', 
        'secondary_sample_accession', 
        'sel_Beghini_2021', 
        'study_accession',
        'sample_accession',
        'instrument_platform',
        'instrument_model',
        'library_layout',
        'sample_alias',
        'country',
        'individual_id',
        'timepoint',
        'body_site',
        'host_phenotype',
        'host_subphenotype',
        'class',
        'to_exclude',
        'low_read',
        'low_map',
        'excluded',
        'excluded_comment',
        'mgp_sample_alias',
        'westernised',
        'body_subsite']

    df = merged_df.drop(columns=to_drop)

    return df