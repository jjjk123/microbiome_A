import pandas as pd
import matplotlib.pyplot as plt


class Preprocessing():

    @staticmethod
    def miss_val_tbl(df):
        # Total missing values
        mis_val = df.isnull().sum()

        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)

        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
            columns={0: 'Missing Values', 1: '% of Total Values'})

        # Print some summary information
        tmp = df.shape[1] - mis_val_table_ren_columns['Missing Values'].astype(bool).sum(axis=0)
        """print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
                                                                  "There are " + str(tmp) +
              " columns that have no missing values.")"""

        # Return the dataframe with missing information
        return mis_val_table_ren_columns

    @staticmethod
    def data_proc(df_input):

        # Count the number of times each unique value appears in column CLASS
        c1 = pd.value_counts(df_input['health_status'])

        # Drop 'sample' column since it does not provide any meaningful information
        df_input = df_input.drop(columns=['sample'])

        # Drop 'secondary_sample_accession' column since it does not provide any meaningful information
        df_input = df_input.drop(columns=['secondary_sample_accession'])

        # Drop 'sel_Beghini_2021' column since it does not provide any meaningful information
        df_input = df_input.drop(columns=['sel_Beghini_2021'])

        # Drop 'study_accession' column since it does not provide any meaningful information
        df_input = df_input.drop(columns=['study_accession'])

        # Drop 'sample_accession' column since it does not provide any meaningful information
        df_input = df_input.drop(columns=['sample_accession'])

        # Drop 'instrumemt_platform' column since it does not provide any meaningful information
        df_input = df_input.drop(columns=['instrument_platform'])

        # Drop 'instrumemt_model' column since it does not provide any meaningful information
        df_input = df_input.drop(columns=['instrument_model'])

        # Drop 'library_layout' column since it does not provide any meaningful information
        df_input = df_input.drop(columns=['library_layout'])

        # Drop 'sample_alias' column since it does not provide any meaningful information
        df_input = df_input.drop(columns=['sample_alias'])

        # Drop 'country' column since it does not provide any meaningful information
        df_input = df_input.drop(columns=['country'])

        # Drop 'individual_id' column since it does not provide any meaningful information
        df_input = df_input.drop(columns=['individual_id'])

        # Drop 'timepoint' column since it does not provide any meaningful information
        df_input = df_input.drop(columns=['timepoint'])

        # Drop 'body_site' column since it does not provide any meaningful information
        df_input = df_input.drop(columns=['body_site'])

        # Drop 'host_phenotype' column since it does not provide any meaningful information
        df_input = df_input.drop(columns=['host_phenotype'])

        # Drop 'host_subphenotype' column since it does not provide any meaningful information
        df_input = df_input.drop(columns=['host_subphenotype'])

        # Drop 'class' column since it does not provide any meaningful information
        df_input = df_input.drop(columns=['class'])

        # Drop 'to_exclude' column since it does not provide any meaningful information
        df_input = df_input.drop(columns=['to_exclude'])

        # Drop 'low_read' column since it does not provide any meaningful information
        df_input = df_input.drop(columns=['low_read'])

        # Drop 'low_map' column since it does not provide any meaningful information
        df_input = df_input.drop(columns=['low_map'])

        # Drop 'excluded' column since it does not provide any meaningful information
        df_input = df_input.drop(columns=['excluded'])

        # Drop 'excluded_comment' column since it does not provide any meaningful information
        df_input = df_input.drop(columns=['excluded_comment'])

        # Drop 'mgp_sample_alias' column since it does not provide any meaningful information
        df_input = df_input.drop(columns=['mgp_sample_alias'])

        # Drop 'westernised' column since it does not provide any meaningful information
        df_input = df_input.drop(columns=['westernised'])

        # Drop 'body_subsite' column since it does not provide any meaningful information
        df_input = df_input.drop(columns=['body_subsite'])

        # Drop 'HQ_clean_read_count' column since it does not provide any meaningful information
        df_input = df_input.drop(columns=['HQ_clean_read_count'])

        # Drop 'gut_mapped_read_count' column since it does not provide any meaningful information
        df_input = df_input.drop(columns=['gut_mapped_read_count'])

        # Drop 'oral_mapped_read_count' column since it does not provide any meaningful information
        df_input = df_input.drop(columns=['oral_mapped_read_count'])

        # Get one hot encoding of column 'gender'
        one_hot_gender = pd.get_dummies(df_input['gender'])
        # Drop column 'gender' as it is now encoded
        df_input = df_input.drop('gender', axis=1)
        # Join the encoded dataframe
        df_input = df_input.join(one_hot_gender)

        return (df_input)

    @staticmethod
    def data_checks(df_merged):
        # The elements of CLASS/health_status become as: minority H--> 0,  P--> 1
        df_merged['CLASS'] = (-1) * (pd.factorize(df_merged['health_status'])[0] - 1)

        # Drop 'health_status' column since it does not provide any meaningful information
        df_merged = df_merged.drop(columns=['health_status'])

        # Contains two columns: [1] total missing values, [2] % of missing values
        MissValsTable = Preprocessing.miss_val_tbl(df_merged)

        # Type of each column/feature
        cols_type_df = df_merged.dtypes

        cols = df_merged.dtypes  # Type of each column/feature
        MissValsTable = pd.concat([MissValsTable, cols], axis=1)

        # Contains three columns: [1] total missing values, [2] % of missing values, [3] type of each column/feature
        MissValsTable.columns = ['missing_values', 'perc_miss_vals', 'data_type']

        sorted_MissValsTable = MissValsTable.sort_values('missing_values', ascending=False)
        plt.figure()
        plt.plot(sorted_MissValsTable['missing_values'].to_numpy())
        plt.title('(sorted) missing values per column in MIMIC')

        return sorted_MissValsTable, df_merged

    @staticmethod
    def fill_empty_columns(df):
        filling_list = df.columns[df.isna().any()].tolist()

        for f in filling_list:
            df[f] = df[f].fillna(0)

        return df

    @staticmethod
    def read_data(path_anno: str, path_exp: str, taxonomy_path: str):
        df_anno = pd.read_csv(path_anno)
        df_exp = pd.read_csv(path_exp)

        df_merged = pd.merge(df_anno, df_exp.rename(columns={'Unnamed: 0': 'sample'}), on='sample', how='left')

        # adenoma
        adenoma_genus_proc = 1
        if adenoma_genus_proc == 1:
            # load taxonomy file
            taxo = pd.read_csv(taxonomy_path, sep="\t")
            taxo["msp"] = list(taxo.index)
            taxo = taxo[["msp", "genus"]]

            all_species = list(taxo.index)
            genus_dictionary = {}
            for species in all_species:
                genus = taxo["genus"][species]
                msp = taxo["msp"][species]

                if genus in genus_dictionary:
                    genus_dictionary[genus].append(msp)
                else:
                    genus_dictionary[genus] = [msp]

            all_genus = genus_dictionary.keys()
            for genus in all_genus:
                df_exp[genus] = df_exp[genus_dictionary[genus]].sum(axis=1)
                df_exp = df_exp.drop(columns=genus_dictionary[genus])
            df_merged = pd.merge(df_anno, df_exp.rename(columns={'Unnamed: 0': 'sample'}), on='sample',how='left')
            df_merged = df_merged.drop(list(df_anno[df_anno['host_phenotype'] == "adenoma"].index))

        df_merged = Preprocessing.data_proc(df_merged)

        sorted_MissValsTable, df_merged = Preprocessing.data_checks(df_merged)
        df_merged = Preprocessing.fill_empty_columns(df_merged)

        return df_merged