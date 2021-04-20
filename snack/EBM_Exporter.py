
import pandas as pd
import numpy as np
import joblib
import math
import random
from sklearn import metrics

from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
from interpret import show

### Helper functions ###

def simplify_categorical_bucket_coefficients(bucket_tbl, tol=1e-12):
    rows = []
    for feature_name in set(bucket_tbl['feature']):
        feature_buckets = bucket_tbl[bucket_tbl['feature'] == feature_name].reset_index(drop=True)
        current_edge = feature_buckets['lo'][0]
        for i in range(feature_buckets.shape[0] - 1):
            if abs(feature_buckets['term'][i+1] - feature_buckets['term'][i]) > tol:
                new_edge = feature_buckets['lo'][i+1]
                rows.append({'feature':feature_name, 'lo':current_edge, 'hi':new_edge, 'term':feature_buckets['term'][i]})
                current_edge = new_edge
        last_row = feature_buckets.shape[0] - 1
        rows.append({'feature':feature_name, 
                     'lo':current_edge, 
                     'hi':feature_buckets['hi'][last_row], 
                     'term':feature_buckets['term'][last_row]})
    return(pd.DataFrame(rows))


def squash(x_vec): 
    import math
    return [1/(1 + math.exp(-x)) for x in x_vec]


def bool_cols_to_char(pdf):
    pdf2 = pdf.copy()
    bool_cols = pdf2.columns[pdf2.dtypes == 'bool']
    for bc in bool_cols:
        pdf2[bc] = pdf2[bc].astype(str)
    return pdf2


def pandas_to_db(pdf, table_name, db):
    """
    Save Pandas DataFrame as table in database.
    The `pandas.to_sql()` method converts Boolean values to int64; 
    if you read them back with `pandas.read_sql_query()` they come back as 1 or 0.
    """
    bool_cols_to_char(pdf).to_sql(table_name, db, if_exists='replace')


def probability_comparison_plot(scores_table, x_prob_col, y_prob_col, label_col='success'):
    import matplotlib.pyplot as plt
    colors = {True:'red', False:'blue'}

    plt.scatter(scores_table[x_prob_col], 
                scores_table[y_prob_col], 
                c=scores_table[label_col].map(colors))

    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = 0 + 1 * x_vals
    plt.plot(x_vals, y_vals, 'k', '--')


def pdf2cte(terms_table, table_name):
    """Generate SQL code to represent Pandas DataFrame as a Common Table Expression"""
    quotify = lambda c: c  # '[' + c + ']'
    row2tuple_str = lambda row: '(' + ', '.join([ f"'{x}'" if (isinstance(x, str) or isinstance(x, bool)) else str(x) 
                                                 for x in row ]) + ')'
    cte_strings = []
    colnames_str = ', '.join([quotify(c) for c in terms_table.columns])
    tuple_strings = terms_table.apply(row2tuple_str, axis=1)
    tuples = ',\n\t\t'.join([ts for ts in tuple_strings])
    cte_strings.append( f"{table_name} ({colnames_str}) as\n\t(VALUES\n\t\t{tuples}\n\t)" )
    return ',\n'.join(cte_strings)


class EBM_Exporter:
    
    ### Methods ###
    
    def __init__(self, ebm):
        self.ebm = ebm
    
    
    def greeting(self):
        return "Hello"
    
    
    def get_feature_types(self):
        return pd.DataFrame({'feature':self.ebm.feature_names, 'type': self.ebm.feature_types})
    
    
    def get_intercept_table_sql(self):
        intercept = self.ebm.intercept_[0]
        return f"intercept as (\n\tselect {intercept:f} term\n)"


    def get_continuous_terms_as_pdf(self):
      """ Extract continuous terms from EBM as a Pandas DataFrame. """
      terms_pdf_list = []

      for feature_idx in range(len(self.ebm.feature_names)):
        feature_type = self.ebm.feature_types[feature_idx]    # categorical, continuous, pairwise
        feature = self.ebm.feature_names[feature_idx]

        if feature_type == 'continuous':
          bin_edges = self.ebm.preprocessor_.col_bin_edges_[feature_idx].copy()
          bin_edges[0] = float('-1e6')  # '-inf'
          bin_edges[-1] = float('1e6')  # 'inf'
          terms = self.ebm.additive_terms_[feature_idx]
          terms_pdf_list.append(pd.DataFrame({'feature':feature, 
                                              'lo':bin_edges[0:(len(bin_edges) - 1)], 
                                              'hi':bin_edges[1:], 
                                              'term':terms}))

      continuous_terms_pdf = pd.concat(terms_pdf_list).reset_index(drop=True)
      return simplify_categorical_bucket_coefficients(continuous_terms_pdf)


    def get_categorical_terms_as_pdf(self):
      """ Extract categorical terms from EBM as a Pandas DataFrame. """
      terms_pdf_list = []

      for feature_idx in range(len(self.ebm.feature_names)):
        feature_type = self.ebm.feature_types[feature_idx]    # categorical, continuous, pairwise
        feature = self.ebm.feature_names[feature_idx]

        if feature_type == 'categorical':
          col_map = self.ebm.preprocessor_.col_mapping_[feature_idx]
          terms_pdf_list.append(pd.DataFrame({'feature':feature, 
                              'value':[k for k in col_map.keys() if k==k], 
                              'term':self.ebm.additive_terms_[feature_idx]}))

      return pd.concat(terms_pdf_list).reset_index(drop=True)


    def get_input_table_long_sql(self, input_table_name, id_column_name):
        feature_type = {n:t for n, t in zip(self.ebm.feature_names, self.ebm.feature_types)}
        qvec = [f"\tSELECT {id_column_name}, '{feature_name}' AS feature, " +
                f"'{feature_type[feature_name]}' as type, {feature_name} AS value " +
                f"FROM {input_table_name}"
                for feature_name in self.ebm.feature_names if feature_type[feature_name] != 'pairwise']
        return "input_data_long1 AS (" + \
                "\n\tUNION ALL\n".join(qvec) + \
                "\n)\n,\n" + \
                f"input_data_long as (\n\tselect * from input_data_long1 order by {id_column_name}, feature\n" + \
                ")"


    def get_feature_pairs(self):
        rows = []
        for pairwise_feature in np.array(self.ebm.feature_names)[np.array(self.ebm.feature_types) == 'pairwise']:
            f1, f2 = pairwise_feature.split(' x ')
            rows.append({'feature1': f1, 'feature2':f2})
        return pd.DataFrame(rows)


    def get_categorical_pairwise_index_table(self):
        rows = []

        categorical_pairwise_dict = {self.ebm.pair_preprocessor_.feature_names[k]:v 
                                         for k,v in self.ebm.pair_preprocessor_.col_mapping_.items()
                                             if k==k}

        for col in categorical_pairwise_dict.keys():
            col_mapping = categorical_pairwise_dict[col]
            for cat in col_mapping.keys():
                if cat==cat:
                    cat_idx = col_mapping[cat]
                    rows.append({'feature':col, 'value':cat, 'idx': cat_idx})

        return pd.DataFrame(rows)


    def get_continuous_pairwise_index_table(self):
        col_edge_df_list = []

        continuous_pairwise_dict = {self.ebm.pair_preprocessor_.feature_names[k]:v 
                                         for k,v in self.ebm.pair_preprocessor_.col_bin_edges_.items()
                                             if k==k}

        for col in continuous_pairwise_dict.keys():
            bin_edges = [x for x in continuous_pairwise_dict[col]]
            bin_edges[0] = -1e6
            bin_edges[-1] = 1e6
            col_edge_df = pd.DataFrame({'feature':col, 
                         'lo': bin_edges[0:len(bin_edges) - 1], 
                         'hi': bin_edges[1:], 
                         'idx':[x for x in range(len(bin_edges) - 1)]})
            col_edge_df_list.append(col_edge_df)

        return pd.concat(col_edge_df_list).reset_index(drop=True)


    def get_pairwise_terms(self):
        feature_type = {f:t for f, t in zip(self.ebm.feature_names, self.ebm.feature_types)}
        pairwise_terms = {k:v for k, v in zip(self.ebm.feature_names, self.ebm.additive_terms_) if feature_type[k] == 'pairwise'}
        pairwise_term_tables = []
        for feature_name in self.ebm.feature_names:
            if feature_type[feature_name] == 'pairwise':
                rows = []
                feature1, feature2 = feature_name.split(' x ')
                ttbl = pairwise_terms[feature_name]
                for i in range(ttbl.shape[0]):
                    for j in range(ttbl.shape[1]):
                        rows.append({'feature1':feature1, 'feature1_idx':i, 'feature2':feature2, 'feature2_idx':j, 'term':ttbl[i,j]})
                pairwise_term_tables.append(pd.DataFrame(rows))
        return pd.concat(pairwise_term_tables).reset_index(drop=True)


    def export_as_sql(self, input_table_name='input_table', id_column_name='id', data_table_format='temporary_view'):
        """
          @param data_table_format: 'cte' or 'temporary_view'
        """
        
        cte_sql_list = [
            self.get_intercept_table_sql(),
            self.get_input_table_long_sql(input_table_name, id_column_name)
        ]
        cte_definitions = '\n,\n'.join(cte_sql_list)

        data_tables = {
            'categorical_terms':self.get_categorical_terms_as_pdf(),
            'continuous_terms': self.get_continuous_terms_as_pdf(),
        }

        feature_pairs = self.get_feature_pairs()
        if len(feature_pairs) > 0:
            data_tables['feature_pairs'] = feature_pairs
            data_tables['pairwise_terms'] = self.get_pairwise_terms()
            data_tables['categorical_pairwise_index_table'] = self.get_categorical_pairwise_index_table()
            data_tables['continuous_pairwise_index_table'] = self.get_continuous_pairwise_index_table()

        data_table_sql_list = []
        for table_name, table_def in data_tables.items():
            data_table_sql_list.append("CREATE OR REPLACE TEMPORARY VIEW " + pdf2cte(table_def, table_name) + ";")

        data_table_definitions = "\n\n".join(data_table_sql_list)

        first_order_query = (
            f"WITH \n"
            f"{cte_definitions} \n"
            f", \n"
            f"first_order_terms as ( \n"
            f"    select idl.id, idl.feature, cat.term \n"
            f"      from input_data_long idl \n"
            f"        join categorical_terms cat \n"
            f"            on idl.feature == cat.feature \n"
            f"            and idl.value = cat.value \n"
            f"        where idl.type == 'categorical' \n"
            f"    union all \n"
            f"    select idl.id, idl.feature, num.term \n"
            f"        from input_data_long idl \n"
            f"            join continuous_terms num \n"
            f"                on idl.feature == num.feature \n"
            f"                and cast(idl.value as float) > num.lo \n"
            f"                and cast(idl.value as float) <= num.hi \n"
            f"            where idl.type == 'continuous' \n"
            f"         \n"
            f") \n"
            f", \n"
            f"id_term_sum as ( \n"
            f"    select id, sum(term) term_sum \n"
            f"        from first_order_terms group by id \n"
            f") \n"
            f", \n"
            f"get_first_order_score as ( \n"
            f"  select its.id, (its.term_sum + intercept.term) score  \n"
            f"      from id_term_sum its CROSS JOIN intercept \n"
            f") \n"
        )
        pairwise_extensions = """
            ,
            idlp as (
                select idl_cat.*, cat_pit.idx pairwise_idx
                    from (select * from input_data_long where type='categorical') as idl_cat
                        join categorical_pairwise_index_table as cat_pit
                            on idl_cat.feature = cat_pit.feature
                            and idl_cat.value = cat_pit.value
                UNION ALL
                select idl_num.*, num_pit.idx pairwise_idx
                    from (select * from input_data_long where type='continuous') as idl_num
                        join continuous_pairwise_index_table num_pit
                            on idl_num.feature = num_pit.feature
                            and idl_num.value > num_pit.lo
                            and idl_num.value <= num_pit.hi
            )
            ,
            interaction_pairs as (
                select idlp1.id, fp.feature1, idlp1.pairwise_idx feature1_idx, fp.feature2, idlp2.pairwise_idx feature2_idx
                    from idlp idlp1,
                         idlp idlp2,
                         feature_pairs fp
                    where idlp1.id = idlp2.id
                    and idlp1.feature = fp.feature1
                    and idlp2.feature = fp.feature2
            ),
            interaction_pair_terms as (
                select ip.*, pt.term
                    from interaction_pairs ip
                    left join pairwise_terms pt
                    on ip.feature1 = pt.feature1
                    and ip.feature1_idx = pt.feature1_idx
                    and ip.feature2 = pt.feature2
                    and ip.feature2_idx = pt.feature2_idx
            ),
            pairwise_terms_score as (
                select id, sum(term) pairwise_term_sum from interaction_pair_terms group by id order by id
            ),
            get_total_score as (
                select fos.id, (score + pairwise_term_sum) total_score
                    from get_first_order_score fos
                    join pairwise_terms_score pts
                        on fos.id = pts.id
            )
        """.replace('\n        ', '\n')

        if len(feature_pairs) == 0:
            scoring_query = first_order_query + "\nselect * from get_first_order_score"
        else:
            scoring_query = first_order_query + pairwise_extensions + "\nselect * from get_total_score"

        return data_table_definitions, scoring_query

