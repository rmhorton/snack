# snack-0.0.1-py3-none-any.whl

__version__ = '0.0.1'
__doc__ = "Not Feast, just a snack."

import pandas as pd
import datetime
import time
# from sklearn.externals import joblib

import numpy as np
import pandas as pd
from scipy.stats import norm

import networkx as nx
import json
import community as community_louvain

# import pyspark.sql.functions as fn
# from pyspark.sql.types import *
# from pyspark.sql import Window
# from pyspark.sql.dataframe import DataFrame

# we won't need to do this in Spark 3
# if getattr(DataFrame, "transform", None) is None:
#   DataFrame.transform = lambda self,f: f(self)  # 'monkey patching'

def times2(x):
    return x*2


def timestamp_to_timeslice(sdf, timestamp_col, timeslice_id_col, time_unit='hour'):
  """ Convert timestamp column to integer ID representing number of time units since Unix epoch.
  Args:
    sdf (Spark dataframe): input dataframe.
    timestamp_col (str): name of input timestamp column; this will be replaced by a timestamp_id.
    timeslice_id_col (str): name of the column to be generated
    time_unit (str): period of time defined by a fixed number of seconds 
        ('day', 'hour', etc. as defined in `seconds_per` dict in this function)
  
  Returns:
    Spark dataframe with timestamp column replaced by timeslice ID.
  """
  # time_udf = fn.udf(lambda seconds: int(seconds/seconds_per[time_unit]), IntegerType())
  def convert_seconds(num_seconds, time_unit):
    seconds_per = {'week': 60*60*24*7, 'day': 60*60*24, 'hour':60*60, 'minute':60, 'second':1}
    result = None
    try:
      result = int(num_seconds/seconds_per[time_unit])
    except:
      pass
    
    return result
  
  time_udf = fn.udf(lambda sec: convert_seconds(sec, time_unit), IntegerType())
  
  return sdf\
    .withColumn('posix_timestamp', fn.unix_timestamp(fn.col(timestamp_col)))\
    .withColumn(timeslice_id_col, time_udf(fn.col('posix_timestamp')))\
    .drop('posix_timestamp', timestamp_col)


def expand_rows(sdf, from_col, to_col, sequence_col_name, *id_cols):
  """ Expand a range of integers into a set of rows representing all values in the sequence.
  
  Args:
  
    sdf (spark dataframe): input dataframe
    from_col (str): name of column specifying beginning integer value of sequence.
    to_col (str): name of column specifying ending integer value of sequence
    sequence_col_name (str): name of new column to be generated with sequence values
    id_col_names (array of str): names of id columns
  
  Returns: 
  
    spark dataframe with columns specified in `id_col_names` and `sequence_col_name`, with one row per sequence element.
  
  Example::
  
    range_df = spark.createDataFrame(data=[('Ali',3,7), ('Bay',5,10), ('Cal',1,3)], schema = ['name','from','to'])
    expand_rows(range_df, 'from', 'to', 'sequence_id', 'name').show()
    
  """
  
  arrayify_udf = fn.udf(lambda s1, s2: [i for i in range(s1, s2+1)] if s1 is not None and s2 is not None else [], ArrayType(IntegerType()))
  
  id_range_df = sdf.select([ *id_cols, arrayify_udf(fn.col(from_col), fn.col(to_col)).alias('int_range')])

  return id_range_df.select([ *id_cols, fn.explode(fn.col('int_range')).alias(sequence_col_name)])


def fill_missing_values_forward(sdf, ordering_col, cols_to_fill, *id_cols):
  """ Fill missing values by carrying previous values forward.
  
  Args:
    sdf: a Spark DataFrame
    ordering_col: column by which rows should be sorted
    cols_to_fill: list of columns where missing values will be filled
    id_cols: list of columns that collectively form a unique identifier that can be used to partition cases.
  
  """
  lookback_window = Window.partitionBy(*id_cols)\
                 .orderBy(ordering_col)\
                 .rowsBetween( Window.unboundedPreceding, 0)

  for ctf in cols_to_fill:
    filled_col = ctf + "_filled"
    sdf = sdf.withColumn(filled_col, fn.last(sdf[ctf], ignorenulls=True).over(lookback_window))

  return sdf


def add_id_bucket_column(sdf, id_col='pat_id', id_bucket_col='id_bucket', num_buckets=12):
  """ Assign each id value to one of a given set of buckets.
  
  Buckets can be used for train/validate/test splits.
  
  Args:
  
    id_col:
    id_bucket_col: name of new column to generate
    num_buckets: the number of buckets
  
  """
  
# Cluster feature methods

def get_basket_items(sdf, item_col, *key_cols, include_duplicate_items=True, exclude_single_item_baskets=True):
  """ generate sets of items from a table listing items individually (along with the group they belong to)
  
  Args:
    sdf: input Spark dataframe
    item_col: the name of the column indicating the item
    *key_cols: the names of the columns that collectively indicate the group the item should be placed in
    include_duplicate_items:
    exclude_single_item_basekets: if Trye, baskets with only a single item will be filtered out
  
  Notes:
    This function is more or less equivalent to this SQL query:
    
     select patient, encounters.id encounter, encounter_date, 
         collect_list(distinct condition) items, 
         count(distinct condition) num_items
       from encounters 
       group by patient, encounter, encounter_date

  """
  collect_fun = fn.collect_list if include_duplicate_items else fn.collect_set
  basket_items = sdf \
    .groupby(*key_cols) \
    .agg(
      collect_fun(item_col).alias('items')
    )
  
  if exclude_single_item_baskets:
    basket_items = basket_items.filter(fn.size('items') > 1)
  
  return basket_items
  

def fit_embedding(basket_item_sdf, implementation='pyspark', **new_par):
  """ Use create Word2Vec embedding for the items in baskets.
  
  The gensim implementaion is not scalable; it aggregates the input Spark dataframe to a local Pandas dataframe with items aggregated by basket, and returns the embedding results as a Pandas dataframe. The gensim implementation has more (or at least different) options compared to the pyspark implementation (e.g., pyspark only does skipgram method).
  
  Args:
    basket_item_sdf: spark dataframe with basket id (may be multiple columns) and items vector.
    implementation: 'gensim' or 'pyspark'
    new_par: named parameters depend on the implementation (see `par` defaults in code.)
  
  Returns:
    a Pandas dataframe with two columns, 'word' and 'vector'
  """
  if implementation == 'pyspark':
    from pyspark.ml.feature import Word2Vec

    par = {'vectorSize':64, 'windowSize':100, 'minCount':100, 'inputCol':'items', 'outputCol':'embedding'}
    for k in new_par.keys():
      par[k] = new_par[k]

    w2v = Word2Vec(**par)
    w2v_model = w2v.fit(basket_item_sdf)
    wv = w2v_model.getVectors()
    embedding_pdf = wv.toPandas()
  else:
    from gensim.models import word2vec
    import multiprocessing

    par = {'size':64, 'window':50, 'min_count':100}
    for k in new_par.keys():
      par[k] = new_par[k]

    basket_item = basket_item_sdf.toPandas()

    w2v_model = word2vec.Word2Vec([[w for w in v] for v in basket_item['items'].values], 
                                  size=par['size'], window=par['window'], min_count=par['min_count'], workers=multiprocessing.cpu_count(), sg=0)

    embedding_pdf = pd.DataFrame([(k, w2v_model.wv[k]) for k in w2v_model.wv.vocab.keys()], columns=['item', 'vector'])
  
  return embedding_pdf


def flatten_embedding(embedding_pdf):
  """ Convert embedding to 'flat' table that will be easier to save as CSV. """
  rows = []
  for index, row in embedding_pdf.iterrows():
    rows.append([row['item'], *row['vector']])
  return pd.DataFrame(rows, columns=['item', *[f"v{i:03d}" for i in range(len(embedding_pdf['vector'][0]))]])


def get_cluster_table(wv_pdf, num_clusters=[25, 50, 100, 200, 400, 800], metric='cosine', method='ward'):
  """ Generate table of cluster assignments for items in an embedding.
  
  Args:
    wv_pdf: word-vector Pandas datafram containing embedding for items, as produced by `fit_embedding`.
    num_clusters: a one-dimensional numeric array of cutpoints for cutting the hierarchical tree. These values are the number of clusters at the cutpoint.
    metric: distance metric (default 'cosine')
    method: aggomeration method (default 'item')
  """
  import scipy.spatial.distance as ssd
  import scipy.cluster.hierarchy as sch
  
  wv_mat = [[j for j in i] for i in wv_pdf['vector']]
  
  dist = ssd.pdist(wv_mat, metric=metric)
  hclust = sch.linkage(dist, method=method)
  cuts = sch.cut_tree(hclust, n_clusters=num_clusters)

  cluster_cols = [f'cluster{nc:03d}' for nc in num_clusters]
  cluster_assignment = pd.DataFrame(cuts, columns=cluster_cols)

  for col_idx in range(len(cluster_cols)):
    col = cluster_cols[col_idx]
    try:
      col_prefix = 'abcdefghijklmnopqrstuvwxyz'[col_idx]
    except IndexError:
      col_prefix = '?'
    cluster_assignment[col] = [f'{col_prefix}{cl:03d}' for cl in cluster_assignment[col]]

  cluster_assignment['item'] = wv_pdf['item']

  return cluster_assignment[['item', *cluster_cols]]
  sql_expr = f"mod(abs(hash({id_col})), {num_buckets})"
  return sdf.withColumn(id_bucket_col, fn.expr(sql_expr))


# Co-occurrence graphs

def get_item_pair_stats(basket_item_sdf, item_col='item', min_count=1):
  """ From a table of items and baskets, compute conditional probability-related statistics 
  (confidence, lift, etc) for pairs of items based on co-occurence.
  
  All columns other than the designated `item_col` are treated as a composite key specifying a 'basket'; 
  there will be a separate basket for each unique combination of these columns.
  
  Note that even though this function is coded as a SQL query it is not a `q_function` sinse it is not dependent on the database schema.
  """
  
  basket_cols=[col for col in basket_item_sdf.columns if col != item_col]
  basket_col_str = ', '.join(basket_cols)

  basket_item_sdf.createOrReplaceTempView("basket_item");
  
  q = f"""
    with 
      bi as (
        select {basket_col_str}, {item_col}
          from basket_item
          group by {basket_col_str}, {item_col}
      ),
      item_counts as (
        select item, count(*) item_count
          from bi
          group by item
      ),
      bi1 as (
        select bi.*, ic.item_count
          from bi
            join item_counts ic on bi.item=ic.item
          where ic.item_count > {min_count}
      ),
      bi2 as (
        select bi.*, ic.item_count
          from bi
            join item_counts ic on bi.item=ic.item
          where ic.item_count > {min_count}
      ),
      item_pair_stats as (
          select bi1.item item1, bi2.item item2,
                  bi1.item_count item1_count, bi2.item_count item2_count,
                  count(*) as both_count              
              from bi1
                join bi2
                  on bi1.basket = bi2.basket and bi1.item != bi2.item
              group by bi1.item, bi1.item_count, 
                       bi2.item, bi2.item_count
      ),
      cc as (
        SELECT item1, item2, item1_count, item2_count, both_count,
              CAST(item1_count AS FLOAT)/(select count(distinct basket) from basket_item) as item1_prevalence,
              CAST(item2_count AS FLOAT)/(select count(distinct basket) from basket_item) as item2_prevalence,
              CAST(both_count AS FLOAT)/CAST(item1_count AS FLOAT) AS confidence
          FROM item_pair_stats
      )
    select *, confidence/item2_prevalence lift from cc     
  """
  return spark.sql(q)


def benjamini_hochberg_filter(sdf, alpha=0.001, filter=True):
  """
  Input Spark dataframe must contain columns item1_count, both_count, item2_prevalence as produced by `get_item_pair_stats`
  """
  import pandas as pd
  from scipy.stats import norm
  m = sdf.count()
  z_to_pval_pudf = fn.pandas_udf(lambda z: pd.Series(norm.sf(abs(z))*2), DoubleType()) # two-sided p-value
  sdf2 = sdf\
    .withColumn("Z", fn.expr("(both_count - item1_count*item2_prevalence)/sqrt(item1_count*item2_prevalence*(1-item2_prevalence))")) \
    .withColumn("absZ", fn.expr("abs(Z)")) \
    .withColumn("Z_rank_fraction", fn.percent_rank().over(Window.orderBy(fn.desc("absZ")))) \
    .withColumn("p_value", z_to_pval_pudf(fn.col("Z"))) \
    .withColumn('benjamini_hochberg_criterion', fn.col('p_value') <= fn.col('Z_rank_fraction') * alpha)
  
  if filter:
    sdf2 = sdf2.filter('benjamini_hochberg_criterion')
    drop_cols = set(sdf.columns) - set(sdf2.columns)
    for col in drop_cols:
      sdf2 = sdf2.drop(col)
  
  return sdf2


def export_to_vis_js(cooccurrence_pdf, title, html_file_name):
    """
    Generate vis_js graph from cooccurrence Pandas dataframe and write to HTML file.
    """
    lift_coef = max(cooccurrence_pdf['lift'])
    weight_col='lift'
    item_stats = {r['item1']:{'count':r['item1_count'], 'prevalence':r['item1_prevalence']} for idx, r 
                  in cooccurrence_pdf.iterrows()}

    item_stats.update({r['item2']:{'count':r['item2_count'], 'prevalence':r['item2_prevalence']} for idx, r 
                  in cooccurrence_pdf.iterrows()})

    nodes_df = pd.DataFrame([{'label':k,'count':v['count'], 'prevalence':v['prevalence']}  for k,v in item_stats.items()])
    nodes_df['id'] = nodes_df.index

    node_id = {r['label']:r['id'] for idx, r in nodes_df.iterrows()}

    cooccurrence_pdf['from'] = [node_id[nn] for nn in cooccurrence_pdf['item1']]
    cooccurrence_pdf['to'] = [node_id[nn] for nn in cooccurrence_pdf['item2']]

    edges_df = cooccurrence_pdf[[ 'from', 'to', 'both_count', 'confidence', 'lift']]
    
    print("Your graph will have {0} nodes and {1} edges.".format( len(nodes_df), len(edges_df) ))
    
    G = nx.Graph()
    elist = [(r['from'], r['to'], r[weight_col]) for i, r in edges_df.iterrows()]
    G.add_weighted_edges_from(elist)
    dendro = community_louvain.generate_dendrogram(G)
    for level in range(0, len(dendro)):
        cluster_level_name = f"level_{level}_cluster"
        partition = community_louvain.partition_at_level(dendro, level)
        nodes_df[cluster_level_name] = [partition[node_id[x]] for x in nodes_df['label']]
    
    nodes_str = nodes_df.to_json(orient='records')
    edges_str = edges_df.to_json(orient='records')
    
    html_string = ( 
        '<!DOCTYPE html>\n'
        '<html lang="en">\n'
        '<head>\n'
        '	<meta http-equiv="content-type" content="text/html; charset=utf-8" />\n'
        f'	<title>{title}</title>\n'
        '	<script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>\n'
        f'	<script type="text/javascript">NODE_LIST={nodes_str};EDGE_LIST = {edges_str};</script>\n'
        '	<style type="text/css">#mynetwork {width: 100%; height: 700px; border: 1px}</style>\n'
        '	</head>\n'
        '		<body>\n'
        '			<div class="slidercontainer">\n'
        '				<label>minimum edge strength:\n'
        '					<input type="range" min="0" max="1" value="0.5" step="0.01" class="slider" id="min_edge_weight" \n'
        '							onchange="document.getElementById(\'min_edge_weight_display\').value=this.value;">\n'
        '					<input type="text" id="min_edge_weight_display" size="2" value="0.5">\n'
        '				</label>\n'
        '			</div>\n'
        '			<div id="mynetwork"></div>\n'
        '			<script type="text/javascript">\n'
        '	const urlParams = new URLSearchParams(window.location.search);\n'
        '	const weight_param = urlParams.get("weight")\n'
        '	const edge_weight_metric = (weight_param === null) ? "lift" : weight_param\n'
        '	for (var i = 0; i < EDGE_LIST.length; i++) {\n'
        '		EDGE_LIST[i]["arrows"] = "to"\n'
        '		EDGE_LIST[i]["value"] = EDGE_LIST[i][edge_weight_metric]\n'
        '	}\n'
        '	\n'
        '	const edgeFilterSlider = document.getElementById("min_edge_weight")\n'
        '	\n'
        f'	const filter_coef = {{"confidence":1, "lift":{lift_coef} }}\n'
        '	function edgesFilter(edge){return edge.value > edgeFilterSlider.value * filter_coef[edge_weight_metric]}\n'
        '	\n'
        '	const nodes = new vis.DataSet(NODE_LIST)\n'
        '	const edges = new vis.DataSet(EDGE_LIST)\n'
        '	\n'
        '	const nodesView = new vis.DataView(nodes)\n'
        '	const edgesView = new vis.DataView(edges, { filter: edgesFilter })\n'
        '	\n'
        '	edgeFilterSlider.addEventListener("change", (e) => {edgesView.refresh()})\n'
        '	\n'
        '	const container = document.getElementById("mynetwork")\n'
        '	const options = {physics:{maxVelocity: 10, minVelocity: 0.5}}\n'
        '	const data = { nodes: nodesView, edges: edgesView }\n'
        '	new vis.Network(container, data, options)\n'
        '	\n'
        '			</script>\n'
        '		</body>\n'
        '	</html>\n'
    )
    with open(html_file_name, "wt") as html_file:
        html_file.write(html_string)

