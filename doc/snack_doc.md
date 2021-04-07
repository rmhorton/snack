# Table of Contents

* [snack.snack](#snack.snack)
  * [timestamp\_to\_timeslice](#snack.snack.timestamp_to_timeslice)
  * [expand\_rows](#snack.snack.expand_rows)
  * [fill\_missing\_values\_forward](#snack.snack.fill_missing_values_forward)
  * [add\_id\_bucket\_column](#snack.snack.add_id_bucket_column)
  * [get\_basket\_items](#snack.snack.get_basket_items)
  * [fit\_embedding](#snack.snack.fit_embedding)
  * [flatten\_embedding](#snack.snack.flatten_embedding)
  * [get\_cluster\_table](#snack.snack.get_cluster_table)
  * [get\_item\_pair\_stats](#snack.snack.get_item_pair_stats)
  * [benjamini\_hochberg\_filter](#snack.snack.benjamini_hochberg_filter)

<a name="snack.snack"></a>
# snack.snack

<a name="snack.snack.timestamp_to_timeslice"></a>
#### timestamp\_to\_timeslice

```python
timestamp_to_timeslice(sdf, timestamp_col, timeslice_id_col, time_unit='hour')
```

Convert timestamp column to integer ID representing number of time units since Unix epoch.

**Arguments**:

- `sdf` _Spark dataframe_ - input dataframe.
- `timestamp_col` _str_ - name of input timestamp column; this will be replaced by a timestamp_id.
- `timeslice_id_col` _str_ - name of the column to be generated
- `time_unit` _str_ - period of time defined by a fixed number of seconds
  ('day', 'hour', etc. as defined in `seconds_per` dict in this function)
  

**Returns**:

  Spark dataframe with timestamp column replaced by timeslice ID.

<a name="snack.snack.expand_rows"></a>
#### expand\_rows

```python
expand_rows(sdf, from_col, to_col, sequence_col_name, *id_cols)
```

Expand a range of integers into a set of rows representing all values in the sequence.

**Arguments**:

  
- `sdf` _spark dataframe_ - input dataframe
- `from_col` _str_ - name of column specifying beginning integer value of sequence.
- `to_col` _str_ - name of column specifying ending integer value of sequence
- `sequence_col_name` _str_ - name of new column to be generated with sequence values
- `id_col_names` _array of str_ - names of id columns
  

**Returns**:

  
  spark dataframe with columns specified in `id_col_names` and `sequence_col_name`, with one row per sequence element.
  
  Example::
  
  range_df = spark.createDataFrame(data=[('Ali',3,7), ('Bay',5,10), ('Cal',1,3)], schema = ['name','from','to'])
  expand_rows(range_df, 'from', 'to', 'sequence_id', 'name').show()

<a name="snack.snack.fill_missing_values_forward"></a>
#### fill\_missing\_values\_forward

```python
fill_missing_values_forward(sdf, ordering_col, cols_to_fill, *id_cols)
```

Fill missing values by carrying previous values forward.

**Arguments**:

- `sdf` - a Spark DataFrame
- `ordering_col` - column by which rows should be sorted
- `cols_to_fill` - list of columns where missing values will be filled
- `id_cols` - list of columns that collectively form a unique identifier that can be used to partition cases.

<a name="snack.snack.add_id_bucket_column"></a>
#### add\_id\_bucket\_column

```python
add_id_bucket_column(sdf, id_col='pat_id', id_bucket_col='id_bucket', num_buckets=12)
```

Assign each id value to one of a given set of buckets.

Buckets can be used for train/validate/test splits.

**Arguments**:

  
  id_col:
- `id_bucket_col` - name of new column to generate
- `num_buckets` - the number of buckets

<a name="snack.snack.get_basket_items"></a>
#### get\_basket\_items

```python
get_basket_items(sdf, item_col, *key_cols, *, include_duplicate_items=True, exclude_single_item_baskets=True)
```

generate sets of items from a table listing items individually (along with the group they belong to)

Params:
sdf: input Spark dataframe
item_col: the name of the column indicating the item
*key_cols: the names of the columns that collectively indicate the group the item should be placed in
include_duplicate_items:
exclude_single_item_basekets: if Trye, baskets with only a single item will be filtered out

**Notes**:

  This function is more or less equivalent to this SQL query:
  
  select pat_id, prim_enc_csn_id, encounter_date,
  collect_list(distinct icd_code) items,
  count(distinct icd_code) num_items
  from encounter_icd
  group by pat_id, prim_enc_csn_id, encounter_date

<a name="snack.snack.fit_embedding"></a>
#### fit\_embedding

```python
fit_embedding(basket_item_sdf, implementation='pyspark', **new_par)
```

Use create Word2Vec embedding for the items in baskets.

The gensim implementaion is not scalable; it aggregates the input Spark dataframe to a local Pandas dataframe with items aggregated by basket, and returns the embedding results as a Pandas dataframe. The gensim implementation has more (or at least different) options compared to the pyspark implementation (e.g., pyspark only does skipgram method).

Param:
basket_item_sdf: spark dataframe with basket id (may be multiple columns) and items vector.
implementation: 'gensim' or 'pyspark'
new_par: named parameters depend on the implementation (see `par` defaults in code.)

**Returns**:

  a Pandas dataframe with two columns, 'word' and 'vector'

<a name="snack.snack.flatten_embedding"></a>
#### flatten\_embedding

```python
flatten_embedding(embedding_pdf)
```

Convert embedding to 'flat' table that will be easier to save as CSV.

<a name="snack.snack.get_cluster_table"></a>
#### get\_cluster\_table

```python
get_cluster_table(wv_pdf, num_clusters=[25, 50, 100, 200, 400, 800], metric='cosine', method='ward')
```

Generate table of cluster assignments for items in an embedding.

Params:
  wv_pdf: word-vector Pandas datafram containing embedding for items, as produced by `fit_embedding`.
  num_clusters: a one-dimensional numeric array of cutpoints for cutting the hierarchical tree. These values are the number of clusters at the cutpoint.
  metric: distance metric (default 'cosine')
  method: aggomeration method (default 'item')

<a name="snack.snack.get_item_pair_stats"></a>
#### get\_item\_pair\_stats

```python
get_item_pair_stats(basket_item_sdf, item_col='item', min_count=1)
```

From a table of items and baskets, compute conditional probability-related statistics
(confidence, lift, etc) for pairs of items based on co-occurence.

All columns other than the designated `item_col` are treated as a composite key specifying a 'basket'; 
there will be a separate basket for each unique combination of these columns.

Note that even though this function is coded as a SQL query it is not a `q_function` sinse it is not dependent on the database schema.

<a name="snack.snack.benjamini_hochberg_filter"></a>
#### benjamini\_hochberg\_filter

```python
benjamini_hochberg_filter(sdf, alpha=0.001, filter=True)
```

Input Spark dataframe must contain columns item1_count, both_count, item2_prevalence as produced by `get_item_pair_stats`

