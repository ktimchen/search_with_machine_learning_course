import os
import argparse
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import csv

# Useful if you want to perform stemming.
import nltk
stemmer = nltk.stem.PorterStemmer()

categories_file_name = r'/workspace/datasets/product_data/categories/categories_0001_abcat0010000_to_pcmcat99300050000.xml'

queries_file_name = r'/workspace/datasets/train.csv'
output_file_name = r'/workspace/datasets/fasttext/labeled_queries.txt'

parser = argparse.ArgumentParser(description='Process arguments.')
general = parser.add_argument_group("general")
general.add_argument("--min_queries", default=1000,  help="The minimum number of queries per category label (default is 1)")
general.add_argument("--output", default=output_file_name, help="the file to output to")

args = parser.parse_args()
output_file_name = args.output

if args.min_queries:
    min_queries = int(args.min_queries)

# The root category, named Best Buy with id cat00000, doesn't have a parent.
root_category_id = 'cat00000'

tree = ET.parse(categories_file_name)
root = tree.getroot()

# Parse the category XML file to map each category id to its parent category id in a dataframe.
categories = []
parents = []
for child in root:
    id = child.find('id').text
    cat_path = child.find('path')
    cat_path_ids = [cat.find('id').text for cat in cat_path]
    leaf_id = cat_path_ids[-1]
    if leaf_id != root_category_id:
        categories.append(leaf_id)
        parents.append(cat_path_ids[-2])
parents_df = pd.DataFrame(list(zip(categories, parents)), columns =['category', 'parent'])

# Read the training data into pandas, only keeping queries with non-root categories in our category tree.
queries_df = pd.read_csv(queries_file_name)[['category', 'query']]
queries_df = queries_df[queries_df['category'].isin(categories)]

# IMPLEMENT ME: Convert queries to lowercase, and optionally implement other normalization, like stemming.
import re
NON_ALPHANUM_PATTERN = re.compile(r'[^a-zA-Z0-9]')

def process_non_alpha(x: str):
    x = re.sub(NON_ALPHANUM_PATTERN, ' ', x.lower())
    x = " ".join(stemmer.stem(word) for word in x.split())
    return x

queries_df["query"] = queries_df["query"].apply(process_non_alpha)

# IMPLEMENT ME: Roll up categories to ancestors to satisfy the minimum number of queries per category.
queries_counts_threshold = 1000



parents_lookup_table = parents_df.set_index("category")["parent"]
# just to be on a safe side
parents_df.loc[root_category_id] = root_category_id
print(parents_lookup_table)

print(f"Number of categories before roll-ups: {queries_df['category'].nunique()}")

queries_counts_df = (queries_df
 .groupby(["category"], as_index=False)
 [["query"]]
 .agg(q_counts = ("query", "count"))
)

cats_below_threshold = queries_counts_df.query("q_counts <= @queries_counts_threshold")["category"].to_list()
num_cats_under_threshold = len(cats_below_threshold)

while num_cats_under_threshold > 0:
    # bump all of the categories up simultaneously
    mask = queries_df["category"].isin(cats_below_threshold)
    queries_df.loc[mask, "category"] = queries_df.loc[mask, "category"].map(parents_lookup_table)
    queries_counts_df = (queries_df
        .groupby(["category"], as_index=False)
        [["query"]]
        .agg(q_counts = ("query", "count"))
    )
    cats_below_threshold = queries_counts_df.query("q_counts <= @queries_counts_threshold")["category"].to_list()
    num_cats_under_threshold = len(cats_below_threshold)
    print(f"Number of categories below the threshold: {num_cats_under_threshold}")

print(f"Number of categories after roll-ups: {queries_df['category'].nunique()}")


# Create labels in fastText format.
queries_df['label'] = '__label__' + queries_df['category']

# Output labeled query data as a space-separated file, making sure that every category is in the taxonomy.
queries_df = queries_df[queries_df['category'].isin(categories)]
queries_df['output'] = queries_df['label'] + ' ' + queries_df['query']
queries_df[['output']].to_csv(output_file_name, header=False, sep='|', escapechar='\\', quoting=csv.QUOTE_NONE, index=False)
