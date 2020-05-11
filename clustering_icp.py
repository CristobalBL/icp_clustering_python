# Python and OS
from pathlib import Path

# Data
import pandas as pd

# Time
import time

# Clustering
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering


def main():
    print("------- ICP Clustering -------")

    # Parameters
    input_path = "C:\\Users\\crist\\Documents\\Work\\13_ModelClustering\\Datos\\data_comp.csv"  # input data
    output_path = "C:\\Users\\crist\\Documents\\Work\\13_ModelClustering\\clustering_samples_icp\\results\\"  # output dir

    # Vars to cluster (maybe different)
    # Al, B, Ba, Be, Bi, Ca, Cd, Ce, Co, Cr, Cs, Ga, Ge, Hg, K, La,
    # Mg, Mn, Na, Ni, Rb, Re, S, Sb, Sc, Se, Sr, Te, Th, U, V, W.

    loc_columns = ['dhid', 'midx', 'midy', 'midz']  # location vars
    data_columns = ['Al', 'B', 'Ba', 'Be', 'Bi', 'Ca', 'Cd', 'Ce', 'Co',
                    'Cr', 'Cs', 'Ga', 'Ge', 'Hg', 'K', 'La', 'Mg', 'Mn',
                    'Na', 'Ni', 'Rb', 'Re', 'S', 'Sb', 'Sc', 'Se', 'Sr',
                    'Te', 'Th', 'U', 'V', 'W']
    col_group = "litmod"
    group = 1
    n_clusters = 4

    # Get pandas data frame
    # Read Data
    df_group = pd.read_csv(input_path)
    print(df_group.head())
    print(df_group.shape)

    # Filter only target categorical value
    if col_group is not None and group is not None:
        df_group = filter_data_by_column_value(df_group, col_group, group)

    target_cols = find_column(data_columns, df_group.columns, extra_word="_ICP")  # find ICP columns

    # Prepare Data
    df_to_cluster = prepare_data(target_cols=target_cols,
                                 loc_cols=loc_columns,
                                 df=df_group,
                                 output_path=output_path)

    # Convert Data to Numpy Array
    df_target_data = df_to_cluster.loc[:, target_cols]
    np_data = df_target_data.values

    # Scale data with StandardScaler: z = (x - u) / s for a sample x
    np_array_scaled = StandardScaler().fit_transform(np_data)

    # Apply Agglomerative Clustering
    print("Agglomerative Clustering")
    print("N clusters: {}".format(n_clusters))
    start_time = time.time()
    agg_clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(np_array_scaled)
    labels = agg_clustering.labels_
    tag_str = "agg" + '_' + str(n_clusters)
    df_to_cluster['tag_' + tag_str] = labels.tolist()
    print(set(labels))
    print("Clustering samples in {} seconds".format(time.time() - start_time))

    # Save
    df_to_cluster.to_csv(output_path + "data_clustered.csv", index=False)


# Helper to get data
def get_data(data_path, group_col=None, group_value=None):
    # Read Data
    df = pd.read_csv(data_path)
    print(df.head())
    print(df.shape)

    if group_col is not None and group_value is not None:
        df = filter_data_by_column_value(df, group_col, group_value)

    return df


# Helper to find columns in dataframe
def find_column(cols_word, cols, extra_word=""):
    found_cols = []
    for word in cols_word:
        col = [k for k in cols if word + extra_word in k]
        if len(col) >= 0:
            found_cols.append(col[0])
    return found_cols


# Filter data column
def filter_data_by_column_value(df, group_col, group_value):
    df_group = df[df[group_col] == group_value]
    # print(df_group.head())
    # print(df_group.shape)
    return df_group


# Prepare data
def prepare_data(target_cols, loc_cols, df, output_path=None):
    # Remove not used columns
    df_to_cluster = df.loc[:, loc_cols + target_cols]
    # print(df_to_cluster.head())
    # print(df_to_cluster.shape)

    # Remove nan (-99.0)
    flags = mult_conditions(df_to_cluster, target_cols, 0.0)
    # print(flags)

    indexNames = df_to_cluster[flags].index
    df_to_cluster.drop(indexNames, inplace=True)

    # Convert ppm to perc
    ppm_cols = [k for k in df_to_cluster.columns if "ppm" in k]
    df_to_cluster[ppm_cols] = df_to_cluster[ppm_cols].apply(lambda x: x / 10000)

    if output_path is not None:
        Path(output_path).mkdir(parents=True, exist_ok=True)
        df_to_cluster.to_csv(output_path + "data_to_cluster.csv", index=False)

    # print(df_to_cluster.describe())
    return df_to_cluster


def mult_conditions(df, target_cols, threshold):
    cond = False
    for col in target_cols:
        cond |= df[col] <= threshold
    return cond


if __name__ == "__main__":
    main()
