# Python and OS
import os

# Data
import pandas as pd
import numpy as np

# Time
import time

# Clustering
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

# Plot
import probscale
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def main():
    print("------- ICP Clustering -------")
    # Vars to cluster (maybe different)
    # Al, B, Ba, Be, Bi, Ca, Cd, Ce, Co, Cr, Cs, Ga, Ge, Hg, K, La,
    # Mg, Mn, Na, Ni, Rb, Re, S, Sb, Sc, Se, Sr, Te, Th, U, V, W.

    # Parameters
    input_path = "<data_path>\\<file_name>.csv"  # input data
    output_path = "<output_path>"
    output_filepath = output_path + "data_clustered.csv"
    loc_columns = ['dhid', 'midx', 'midy', 'midz']  # location vars
    col_group = "litmod"  # column of the category to filter
    group = 1  # category to filter
    n_clusters = 4
    target_cols = ["Al_ICP_porc", "B_ICP_ppm", "Ba_ICP_ppm", "Be_ICP_ppm", "Bi_ICP_ppm", "Ca_ICP_porc", "Cd_ICP_ppm", "Ce_ICP_ppm", "Co_ICP_ppm", "Cr_ICP_ppm", "Cs_ICP_ppm", "Ga_ICP_ppm", "Ge_ICP_ppm", "Hg_ICP_ppm", "K_ICP_porc", "La_ICP_ppm", "Mg_ICP_porc", "Mn_ICP_ppm", "Na_ICP_porc", "Ni_ICP_ppm", "Rb_ICP_ppm", "Re_ICP_ppm", "S_ICP_porc", "Sb_ICP_ppm", "Sc_ICP_ppm", "Se_ICP_ppm", "Sr_ICP_ppm", "Te_ICP_ppm", "Th_ICP_ppm", "U_ICP_ppm", "V_ICP_ppm", "W_ICP_ppm"]
    cluster_col = "tag_agg" + '_' + str(n_clusters)

    # Get pandas data frame
    # Read Data
    df_group = pd.read_csv(input_path)
    print(df_group.head())
    print(df_group.shape)

    # Filter only target categorical value
    if col_group is not None and group is not None:
        df_group = filter_data_by_column_value(df_group, col_group, group)

    # Prepare Data
    df_to_cluster = prepare_data(target_cols=target_cols,
                                 loc_cols=loc_columns,
                                 df=df_group)

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
    df_to_cluster[cluster_col] = labels.tolist()
    print(set(labels))
    print("Clustering samples in {} seconds".format(time.time() - start_time))

    # Save
    df_to_cluster.to_csv(output_filepath, index=False)

    # Plot results
    start = time.time()

    # Get data frame
    df_plot = pd.read_csv(output_filepath)

    # Get target columns
    target_cols = find_column(target_cols, df_plot.columns)

    get_group_probplot(df_plot, cluster_col, target_cols, output_path)

    end = time.time()

    print("Elapsed Time: {}".format(end - start))


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
def prepare_data(target_cols, loc_cols, df):
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

    # print(df_to_cluster.describe())
    return df_to_cluster


def mult_conditions(df, target_cols, threshold):
    cond = False
    for col in target_cols:
        cond |= df[col] <= threshold
    return cond


def get_group_probplot(df, gcol, tcols, output_path):
    print("Get prob plot per group")
    groups = df.loc[:, gcol].unique().tolist()
    groups = np.sort(groups)

    color_dict = {}
    for k in groups:
        color_dict[k] = np.random.rand(3, )

    for col in tcols:
        # print("Column: {}".format(col))
        fig = plt.figure(figsize=(14, 14))
        for k in groups:
            common_opts = dict(
                plottype='prob', probax='y',
                datascale='log', problabel='Cumulative Probability', datalabel=k
            )

            df_k = df[df[gcol] == k].loc[:, col]

            scatter_kws_opts = dict(marker='.', markersize=5, alpha=0.6, c=color_dict[k], label=k)

            probscale.probplot(data=df_k.values,
                               ax=plt.gca(),
                               scatter_kws=scatter_kws_opts,
                               **common_opts)

        lgnd = plt.legend(loc='best', prop={'size': 20}, title="Clusters")
        for handle in lgnd.legendHandles:
            handle._legmarker.set_markersize(40)
        plt.setp(lgnd.get_title(), fontsize='20')

        fig.suptitle("Probplot: " + col, fontsize=20)
        plt.xlabel('Ordered values', fontsize=16)
        plt.ylabel('Cumulative Probability', fontsize=16)
        for axis in [plt.gca().xaxis]:
            formatter = FuncFormatter(lambda y, _: '{:.16g}'.format(y))
            axis.set_major_formatter(formatter)

        plt.grid(True, which="both", ls="-", color='0.65', alpha=0.5, linewidth=0.5)

        save_current_figure(output_path, "probplot", col)


def save_current_figure(output_path, name, prefix=None):
    output_path = output_path + name + "\\"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if prefix is not None:
        plt.savefig(output_path + prefix + "_" + name + ".png", dpi=90, bbox_inches='tight')
    else:
        plt.savefig(output_path + name + ".png", dpi=90, bbox_inches='tight')
    plt.close('all')


if __name__ == "__main__":
    main()
