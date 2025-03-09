import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import seaborn as sns
import numpy as np
import pandas as pd


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap

def pca_to_original_space(pca_coordinates, pca, scaler):
    # we need it as a numpy array
    if isinstance(pca_coordinates, pd.DataFrame) or isinstance(pca_coordinates, pd.Series):
        pca_coordinates = pca_coordinates.to_numpy()
    
    if pca_coordinates.ndim == 1:
        pca_coordinates = pca_coordinates.reshape(1, -1)

    # Step 1: Multiply the PCA coordinates by the transpose of the PCA components
    reconstructed_data = np.dot(pca_coordinates, pca.components_)

    # Step 2: Inverse standardization (reverse the scaling applied)
    reconstructed_data = scaler.inverse_transform(reconstructed_data)

    return reconstructed_data.reshape(20, 20)


def plot_variances(pca, n_components = 25, folder = 'plots',):
    plt.figure(figsize=(8, 5))
    plt.bar(range(1, n_components+1), pca.explained_variance_ratio_[:n_components], color='b', alpha=0.7)

    # Add labels and title
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance per Principal Component')
    plt.xticks(range(1, n_components + 1))  # Ensure x-ticks match component numbers
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    plt.savefig(folder + '/' +'explained_variance.png', dpi=300, bbox_inches='tight') 
    # Show plot
    plt.show()

def plot_pca(df, x, y, color_feature = 'font', folder = 'plots', title = ''):
    plt.figure(figsize=(8, 5))


    colors_categories = df[color_feature].astype('category')
    colors_codes = colors_categories.cat.codes  # Converts to numeric labels
    unique_colors = colors_categories.cat.categories
    num_categories = len(unique_colors)
    palette = sns.color_palette("rainbow", n_colors=num_categories)
    cmap = ListedColormap(palette)
    col = [palette[i] for i in colors_codes]
    

    plt.scatter(df[x], df[y], s = 5, c=colors_codes, cmap=cmap)

    legend_handles = [mpatches.Patch(color=palette[i], label=color) for i, color in enumerate(unique_colors)]
    plt.legend(handles=legend_handles, title=color_feature, loc="best", fontsize="small", ncol=2)

    # Add labels and title
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title('PCA')

    plt.savefig(folder + '/' + title + '_' + y + '-' + x + '_by-' + color_feature + '.png', dpi=300, bbox_inches='tight') 
    # Show plot
    plt.show()



# QUANDO FILTRO, NELLA TABELLA FINALE LE RIGHE CON LE COORDINATE
# SONO ATTACCATE ALLA FINE DI UNA TABELLA VUOTA MA CON I LABEL CORRETTI
#df = df.loc[df['case'] == 1]


'''print(final_df.iloc[0,:3])

output = pca_to_original_space(final_df.iloc[0,3:], pca, scaler)

plt.imshow(output)
plt.show()'''








'''reducer = umap.UMAP()

embedding = reducer.fit_transform(features_scaled)
print(embedding)
embedding_df = pd.DataFrame(embedding, columns = ['UMAP1', 'UMAP2'])
embedding_df = pd.concat([labels, embedding_df], axis=1)


plot_pca(embedding_df, x = 'UMAP1', y = 'UMAP2', colors = 'case')
plot_pca(embedding_df, x = 'UMAP1', y = 'UMAP2', colors = 'font')
plot_pca(embedding_df, x = 'UMAP1', y = 'UMAP2', colors = 'unicode')'''