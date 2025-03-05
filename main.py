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


def plot_variances(pca, n_components = 25):
    plt.figure(figsize=(8, 5))
    plt.bar(range(1, n_components+1), pca.explained_variance_ratio_[:n_components], color='b', alpha=0.7)

    # Add labels and title
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance per Principal Component')
    plt.xticks(range(1, n_components + 1))  # Ensure x-ticks match component numbers
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    plt.savefig("explained_variance.png", dpi=300, bbox_inches='tight') 
    # Show plot
    plt.show()

def plot_pca(df, x, y, color_feature = 'font', folder = 'plots'):
    plt.figure(figsize=(8, 5))


    colors_categories = df[color_feature].astype('category')
    colors_codes = colors_categories.cat.codes  # Converts to numeric labels
    unique_colors = colors_categories.cat.categories
    num_categories = len(unique_colors)
    palette = sns.color_palette('tab20', n_colors=num_categories)
    cmap = ListedColormap(palette)
    col = [palette[i] for i in colors_codes]
    

    plt.scatter(df[x], df[y], s = 5, c=colors_codes, cmap=cmap)

    legend_handles = [mpatches.Patch(color=palette[i], label=color) for i, color in enumerate(unique_colors)]
    plt.legend(handles=legend_handles, title=color_feature, loc="best", fontsize="small")

    # Add labels and title
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title('PCA')

    plt.savefig(folder + '/' + y + '-' + x + '_by-' + color_feature + '.png', dpi=300, bbox_inches='tight') 
    # Show plot
    plt.show()

fonts = ['Times New Roman', 'DejaVu Serif', 'Georgia',  # serif fonts
'Arial', 'Helvetica', 'DejaVu Sans', 'Verdana',            # sans-serif
'Courier New', 'Consolas', 'DejaVu Sans Mono',          # mono
'Comic Sans MS', 'Papyrus']                             # fantasy and cursive




df = pd.read_csv('table.csv')

# QUANDO FILTRO, NELLA TABELLA FINALE LE RIGHE CON LE COORDINATE
# SONO ATTACCATE ALLA FINE DI UNA TABELLA VUOTA MA CON I LABEL CORRETTI
df = df.loc[df['case'] == 1]



# font and unicode number
labels = df.iloc[:, :3]
features = df.iloc[:, 3:]


scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

pca = PCA(n_components=0.99)  # Retain 99% variance
principal_components = pca.fit_transform(features_scaled)
n_components_chosen = pca.n_components_


pca_df = pd.DataFrame(principal_components, columns=[f'PC{i+1}' for i in range(principal_components.shape[1])])

# Concatenate labels with PCA result
final_df = pd.concat([labels, pca_df], axis=1)

#plot_variances(pca)

#plot_pca(final_df, x='PC1', y='PC2', color_feature='font', folder='A')

'''print(final_df.iloc[0,:3])

output = pca_to_original_space(final_df.iloc[0,3:], pca, scaler)

plt.imshow(output)
plt.show()'''

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Plot the first two principal components
scatter = ax1.scatter(final_df['PC1'], final_df['PC2'])
ax1.set_xlabel('PC1')
ax1.set_ylabel('PC2')
ax1.set_title('PCA - First Two Components')

# Function to handle mouse clicks and convert coordinates back
def on_click(event):
    # Get the coordinates of the mouse click in PCA space
    if event.inaxes != ax1:
        return
    
    # Extract PCA coordinates (mouse click position)
    pca_coordinates = np.zeros(n_components_chosen)
    pca_coordinates[0] = event.xdata
    pca_coordinates[1] = event.ydata
    
    # Convert PCA coordinates back to the original space
    reconstructed_data = pca_to_original_space(pca_coordinates, pca, scaler)

    # Reshape the reconstructed data to match the expected 20x20 shape
    # Assuming the reconstructed data is a 20x20 matrix (for example)
    reconstructed_data_image = reconstructed_data.reshape(20, 20)  # Adjust shape as necessary

    # Clear the second subplot and plot the reconstructed data as an image
    ax2.clear()
    im = ax2.imshow(reconstructed_data_image, cmap='viridis', interpolation='nearest')
    ax2.set_title('reconstructed letter')
    
    # Redraw the plot to reflect changes
    plt.draw()

# Connect the click event to the on_click function
fig.canvas.mpl_connect('button_press_event', on_click)

# Show the plot
plt.show()











'''reducer = umap.UMAP()

embedding = reducer.fit_transform(features_scaled)
print(embedding)
embedding_df = pd.DataFrame(embedding, columns = ['UMAP1', 'UMAP2'])
embedding_df = pd.concat([labels, embedding_df], axis=1)


plot_pca(embedding_df, x = 'UMAP1', y = 'UMAP2', colors = 'case')
plot_pca(embedding_df, x = 'UMAP1', y = 'UMAP2', colors = 'font')
plot_pca(embedding_df, x = 'UMAP1', y = 'UMAP2', colors = 'unicode')'''