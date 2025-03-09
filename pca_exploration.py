from main import *

df = pd.read_csv('table0.csv')


# font and unicode number
labels = df.iloc[:, :4]
features = df.iloc[:, 4:]


scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

pca = PCA(n_components=0.95)  # Retain 99% variance
principal_components = pca.fit_transform(features_scaled)
n_components_chosen = pca.n_components_


pca_df = pd.DataFrame(principal_components, columns=[f'PC{i+1}' for i in range(principal_components.shape[1])])

# Concatenate labels with PCA result
final_df = pd.concat([labels, pca_df], axis=1)

print(final_df[final_df['PC1'] > 40])

final_df_UPPER = final_df[final_df['case'] == 1]
final_df_LOWER = final_df[final_df['case'] == 0]

plot_variances(pca)

plot_pca(final_df, x='PC1', y='PC2', color_feature='case', folder='plots')

plot_pca(final_df_UPPER, x='PC1', y='PC2', color_feature='letter', folder='plots', title = 'upper')
plot_pca(final_df_UPPER, x='PC3', y='PC4', color_feature='letter', folder='plots', title = 'upper')

plot_pca(final_df_LOWER, x='PC1', y='PC2', color_feature='letter', folder='plots', title = 'lower')
plot_pca(final_df_LOWER, x='PC3', y='PC4', color_feature='letter', folder='plots', title = 'lower')

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