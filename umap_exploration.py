from main import *

df = pd.read_csv('table0.csv')


# font and unicode number
labels = df.iloc[:, :4]
features = df.iloc[:, 4:]


scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

reducer = umap.UMAP()

embedding = reducer.fit_transform(features_scaled)
print(embedding)
embedding_df = pd.DataFrame(embedding, columns = ['UMAP1', 'UMAP2'])
embedding_df = pd.concat([labels, embedding_df], axis=1)

embedding_df_UPPER = embedding_df[embedding_df['case'] == 1]
embedding_df_LOWER = embedding_df[embedding_df['case'] == 0]

plot_pca(embedding_df, x = 'UMAP1', y = 'UMAP2', color_feature = 'case')

plot_pca(embedding_df_UPPER, x = 'UMAP1', y = 'UMAP2', color_feature = 'letter', title='upper')
plot_pca(embedding_df_LOWER, x = 'UMAP1', y = 'UMAP2', color_feature = 'letter', title='lower')