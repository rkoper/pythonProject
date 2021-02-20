corr = data05.astype('float64').corr()
corr

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr, mask=mask, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


plt.figure(figsize=(15, 10))
sns.heatmap(corr, annot=True)
plt.title("Schema 56 - HeatMap : Etudes de Correlation / 1", fontsize=15)
plt.show()
