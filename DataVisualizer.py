import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class DataVisualizer:
    def __init__(self, original_data, preprocessed_data, category_mapping):
        self.original_data = original_data
        self.preprocessed_data = preprocessed_data
        self.category_mapping = category_mapping
                
        # Determine number of clusters and set a consistent color palette
        self.clusters = self.preprocessed_data['Cluster'].unique()
        self.colors = sns.color_palette("viridis", len(self.clusters))
    
    def plot_histograms_by_cluster_for_each_field(self, numeric_fields):
        num_clusters = len(self.clusters)

        # Iterate over each numeric field
        for field in numeric_fields:
            fig, axes = plt.subplots(1, num_clusters, figsize=(5 * num_clusters, 4), sharey=True)
            if num_clusters == 1:
                axes = [axes]  # Make axes iterable

            # Plot histograms for each cluster in subplots
            for i, cluster in enumerate(sorted(self.clusters)):
                cluster_data = self.original_data[self.original_data['Cluster'] == cluster]
                sns.histplot(cluster_data[field], ax=axes[i], color=self.colors[i], kde=False, bins=30)
                axes[i].set_title(f'Cluster {cluster}')
                axes[i].set_xlabel(field)
                axes[i].set_ylabel('Frequency')
                # Rotate x-axis labels
                for label in axes[i].get_xticklabels():
                    label.set_rotation(45)
                    label.set_ha('right')  # Set the horizontal alignment to right
                    
            plt.suptitle(f'Histograms of {field} by Cluster')
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for title
            plt.show(block=False)
        
    def plot_clustered_bar_chart(self):
        # Filter for tag columns and assume the existence of a 'Cluster' column
        tag_columns = [col for col in self.original_data.columns if col.startswith('tag_')]
        if not tag_columns:
            print("No tag columns found.")
            return

        # Melt the DataFrame so each tag is in its own row with the associated cluster
        melted = self.preprocessed_data.melt(id_vars=['Cluster'], value_vars=tag_columns, var_name='TagType', value_name='Tag')
        
        # Aggregate data
        grouped = melted.groupby(['Cluster', 'Tag']).size().reset_index(name='counts')

        # Plot
        plt.figure(figsize=(14, 10))
        sns.barplot(x='Cluster', y='counts', data=grouped)
        plt.title('Distribution of Tags across Clusters')
        plt.xlabel('Cluster')
        plt.ylabel('Counts of Tags')
        plt.legend(title='Tag', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show(block=False)
            
    def plot_tag_cluster_heatmap(self):
        # Filter for tag columns and assume the existence of a 'Cluster' column
        tag_columns = [col for col in self.preprocessed_data.columns if col.startswith('tag_')]
        if not tag_columns:
            print("No tag columns found.")
            return

        # Melt the DataFrame so each tag is in its own row with the associated cluster
        melted = self.preprocessed_data.melt(id_vars=['Cluster'], value_vars=tag_columns, var_name='TagType', value_name='Tag')
        
        # Create a crosstab of Cluster vs Tag
        crosstab = pd.crosstab(melted['Cluster'], melted['Tag'])
        if crosstab.isnull().any().any():
            print("Null values found in crosstab.")
        if np.isinf(crosstab).any().any():
            print("Infinite values found in crosstab.")
        
        # Check if crosstab is empty
        if crosstab.empty:
            print("Crosstab resulted in an empty DataFrame. Check tag and cluster data.")
            return
        
        # Create the heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(crosstab, cmap='viridis')
        plt.title('Frequency of Tags by Cluster')
        plt.xlabel('Tags')
        plt.ylabel('Clusters')
        plt.show(block=False)
        
    def plot_category_distribution(self):
        if 'category' in self.category_mapping:
            # Convert category indices from float to integer if needed
            self.preprocessed_data['category'] = self.preprocessed_data['category'].astype(int)
            self.preprocessed_data['category_name'] = self.preprocessed_data['category'].map(self.category_mapping['category'])
            plt.figure(figsize=(12, 8))
            sns.countplot(y='category_name', hue='Cluster', data=self.preprocessed_data, palette='viridis')
            plt.title('Category Distribution by Cluster')
            plt.xlabel('Count')
            plt.ylabel('Category')
            plt.show(block=False)
        else:
            print("Mapping for 'category' not found.")

    def plot_price_distribution_by_cluster(self):
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Cluster', y='price', data=self.original_data)
        plt.title('Price Distribution by Cluster')
        plt.xlabel('Cluster')
        plt.ylabel('Price')
        plt.show(block=False)   

    def plot_shop_age_vs_reviews(self):
        clusters = self.original_data['Cluster'].unique()
        n_clusters = len(clusters)
        colors = sns.color_palette("viridis", n_clusters)  # Generate a color palette

        # Create a grid of subplots
        fig, axes = plt.subplots(nrows=int(np.ceil(n_clusters / 3)), ncols=3, figsize=(15, n_clusters * 2))
        axes = axes.flatten()  # Flatten the axes array for easier iteration

        for i, cluster in enumerate(sorted(clusters)):
            # Filter data for each cluster
            cluster_data = self.original_data[self.original_data['Cluster'] == cluster]
            # Plot each cluster on its own subplot using a consistent color from the palette
            sns.scatterplot(ax=axes[i], data=cluster_data, x='shop_age', y='reviews', color=self.colors[i], alpha=0.7)
            axes[i].set_title(f'Cluster {cluster}')
            axes[i].set_xlabel('Shop Age (months)')
            axes[i].set_ylabel('Number of Reviews')

        # Hide any unused axes if the number of clusters is not a multiple of 3
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.show(block=False)

    def plot_favorites_vs_views(self):
        clusters = self.original_data['Cluster'].unique()
        n_clusters = len(clusters)
        colors = sns.color_palette("viridis", n_clusters)  # Generate a color palette

        # Create a grid of subplots
        fig, axes = plt.subplots(nrows=int(np.ceil(n_clusters / 3)), ncols=3, figsize=(15, n_clusters * 2))
        axes = axes.flatten()  # Flatten the axes array for easier iteration

        for i, cluster in enumerate(sorted(clusters)):
            # Filter data for each cluster
            cluster_data = self.original_data[self.original_data['Cluster'] == cluster]
            # Plot each cluster on its own subplot using a consistent color from the palette
            sns.scatterplot(ax=axes[i], data=cluster_data, x='favorites', y='views', color=self.colors[i], alpha=0.7)
            axes[i].set_title(f'Cluster {cluster}')
            axes[i].set_xlabel('Favorites')
            axes[i].set_ylabel('Views')

        # Hide any unused axes if the number of clusters is not a multiple of 3
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.show(block=False)