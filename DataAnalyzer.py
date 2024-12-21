import json
import os
from DataLoader import DataLoader
from DataPreprocessor import DataPreprocessor
from ClusterData import ClusterData
from DataVisualizer import DataVisualizer
from ClusterEvaluator import ClusterEvaluator

CONFIG_FILE = "config.json"

def load_config():
    """Load configuration from a JSON file."""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as file:
            return json.load(file)
    return {}

def save_config(config):
    """Save configuration to a JSON file."""
    with open(CONFIG_FILE, 'w') as file:
        json.dump(config, file, indent=4)

def get_file_path(config, key, prompt):
    """Get file path from config or prompt user if not available."""
    if key in config:
        return config[key]
    path = input(prompt).strip()
    config[key] = path
    save_config(config)
    return path

def main():
    
    # Load or initialize configuration
    config = load_config()

    # Get file paths from config or prompt user
    input_file_path = get_file_path(config, "input_file_path", "Enter the path to the input CSV file: ")
    output_clustered_file_path = get_file_path(config, "output_clustered_file_path", "Enter the path to save the clustered data Excel file: ")
    output_original_with_clusters_path = get_file_path(config, "output_original_with_clusters_path", "Enter the path to save the original data with clusters Excel file: ")
    
    # Initialize data loader and load data
    loader = DataLoader(input_file_path)
    original_data = loader.load_csv()

    # Initialize and process the data
    preprocessor = DataPreprocessor(original_data)
    preprocessor.preprocess()  # Ensure data is ready for clustering
    
    #Pass the mappings to the DataVisualizer
    category_mappings = preprocessor.category_mapping  # Assuming this dictionary is returned from preprocess
    
    # Now you can access the preprocessed and the original data
    preprocessed_data = preprocessor.get_preprocessed_data()
    original_data = preprocessor.get_original_data()

    # Now the data is preprocessed, we can use it to determine the best number of clusters
    evaluator = ClusterEvaluator(preprocessor.data, max_k=10)
    # Uncomment these to view the inertia and silhoutte scores
    evaluator.calculate_inertia()
    evaluator.calculate_silhouette_scores()

    # After determining the best k, perform actual clustering
    best_k = 4  # This should be set based on your analysis from the inertia and silhouette scores
    clusterer = ClusterData(preprocessed_data, num_clusters=best_k)
    clustered_data = clusterer.perform_clustering()
        
    # Save the DataFrame with the cluster labels to a new Excel file
    clustered_data.to_excel(output_clustered_file_path, index=False)

    # Optionally, save the original data with cluster labels
    original_data['Cluster'] = clustered_data['Cluster']

    original_data.to_excel(output_original_with_clusters_path, index=False)
    
    # Initialize visualizer with category mappings
    visualizer = DataVisualizer(original_data, preprocessed_data, category_mappings)

    # Call updated visualization methods
    visualizer.plot_price_distribution_by_cluster()
    visualizer.plot_shop_age_vs_reviews()
    visualizer.plot_favorites_vs_views()
    visualizer.plot_category_distribution()  # This now uses the mappings to plot with original category names

    # Plot the tag-cluster heatmap
    #visualizer.plot_tag_cluster_heatmap()  # Assuming 'clustered_data' has the 'Cluster' and 'tag_x' columns
    visualizer.plot_clustered_bar_chart()
    
    # Define the list of numeric fields you want to include in the histograms
    numeric_fields = ['price', 'reviews', 'listing_age', 'favorites', 'avg_reviews', 'views', 'shop_age', 'total_shop_sales']

    # Call the method with the list of numeric fields
    visualizer.plot_histograms_by_cluster_for_each_field(numeric_fields)
    
    # Ensure plots do not close immediately
    input("Press Enter to exit...")

if __name__ == '__main__':
    main()
