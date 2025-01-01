import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_plots(file_path):
    """
    Generates Working Time vs. Centroids and Dice Score vs. Centroids plots from a given .txt file.

    Parameters:
    - file_path (str): Path to the .txt data file.
    """
    try:
        # Check if the file exists
        if not os.path.isfile(file_path):
            print(f"Error: The file '{file_path}' does not exist.")
            return

        # Extract experiment name from file name (without extension)
        experiment_name = "Entropy_Weights_128"
        
        # Read the data
        # Assuming the delimiter is either tab or spaces
        df = pd.read_csv(file_path, delim_whitespace=True)
        
        # Display first few rows (optional)
        print(f"\nProcessing '{experiment_name}':")
        print(df.head())
        
        # Required columns for plotting
        required_columns = ['Num_Centroids', 'Centroid_time', 'Dice_Score']
        if not all(col in df.columns for col in required_columns):
            print(f"Error: The file must contain the following columns: {required_columns}")
            return
        
        # Create output directory for plots if it doesn't exist
        output_dir = 'plots'
        os.makedirs(output_dir, exist_ok=True)
        
        # --- Plot 1: Working Time vs. Centroids ---
        plt.figure(figsize=(8, 6))
        plt.plot(df['Num_Centroids'], df['Centroid_time'], marker='o', linestyle='-', color='blue')
        plt.title(f'{experiment_name} - Working Time vs. Centroids')
        plt.xlabel('Number of Centroids')
        plt.ylabel('Working Time (Centroid_time)')
        plt.grid(True)
        plt.tight_layout()
        plot1_filename = f"{experiment_name}_WorkingTime_vs_Centroids.png"
        plot1_path = os.path.join(output_dir, plot1_filename)
        plt.savefig(plot1_path)
        plt.close()
        print(f"Saved plot: {plot1_path}")
        
        # --- Plot 2: Dice Score vs. Centroids ---
        plt.figure(figsize=(8, 6))
        plt.plot(df['Num_Centroids'], df['Dice_Score'], marker='s', linestyle='--', color='green')
        plt.title(f'{experiment_name} - Dice Score vs. Centroids')
        plt.xlabel('Number of Centroids')
        plt.ylabel('Dice Score')
        plt.grid(True)
        plt.tight_layout()
        plot2_filename = f"{experiment_name}_DiceScore_vs_Centroids.png"
        plot2_path = os.path.join(output_dir, plot2_filename)
        plt.savefig(plot2_path)
        plt.close()
        print(f"Saved plot: {plot2_path}")
        
        print("\nAll plots have been generated and saved in the 'plots' directory.")
        
    except pd.errors.EmptyDataError:
        print(f"Error: The file '{file_path}' is empty or malformed.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # --- Specify the path to your .txt file here ---
    # Example: 'experiment_data/experiment1.txt'
    data_file_path = '/home/chopra/lab-git/MedImSeg-Lab24/results/entropy_weights/128/results_test_100.txt'  # <-- Replace this with your actual file path
    
    generate_plots(data_file_path)
