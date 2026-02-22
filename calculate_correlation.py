"""
Script to Calculate Correlation Between Response Length and Adversarial Tags

This script demonstrates the complete workflow for analyzing LLM responses
and calculating the Pearson correlation coefficient between response length
and adversarial tags assigned by a meta-LLM.

Step-by-step guide:
1. Extract the Data: Load all raw game logs and extract LLM responses
2. Calculate Metrics: Compute Response_Length and Adversarial_Count for each response
3. Aggregate (Optional): Group by Model_Name to calculate average metrics
4. Calculate Pearson Correlation: Compute r coefficient
5. Interpret: Analyze the correlation strength
"""

import json
import os
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from collections import defaultdict
import re


class CorrelationAnalyzer:
    """Analyzes correlation between response length and adversarial tags."""
    
    def __init__(self, game_records_dirs):
        """
        Initialize the analyzer.
        
        Args:
            game_records_dirs: List of directories containing JSON game records
        """
        self.game_records_dirs = [Path(d) for d in game_records_dirs]
        self.responses_data = []
        
    def load_game_records(self):
        """
        Step 1: Extract and load all game records from JSON files.
        
        This method:
        - Scans the game_records directory for all .json files
        - Loads each JSON file
        - Extracts relevant LLM response information
        - Returns a list of dictionaries containing response metadata
        
        Returns:
            list: List of dictionaries with game record data
        """
        all_game_data = []
        total_files = 0
        for game_dir in self.game_records_dirs:
            game_files = list(game_dir.glob("*.json"))
            print(f"Found {len(game_files)} game record files in {game_dir}")
            total_files += len(game_files)
            for game_file in game_files:
                try:
                    with open(game_file, 'r', encoding='utf-8') as f:
                        game_data = json.load(f)
                        all_game_data.append(game_data)
                except (json.JSONDecodeError, IOError) as e:
                    print(f"Error loading {game_file}: {e}")
                    continue
        print(f"Successfully loaded {len(all_game_data)} game records from all folders")
        return all_game_data
    
    def extract_responses(self, all_game_data):
        """
        Step 2: Extract individual LLM responses from game data.
        
        For each game, this method extracts:
        - discussion_statements: Each player's discussion statement in day phases
        
        Each response is associated with:
        - game_id: The game this response came from
        - player_name: The model/player that generated the response
        - round_id: The round number
        - response_type: 'discussion_statement'
        
        Args:
            all_game_data: List of loaded game data dictionaries
            
        Returns:
            list: List of response records
        """
        responses = []
        for game in all_game_data:
            game_id = game.get('game_id', 'unknown')
            # Extract day phases
            for day_phase in game.get('day_phases', []):
                day_number = day_phase.get('day_number', 0)
                discussion_statements = day_phase.get('discussion_statements', [])
                if isinstance(discussion_statements, list):
                    for statement in discussion_statements:
                        for player_name, response_text in statement.items():
                            responses.append({
                                'game_id': game_id,
                                'player_name': player_name,
                                'day_number': day_number,
                                'response_type': 'discussion_statement',
                                'response_text': response_text,
                            })
                elif isinstance(discussion_statements, dict):
                    for player_name, response_text in discussion_statements.items():
                        responses.append({
                            'game_id': game_id,
                            'player_name': player_name,
                            'day_number': day_number,
                            'response_type': 'discussion_statement',
                            'response_text': response_text,
                        })
        print(f"Extracted {len(responses)} discussion statements from day phases")
        return responses
    
    def calculate_response_length(self, text):
        """
        Calculate the length of a response in words.
        
        This method:
        - Splits text by whitespace to count words
        - Handles empty or None strings gracefully
        - Can be modified to use token counts if needed
        
        Args:
            text (str): The response text
            
        Returns:
            int: Number of words in the text
            
        Note: For more accurate token counting, you could replace this with:
            - tiktoken.encoding_for_model("gpt-3.5-turbo").encode(text)
            - Or use a custom tokenizer from your LLM library
        """
        if not text:
            return 0
        # Split by whitespace and filter empty strings
        words = text.split()
        return len(words)
    
    def count_adversarial_tags(self, text):
        """
        Count adversarial tags in a response.
        
        This method counts occurrences of adversarial behavior indicators.
        Adjust the patterns based on your meta-LLM's output format.
        
        Common patterns might include:
        - <adversarial> tags (if using XML-style markup)
        - [ADVERSARIAL] tags
        - Specific keyword indicators
        
        Args:
            text (str): The response text to analyze
            
        Returns:
            int: Count of adversarial tags/indicators
            
        TODO: Update these patterns based on your actual meta-LLM output format
        """
        if not text:
            return 0
        
        # Count different adversarial tag patterns
        # Modify these patterns based on your meta-LLM's output
        count = 0
        
        # Pattern 1: XML-style tags
        count += len(re.findall(r'<adversarial>', text, re.IGNORECASE))
        count += len(re.findall(r'\[ADVERSARIAL\]', text, re.IGNORECASE))
        
        # Pattern 2: Keyword-based (adjust keywords as needed)
        adversarial_keywords = ['deceptive', 'bluff', 'manipulative', 'hostile']
        for keyword in adversarial_keywords:
            # Count exact word matches only
            pattern = r'\b' + keyword + r'\b'
            count += len(re.findall(pattern, text, re.IGNORECASE))
        
        return count
    
    def create_response_dataframe(self, responses):
        """
        Step 3: Create a DataFrame with calculated metrics for each response.
        
        This method:
        - Calculates Response_Length (words) for each response
        - Calculates Adversarial_Count using tag detection
        - Creates a pandas DataFrame for analysis
        
        Args:
            responses: List of response records
            
        Returns:
            pd.DataFrame: DataFrame with columns:
                - game_id
                - player_name
                - round_id
                - response_type
                - response_text
                - Response_Length
                - Adversarial_Count
        """
        print("\n=== STEP 1: Calculating Response Metrics ===")
        
        # Calculate metrics for each response
        for response in responses:
            response['Response_Length'] = self.calculate_response_length(
                response.get('response_text', '')
            )
            response['Adversarial_Count'] = self.count_adversarial_tags(
                response.get('response_text', '')
            )
        
        # Create DataFrame
        df = pd.DataFrame(responses)
        
        print(f"Created DataFrame with {len(df)} rows")
        print("\nFirst few rows:")
        print(df[['player_name', 'response_type', 'Response_Length', 
                   'Adversarial_Count']].head(10))
        print("\nDataFrame Statistics:")
        print(df[['Response_Length', 'Adversarial_Count']].describe())
        
        return df
    
    def create_aggregated_dataframe(self, df):
        """
        Step 4 (Optional): Aggregate responses by model to reduce variance.
        
        Instead of using every single message, this method:
        - Groups responses by player_name (each LLM model)
        - Calculates average Response_Length per model
        - Calculates total/average Adversarial_Count per model
        - Reduces the number of data points to 10 (one per model)
        
        Args:
            df: DataFrame with individual response metrics
            
        Returns:
            pd.DataFrame: Aggregated DataFrame with one row per model
        """
        print("\n=== STEP 2 (Optional): Aggregating by Model ===")
        
        # Group by player_name (model)
        aggregated = df.groupby('player_name').agg({
            'Response_Length': ['mean', 'count'],
            'Adversarial_Count': ['sum', 'mean']
        }).reset_index()
        
        # Flatten column names
        aggregated.columns = ['Model_Name', 'Avg_Response_Length', 
                             'Response_Count', 'Total_Adversarial_Count',
                             'Avg_Adversarial_Count']
        
        print(f"\nAggregated to {len(aggregated)} models:")
        print(aggregated)
        
        return aggregated
    
    def calculate_correlation_granular(self, df):
        """
        Calculate Pearson correlation at the granular level (every single message).
        
        Args:
            df: DataFrame with all individual responses
            
        Returns:
            tuple: (correlation coefficient r, p-value)
        """
        print("\n=== STEP 3: Calculating Pearson Correlation (Granular) ===")
        
        # Filter out any rows with missing data
        df_clean = df.dropna(subset=['Response_Length', 'Adversarial_Count'])
        
        if len(df_clean) < 2:
            print("Not enough data points for correlation")
            return None, None
        
        # Calculate Pearson correlation
        r, p_value = pearsonr(df_clean['Response_Length'], 
                             df_clean['Adversarial_Count'])
        
        print(f"\nGranular Level Analysis (n={len(df_clean)} messages):")
        print(f"  Pearson Correlation Coefficient (r): {r:.4f}")
        print(f"  P-value: {p_value:.6f}")
        
        return r, p_value
    
    def calculate_correlation_aggregated(self, df_agg):
        """
        Calculate Pearson correlation on aggregated data (by model).
        
        Args:
            df_agg: Aggregated DataFrame with one row per model
            
        Returns:
            tuple: (correlation coefficient r, p-value)
        """
        print("\n=== STEP 3: Calculating Pearson Correlation (Aggregated) ===")
        
        if len(df_agg) < 2:
            print("Not enough models for correlation")
            return None, None
        
        # Calculate Pearson correlation on aggregated data
        r, p_value = pearsonr(df_agg['Avg_Response_Length'], 
                             df_agg['Avg_Adversarial_Count'])
        
        print(f"\nAggregated Level Analysis (n={len(df_agg)} models):")
        print(f"  Pearson Correlation Coefficient (r): {r:.4f}")
        print(f"  P-value: {p_value:.6f}")
        
        return r, p_value
    
    def interpret_correlation(self, r, p_value):
        """
        Step 5: Interpret the correlation coefficient.
        
        Interpretation guidelines:
        - r ≈ 0 (-0.2 to 0.2): No correlation
        - r ≈ ±0.3 to ±0.5: Weak correlation
        - r ≈ ±0.5 to ±0.7: Moderate correlation
        - r ≈ ±0.7 to ±0.9: Strong correlation
        - r ≈ ±1.0: Perfect correlation
        
        P-value interpretation:
        - p < 0.05: Statistically significant (reject null hypothesis)
        - p ≥ 0.05: Not statistically significant
        
        Args:
            r: Pearson correlation coefficient
            p_value: P-value from the correlation test
        """
        print("\n=== STEP 4: Interpreting Results ===\n")
        
        if r is None:
            print("Cannot interpret - insufficient data")
            return
        
        print(f"Correlation Coefficient: r = {r:.4f}")
        print(f"P-value: {p_value:.6f}")
        
        # Strength interpretation
        print("\n--- Strength of Relationship ---")
        abs_r = abs(r)
        if abs_r < 0.2:
            strength = "No correlation"
            details = "There is no meaningful linear relationship between response length and adversarial tags."
        elif abs_r < 0.4:
            strength = "Weak correlation"
            details = "There is a weak linear relationship. Response length and adversarial tags are loosely related."
        elif abs_r < 0.6:
            strength = "Moderate correlation"
            details = "There is a moderate linear relationship. Changes in one variable are moderately associated with changes in the other."
        elif abs_r < 0.8:
            strength = "Strong correlation"
            details = "There is a strong linear relationship. Response length and adversarial tags are closely related."
        else:
            strength = "Very strong correlation"
            details = "There is a very strong linear relationship. The variables are highly dependent on each other."
        
        print(f"Strength: {strength}")
        print(f"Details: {details}")
        
        # Direction interpretation
        print("\n--- Direction of Relationship ---")
        if r > 0:
            direction = "positive"
            directional_text = "As response length increases, adversarial tag count tends to increase."
        else:
            direction = "negative"
            directional_text = "As response length increases, adversarial tag count tends to decrease."
        
        print(f"Direction: {direction.capitalize()}")
        print(f"Details: {directional_text}")
        
        # Statistical significance
        print("\n--- Statistical Significance ---")
        if p_value < 0.05:
            significance = "Yes (p < 0.05)"
            sig_text = "This correlation is statistically significant. We reject the null hypothesis of no correlation."
        else:
            significance = "No (p ≥ 0.05)"
            sig_text = "This correlation is NOT statistically significant. We fail to reject the null hypothesis of no correlation."
        
        print(f"Significant: {significance}")
        print(f"Details: {sig_text}")
        
        # Practical interpretation
        print("\n--- Practical Interpretation ---")
        r_squared = r ** 2
        print(f"R-squared (r²): {r_squared:.4f}")
        print(f"This means {r_squared * 100:.2f}% of the variance in one variable")
        print(f"is explained by the variance in the other variable.")
        
        # Recommendations
        print("\n--- Recommendations for Further Analysis ---")
        if abs_r < 0.3:
            print("• Low correlation suggests other factors may be important")
            print("• Consider analyzing by model type or response category separately")
            print("• Examine if there are non-linear relationships")
            print("• Look for potential confounding variables")
        elif abs_r > 0.7:
            print("• Strong correlation suggests a robust relationship")
            print("• Be cautious about causal interpretation")
            print("• Consider whether longer responses naturally contain more tags")
            print("• Validate findings with additional data if possible")
        else:
            print("• Moderate correlation warrants further investigation")
            print("• Consider segment-specific analysis")
            print("• Look for threshold effects or non-linear patterns")
    
    def run_full_analysis(self):
        """Run the complete analysis pipeline."""
        print("=" * 70)
        print("CORRELATION ANALYSIS: Response Length vs Adversarial Tags")
        print("=" * 70)
        
        # Step 1: Load game records
        print("\n=== STEP 0: Loading Game Records ===")
        all_game_data = self.load_game_records()
        
        # Step 2: Extract responses
        responses = self.extract_responses(all_game_data)
        
        # Step 3: Create DataFrame with metrics
        df = self.create_response_dataframe(responses)
        
        # Step 4: Create aggregated DataFrame
        df_agg = self.create_aggregated_dataframe(df)

        # Replace adversarial counts with user-provided values
        user_adversarial_counts = {
            'Derek': 655,
            'Enrique': 591,
            'George': 688,
            'Maria': 844,
            'Sarah': 666,
            'Anika': 855,
            'Peter': 512,
            'Philip': 941,
            'Talia': 677,
            'Emma': 997
        }
        # Update Total_Adversarial_Count column
        df_agg['Total_Adversarial_Count'] = df_agg['Model_Name'].map(user_adversarial_counts)

        # Calculate Pearson correlation using Avg_Response_Length and Total_Adversarial_Count
        print("\n=== STEP: Calculating Pearson Correlation with User Values ===")
        valid_df = df_agg.dropna(subset=['Avg_Response_Length', 'Total_Adversarial_Count'])
        r, p_value = pearsonr(valid_df['Avg_Response_Length'], valid_df['Total_Adversarial_Count'])
        print(f"Pearson Correlation Coefficient (r): {r:.4f}")
        print(f"P-value: {p_value:.6f}")

        # Save results to CSV
        print("\n=== Saving Results ===")
        valid_df.to_csv('correlation_analysis_aggregated_user_branch.csv', index=False)
        print("Aggregated results with user values saved to: correlation_analysis_aggregated_user_branch.csv")

        # Save correlation summary
        summary_data = {
            'Correlation_Coefficient_r': [r],
            'P_Value': [p_value],
            'Sample_Size': [len(valid_df)],
            'Statistically_Significant': [p_value < 0.05]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv('correlation_analysis_summary_user_branch.csv', index=False)
        print("Summary results saved to: correlation_analysis_summary_user_branch.csv")

        print("\n" + "=" * 70)
        print("Analysis Complete!")
        print("=" * 70)


def main():
    """Main entry point for the correlation analysis script."""
    
    # Set the path to your game records directory
    game_records_dirs = ["./game_records"]
    # Create analyzer and run analysis
    analyzer = CorrelationAnalyzer(game_records_dirs)
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()
