"""
Dataset builder for creating train/val/test splits with interaction terms.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import gc


class HighRAMDatasetBuilder:
    """
    Builder for creating ML-ready datasets with firm-macro interactions.

    Pipeline:
    1. Merge firm and macro data on yyyymm
    2. Calculate excess returns (ret - Rfree)
    3. Create interaction terms: firm_chars × macro_predictors
    4. Split by time into train/val/test
    """

    def __init__(self, firm_df: pd.DataFrame, macro_df: pd.DataFrame):
        """
        Initialize dataset builder.

        Args:
            firm_df: Processed firm characteristics DataFrame
            macro_df: Processed macro variables DataFrame
        """
        self.firm_df = firm_df.copy()
        self.macro_df = macro_df.copy()
        self.full_df = None

    def merge_and_calculate_target(self) -> pd.DataFrame:
        """
        Merge firm and macro data, calculate excess returns.

        Returns:
            Merged DataFrame with excess_ret column

        Side Effects:
            Sets self.full_df
        """
        print("--- Step 1: Merging Firm & Macro Data ---")
        self.full_df = pd.merge(self.firm_df, self.macro_df, on='yyyymm', how='inner')

        print("Calculating Excess Returns (ret_tplus1 - Rfree)...")
        self.full_df['excess_ret'] = self.full_df['ret_tplus1'] - self.full_df['Rfree']
        self.full_df = self.full_df.dropna(subset=['excess_ret']).reset_index(drop=True)

        print(f"Total Rows: {len(self.full_df):,}")
        return self.full_df

    def create_interactions(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate interaction terms between firm characteristics and macro predictors.

        Creates features: [firm_chars] + [firm_chars × macro_pred1] + ...

        Returns:
            Tuple of (X_final, metadata)
            - X_final: Feature matrix with base + interaction terms
            - metadata: DataFrame with yyyymm, permno, excess_ret
        """
        print("\n--- Step 2: Generating Interaction Terms (In-Memory) ---")

        # Identify column types
        meta_cols = ['permno', 'yyyymm', 'ret_tplus1', 'excess_ret', 'Rfree', 'Price', 'Size']
        macro_predictors = [c for c in self.macro_df.columns if c not in ['yyyymm', 'Rfree']]
        firm_cols = [
            c for c in self.full_df.columns
            if c not in meta_cols and c not in self.macro_df.columns
        ]

        print(f"Base Firm Features: {len(firm_cols)}")
        print(f"Macro Predictors: {len(macro_predictors)} ({macro_predictors})")

        # Start with base firm features
        X_parts = [self.full_df[firm_cols]]

        # Create interaction terms for each macro predictor
        for macro in macro_predictors:
            print(f"   Interacting {len(firm_cols)} features with '{macro}'...")
            interaction_block = self.full_df[firm_cols].multiply(self.full_df[macro], axis=0)
            interaction_block.columns = [f"{col}_x_{macro}" for col in firm_cols]
            X_parts.append(interaction_block)

        # Concatenate all feature blocks
        print("Concatenating full feature matrix...")
        X_final = pd.concat(X_parts, axis=1)

        print(f"Final Feature Matrix Shape: {X_final.shape}")
        return X_final, self.full_df[['yyyymm', 'permno', 'excess_ret']]

    def create_interactions_chunked(
        self,
        output_dir: str,
        train_end: int = 200512,
        val_end: int = 201512,
        chunk_size: int = 100000
    ) -> Dict[str, str]:
        """
        Generate interactions and save splits in memory-efficient chunks.

        This method processes data in chunks to minimize memory usage.
        Instead of creating the full feature matrix in memory, it:
        1. Splits data by train/val/test first
        2. Processes each split in chunks
        3. Saves directly to parquet files

        Args:
            output_dir: Directory to save output files
            train_end: Training period end (YYYYMM)
            val_end: Validation period end (YYYYMM)
            chunk_size: Number of rows to process at once (default: 100000)

        Returns:
            Dictionary with paths to saved files

        Side Effects:
            Saves parquet files to output_dir
        """
        print("\n--- Memory-Efficient Chunked Processing ---")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Identify columns
        meta_cols = ['permno', 'yyyymm', 'ret_tplus1', 'excess_ret', 'Rfree', 'Price', 'Size']
        macro_predictors = [c for c in self.macro_df.columns if c not in ['yyyymm', 'Rfree']]
        firm_cols = [
            c for c in self.full_df.columns
            if c not in meta_cols and c not in self.macro_df.columns
        ]

        print(f"Base Firm Features: {len(firm_cols)}")
        print(f"Macro Predictors: {len(macro_predictors)} ({macro_predictors})")
        print(f"Total Features (with interactions): {len(firm_cols) * (1 + len(macro_predictors))}")
        print(f"Chunk size: {chunk_size:,} rows")

        # Create time-based masks
        metadata = self.full_df[['yyyymm', 'permno', 'excess_ret']].copy()
        train_mask = metadata['yyyymm'] <= train_end
        val_mask = (metadata['yyyymm'] > train_end) & (metadata['yyyymm'] <= val_end)
        test_mask = metadata['yyyymm'] > val_end

        splits = {
            'train': train_mask,
            'val': val_mask,
            'test': test_mask
        }

        # Process each split
        for split_name, mask in splits.items():
            print(f"\n[{split_name.upper()}] Processing {mask.sum():,} rows...")

            indices = np.where(mask)[0]
            n_chunks = int(np.ceil(len(indices) / chunk_size))

            # Initialize output files
            X_file = output_path / f"X_{split_name}.parquet"
            y_file = output_path / f"y_{split_name}.parquet"
            meta_file = output_path / f"meta_{split_name}.parquet"

            # Skip if no data
            if len(indices) == 0:
                print(f"  Warning: No data for {split_name} split, skipping...")
                continue

            # Create temporary directory for chunk files
            temp_dir = output_path / f"_temp_{split_name}"
            temp_dir.mkdir(exist_ok=True)

            # Track total rows for reporting
            total_rows = 0
            total_features = None

            # Process and save each chunk to separate file
            for i in range(n_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, len(indices))
                chunk_indices = indices[start_idx:end_idx]

                print(f"  Chunk {i+1}/{n_chunks}: rows {start_idx:,} to {end_idx:,}")

                # Get chunk data
                chunk_df = self.full_df.iloc[chunk_indices].copy()

                # Base firm features
                X_chunk_parts = [chunk_df[firm_cols]]

                # Add interactions
                for macro in macro_predictors:
                    interaction_block = chunk_df[firm_cols].multiply(chunk_df[macro], axis=0)
                    interaction_block.columns = [f"{col}_x_{macro}" for col in firm_cols]
                    X_chunk_parts.append(interaction_block)
                    del interaction_block
                    gc.collect()

                # Concatenate this chunk
                X_chunk = pd.concat(X_chunk_parts, axis=1)
                y_chunk = chunk_df['excess_ret'].to_frame(name='excess_ret')
                meta_chunk = chunk_df[['yyyymm', 'permno']]

                # Save chunk to temporary files
                X_chunk.to_parquet(temp_dir / f"X_chunk_{i:04d}.parquet", index=False)
                y_chunk.to_parquet(temp_dir / f"y_chunk_{i:04d}.parquet", index=False)
                meta_chunk.to_parquet(temp_dir / f"meta_chunk_{i:04d}.parquet", index=False)

                if total_features is None:
                    total_features = X_chunk.shape[1]
                total_rows += len(X_chunk)

                # Clean up this chunk from memory
                del X_chunk_parts, chunk_df, X_chunk, y_chunk, meta_chunk
                gc.collect()

            # Combine chunk files using pyarrow (memory-efficient)
            print(f"  Combining {n_chunks} chunk files...")
            import pyarrow.parquet as pq
            import pyarrow as pa

            # Combine X files
            X_tables = []
            for i in range(n_chunks):
                table = pq.read_table(temp_dir / f"X_chunk_{i:04d}.parquet")
                X_tables.append(table)
            X_combined_table = pa.concat_tables(X_tables)
            pq.write_table(X_combined_table, X_file, compression='snappy')
            del X_tables, X_combined_table
            gc.collect()

            # Combine y files
            y_tables = []
            for i in range(n_chunks):
                table = pq.read_table(temp_dir / f"y_chunk_{i:04d}.parquet")
                y_tables.append(table)
            y_combined_table = pa.concat_tables(y_tables)
            pq.write_table(y_combined_table, y_file, compression='snappy')
            del y_tables, y_combined_table
            gc.collect()

            # Combine meta files
            meta_tables = []
            for i in range(n_chunks):
                table = pq.read_table(temp_dir / f"meta_chunk_{i:04d}.parquet")
                meta_tables.append(table)
            meta_combined_table = pa.concat_tables(meta_tables)
            pq.write_table(meta_combined_table, meta_file, compression='snappy')
            del meta_tables, meta_combined_table
            gc.collect()

            # Clean up temporary directory
            import shutil
            shutil.rmtree(temp_dir)

            print(f"  Saved: X=({total_rows:,}, {total_features}), y={total_rows:,}")
            gc.collect()

        # Save metadata
        metadata.to_parquet(output_path / 'metadata.parquet', index=False)

        print("\n✓ Chunked processing complete!")

        # Build return dict with only existing files
        result = {'output_dir': str(output_path)}
        for split_name in ['train', 'val', 'test']:
            split_file = output_path / f'X_{split_name}.parquet'
            if split_file.exists():
                result[split_name] = str(split_file)

        return result

    def split_data(
        self,
        X: pd.DataFrame,
        metadata: pd.DataFrame,
        train_end: int = 200512,
        val_end: int = 201512
    ) -> Dict[str, pd.DataFrame]:
        """
        Split data by time into train/validation/test sets.

        Args:
            X: Feature matrix from create_interactions()
            metadata: Metadata from create_interactions()
            train_end: Training period end (YYYYMM format, default: 200512)
            val_end: Validation period end (YYYYMM format, default: 201512)

        Returns:
            Dictionary with keys: X_train, y_train, X_val, y_val, X_test, y_test, metadata
        """
        print("\n--- Step 3: Train/Val/Test Split ---")

        # Create time-based masks
        train_mask = metadata['yyyymm'] <= train_end
        val_mask = (metadata['yyyymm'] > train_end) & (metadata['yyyymm'] <= val_end)
        test_mask = metadata['yyyymm'] > val_end

        # Split features
        X_train = X[train_mask]
        X_val = X[val_mask]
        X_test = X[test_mask]

        # Split targets
        y_train = metadata.loc[train_mask, 'excess_ret']
        y_val = metadata.loc[val_mask, 'excess_ret']
        y_test = metadata.loc[test_mask, 'excess_ret']

        print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'metadata': metadata
        }
