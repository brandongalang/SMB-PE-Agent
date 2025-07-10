"""
CSV processing script for Form ADV data.

This script reads the SEC Form ADV CSV files (Part A and Part B), joins them,
cleans the data, and populates the SQLAlchemy database with processed records.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging
import json
from tqdm import tqdm
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from adv_models import AdvFirm, DatabaseManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvDataProcessor:
    """Main class for processing Form ADV CSV data."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.state_columns = [
            '2-AL', '2-AK', '2-AZ', '2-AR', '2-CA', '2-CO', '2-CT', '2-DE', '2-DC',
            '2-FL', '2-GA', '2-HI', '2-ID', '2-IL', '2-IN', '2-IA', '2-KS', '2-KY',
            '2-LA', '2-ME', '2-MD', '2-MA', '2-MI', '2-MN', '2-MS', '2-MO', '2-MT',
            '2-NE', '2-NV', '2-NH', '2-NJ', '2-NM', '2-NY', '2-NC', '2-ND', '2-OH',
            '2-OK', '2-OR', '2-PA', '2-PR', '2-RI', '2-SC', '2-SD', '2-TN', '2-TX',
            '2-UT', '2-VT', '2-VA', '2-WA', '2-WV', '2-WI', '2-GU', '2-VI'
        ]
        
    def load_csv_files(self, part_a_path: str, part_b_path: str) -> pd.DataFrame:
        """Load and join the two CSV files on FilingID."""
        logger.info(f"Loading CSV files...")
        
        # Load Part A (core firm data)
        logger.info(f"Loading Part A from: {part_a_path}")
        try:
            df_a = pd.read_csv(part_a_path, low_memory=False, encoding='utf-8')
        except UnicodeDecodeError:
            logger.info("UTF-8 encoding failed, trying latin-1...")
            df_a = pd.read_csv(part_a_path, low_memory=False, encoding='latin-1')
        logger.info(f"Part A loaded: {len(df_a)} records, {len(df_a.columns)} columns")
        
        # Load Part B (business structure data)
        logger.info(f"Loading Part B from: {part_b_path}")
        try:
            df_b = pd.read_csv(part_b_path, low_memory=False, encoding='utf-8')
        except UnicodeDecodeError:
            logger.info("UTF-8 encoding failed, trying latin-1...")
            df_b = pd.read_csv(part_b_path, low_memory=False, encoding='latin-1')
        logger.info(f"Part B loaded: {len(df_b)} records, {len(df_b.columns)} columns")
        
        # Join on FilingID
        logger.info("Joining Part A and Part B on FilingID...")
        df_combined = pd.merge(df_a, df_b, on='FilingID', how='inner')
        logger.info(f"Combined dataset: {len(df_combined)} records, {len(df_combined.columns)} columns")
        
        return df_combined
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize the combined dataset."""
        logger.info("Starting data cleaning...")
        
        # Create a copy to avoid modifying original
        df_clean = df.copy()
        
        # Clean date fields
        df_clean['DateSubmitted'] = pd.to_datetime(df_clean['DateSubmitted'], errors='coerce')
        
        # Clean numeric fields
        numeric_fields = ['5A-Number', '5B1-Number', '5B2-Number', '5B3-Number', '5C-Number']
        for field in numeric_fields:
            if field in df_clean.columns:
                df_clean[field] = pd.to_numeric(df_clean[field], errors='coerce')
        
        # Clean asset fields (handle both string and numeric formats)
        asset_fields = ['5F2a', '5F2b', '5F2c', '5F2d', '5F2e', '5F2f']
        for field in asset_fields:
            if field in df_clean.columns:
                # Convert to numeric, handling commas and other formatting
                df_clean[field] = df_clean[field].astype(str).str.replace(',', '').str.replace('$', '')
                df_clean[field] = pd.to_numeric(df_clean[field], errors='coerce')
        
        # Clean boolean fields (Y/N to True/False)
        boolean_fields = ['1M', '1N', '5E1', '5E2', '5E3', '5E4', '5E5', '5E6', '5E7',
                         '5D1/5D1a', '5D2/5D1b', '5D3/5D1c', '5D4/5D1d'] + self.state_columns
        for field in boolean_fields:
            if field in df_clean.columns:
                df_clean[field] = df_clean[field].map({'Y': True, 'N': False})
        
        # Clean text fields (remove extra whitespace, handle nulls)
        text_fields = ['1A', '1B', '1C-Legal', '1C-Business', '1C-New Name', '1D', '1E',
                      '1F1-Street 1', '1F1-Street 2', '1F1-City', '1F1-State', '1F1-Country',
                      '1F1-Postal', '1J-Name', '1J-Title', '1J-Phone', '1J-Email',
                      '3A', '3A-Other', '3C-State', '3C-Country']
        for field in text_fields:
            if field in df_clean.columns:
                df_clean[field] = df_clean[field].astype(str).str.strip()
                df_clean[field] = df_clean[field].replace('nan', np.nan)
        
        # Remove completely empty rows
        df_clean = df_clean.dropna(how='all')
        
        # Remove duplicates based on FilingID
        df_clean = df_clean.drop_duplicates(subset=['FilingID'])
        
        logger.info(f"Data cleaning complete. {len(df_clean)} records remaining.")
        
        return df_clean
        
    def extract_registered_states(self, row: pd.Series) -> Optional[str]:
        """Extract list of registered states from state columns."""
        registered_states = []
        
        state_mapping = {
            '2-AL': 'Alabama', '2-AK': 'Alaska', '2-AZ': 'Arizona', '2-AR': 'Arkansas',
            '2-CA': 'California', '2-CO': 'Colorado', '2-CT': 'Connecticut', '2-DE': 'Delaware',
            '2-DC': 'District of Columbia', '2-FL': 'Florida', '2-GA': 'Georgia', '2-HI': 'Hawaii',
            '2-ID': 'Idaho', '2-IL': 'Illinois', '2-IN': 'Indiana', '2-IA': 'Iowa',
            '2-KS': 'Kansas', '2-KY': 'Kentucky', '2-LA': 'Louisiana', '2-ME': 'Maine',
            '2-MD': 'Maryland', '2-MA': 'Massachusetts', '2-MI': 'Michigan', '2-MN': 'Minnesota',
            '2-MS': 'Mississippi', '2-MO': 'Missouri', '2-MT': 'Montana', '2-NE': 'Nebraska',
            '2-NV': 'Nevada', '2-NH': 'New Hampshire', '2-NJ': 'New Jersey', '2-NM': 'New Mexico',
            '2-NY': 'New York', '2-NC': 'North Carolina', '2-ND': 'North Dakota', '2-OH': 'Ohio',
            '2-OK': 'Oklahoma', '2-OR': 'Oregon', '2-PA': 'Pennsylvania', '2-PR': 'Puerto Rico',
            '2-RI': 'Rhode Island', '2-SC': 'South Carolina', '2-SD': 'South Dakota',
            '2-TN': 'Tennessee', '2-TX': 'Texas', '2-UT': 'Utah', '2-VT': 'Vermont',
            '2-VA': 'Virginia', '2-WA': 'Washington', '2-WV': 'West Virginia', '2-WI': 'Wisconsin',
            '2-GU': 'Guam', '2-VI': 'Virgin Islands'
        }
        
        for col, state_name in state_mapping.items():
            if col in row.index and row[col] == True:
                registered_states.append(state_name)
        
        return json.dumps(registered_states) if registered_states else None
        
    def convert_to_adv_firm(self, row: pd.Series) -> AdvFirm:
        """Convert a pandas row to an AdvFirm model instance."""
        
        # Helper function to safely get values
        def get_val(field, default=None):
            val = row.get(field, default)
            if pd.isna(val):
                return default
            return val
            
        # Create the AdvFirm instance
        firm = AdvFirm(
            filing_id=str(get_val('FilingID')),
            firm_name=get_val('1A'),
            business_name=get_val('1B'),
            legal_name=get_val('1C-Legal'),
            new_name=get_val('1C-New Name'),
            
            crd_number=get_val('1D'),
            cik_number=get_val('1N-CIK'),
            
            form_version=get_val('FormVersion'),
            date_submitted=get_val('DateSubmitted'),
            
            # Address fields
            address_street1=get_val('1F1-Street 1'),
            address_street2=get_val('1F1-Street 2'),
            address_city=get_val('1F1-City'),
            address_state=get_val('1F1-State'),
            address_country=get_val('1F1-Country'),
            address_postal=get_val('1F1-Postal'),
            
            # Contact fields
            contact_name=get_val('1J-Name'),
            contact_title=get_val('1J-Title'),
            contact_phone=get_val('1J-Phone'),
            contact_fax=get_val('1J-Fax'),
            contact_email=get_val('1J-Email'),
            
            # Business info
            business_hours=get_val('1F2-Hours'),
            phone_main=get_val('1F3'),
            fax_main=get_val('1F4'),
            
            # Employee info
            employee_range=get_val('5A-Range'),
            employee_number=get_val('5A-Number'),
            
            # Assets
            total_assets=get_val('5F2a'),
            discretionary_assets=get_val('5F2b'),
            non_discretionary_assets=get_val('5F2c'),
            
            # Clients
            client_range=get_val('5C-Range'),
            client_number=get_val('5C-Number'),
            
            # Business structure
            legal_structure=get_val('3A'),
            legal_structure_other=get_val('3A-Other'),
            fiscal_year_end=get_val('3B'),
            incorporation_state=get_val('3C-State'),
            incorporation_country=get_val('3C-Country'),
            
            # Services
            provides_investment_advice=get_val('5D1/5D1a'),
            provides_financial_planning=get_val('5D2/5D1b'),
            provides_pension_consulting=get_val('5D3/5D1c'),
            provides_selection_services=get_val('5D4/5D1d'),
            
            # Registration
            registered_states=self.extract_registered_states(row),
            
            # Investment strategies
            strategy_equity=get_val('5E1'),
            strategy_fixed_income=get_val('5E2'),
            strategy_commodity=get_val('5E3'),
            strategy_mutual_funds=get_val('5E4'),
            strategy_hedge_funds=get_val('5E5'),
            strategy_private_funds=get_val('5E6'),
            strategy_other=get_val('5E7'),
            
            # Registration status
            sec_registered=get_val('1M'),
            state_registered=get_val('1N'),
        )
        
        # Generate searchable text
        firm.generate_searchable_text()
        
        return firm
        
    def process_and_save(self, df: pd.DataFrame, batch_size: int = 1000) -> Tuple[int, int]:
        """Process the dataframe and save to database in batches."""
        logger.info("Starting database population...")
        
        total_records = len(df)
        processed_count = 0
        error_count = 0
        
        # Process in batches
        for start_idx in tqdm(range(0, total_records, batch_size), desc="Processing batches"):
            end_idx = min(start_idx + batch_size, total_records)
            batch_df = df.iloc[start_idx:end_idx]
            
            session = self.db_manager.get_session()
            try:
                batch_firms = []
                for _, row in batch_df.iterrows():
                    try:
                        firm = self.convert_to_adv_firm(row)
                        batch_firms.append(firm)
                    except Exception as e:
                        logger.warning(f"Error converting row {row.get('FilingID', 'unknown')}: {e}")
                        error_count += 1
                        continue
                
                # Bulk insert
                if batch_firms:
                    session.bulk_save_objects(batch_firms)
                    session.commit()
                    processed_count += len(batch_firms)
                    
            except Exception as e:
                logger.error(f"Error processing batch {start_idx}-{end_idx}: {e}")
                session.rollback()
                error_count += len(batch_df)
            finally:
                session.close()
        
        logger.info(f"Processing complete. {processed_count} records saved, {error_count} errors.")
        return processed_count, error_count
        
    def run_full_process(self, part_a_path: str, part_b_path: str, batch_size: int = 1000) -> Dict:
        """Run the complete processing pipeline."""
        start_time = datetime.now()
        
        logger.info("Starting full ADV data processing pipeline...")
        
        # Step 1: Load data
        df_combined = self.load_csv_files(part_a_path, part_b_path)
        
        # Step 2: Clean data
        df_clean = self.clean_data(df_combined)
        
        # Step 3: Create database tables
        logger.info("Creating database tables...")
        self.db_manager.create_tables()
        
        # Step 4: Process and save
        processed_count, error_count = self.process_and_save(df_clean, batch_size)
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        results = {
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration.total_seconds(),
            'total_input_records': len(df_combined),
            'cleaned_records': len(df_clean),
            'processed_successfully': processed_count,
            'errors': error_count,
            'success_rate': processed_count / len(df_clean) * 100 if len(df_clean) > 0 else 0
        }
        
        logger.info(f"Processing pipeline complete!")
        logger.info(f"Results: {results}")
        
        return results


def main():
    """Main function to run the ADV data processing."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process Form ADV CSV files')
    parser.add_argument('--part-a', required=True, help='Path to ADV Part A CSV file')
    parser.add_argument('--part-b', required=True, help='Path to ADV Part B CSV file')
    parser.add_argument('--db-url', default='sqlite:///adv_database.db', help='Database URL')
    parser.add_argument('--batch-size', type=int, default=1000, help='Batch size for processing')
    parser.add_argument('--drop-tables', action='store_true', help='Drop existing tables first')
    
    args = parser.parse_args()
    
    # Initialize database manager
    db_manager = DatabaseManager(args.db_url)
    
    # Drop tables if requested
    if args.drop_tables:
        logger.info("Dropping existing tables...")
        db_manager.drop_tables()
    
    # Initialize processor
    processor = AdvDataProcessor(db_manager)
    
    # Run the process
    results = processor.run_full_process(args.part_a, args.part_b, args.batch_size)
    
    # Save results to file
    with open('adv_processing_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Processing complete! Results saved to adv_processing_results.json")
    print(f"Successfully processed: {results['processed_successfully']} records")
    print(f"Errors: {results['errors']} records")
    print(f"Success rate: {results['success_rate']:.2f}%")


if __name__ == "__main__":
    main()