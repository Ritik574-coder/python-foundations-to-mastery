"""
International Trade Data Generator
Generates 2 million realistic trade transaction records for data engineering practice
Created for: Ritik - Aspiring Data Engineer

Features:
- Single master table with all trade transaction data
- 100% realistic international trade business data
- Exports to CSV or Parquet format
- Optimized for learning SQL, PySpark, Pandas, and Data Warehousing

Business Domain: International Trade (Import/Export, Customs, Logistics)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from faker import Faker
import warnings
warnings.filterwarnings('ignore')

# Initialize Faker for realistic data
fake = Faker()
Faker.seed(42)
np.random.seed(42)
random.seed(42)

print("=" * 80)
print("INTERNATIONAL TRADE DATA GENERATOR")
print("=" * 80)
print(f"Target Records: 2,000,000")
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

# ============================================================================
# MASTER DATA DEFINITIONS (Realistic International Trade Data)
# ============================================================================

# Major Trading Countries
COUNTRIES = {
    'USA': {'currency': 'USD', 'ports': ['Los Angeles', 'New York', 'Houston', 'Savannah', 'Seattle']},
    'China': {'currency': 'CNY', 'ports': ['Shanghai', 'Shenzhen', 'Ningbo', 'Guangzhou', 'Qingdao']},
    'Germany': {'currency': 'EUR', 'ports': ['Hamburg', 'Bremen', 'Wilhelmshaven']},
    'Japan': {'currency': 'JPY', 'ports': ['Tokyo', 'Yokohama', 'Osaka', 'Nagoya', 'Kobe']},
    'India': {'currency': 'INR', 'ports': ['Mumbai', 'Chennai', 'Kolkata', 'Nhava Sheva', 'Mundra']},
    'UK': {'currency': 'GBP', 'ports': ['Felixstowe', 'Southampton', 'London Gateway', 'Liverpool']},
    'Singapore': {'currency': 'SGD', 'ports': ['Singapore Port']},
    'South Korea': {'currency': 'KRW', 'ports': ['Busan', 'Incheon', 'Gwangyang']},
    'Netherlands': {'currency': 'EUR', 'ports': ['Rotterdam', 'Amsterdam']},
    'UAE': {'currency': 'AED', 'ports': ['Jebel Ali', 'Abu Dhabi', 'Sharjah']},
    'Brazil': {'currency': 'BRL', 'ports': ['Santos', 'Rio de Janeiro', 'Paranagua']},
    'Mexico': {'currency': 'MXN', 'ports': ['Manzanillo', 'Veracruz', 'Lazaro Cardenas']},
    'Canada': {'currency': 'CAD', 'ports': ['Vancouver', 'Montreal', 'Halifax']},
    'Australia': {'currency': 'AUD', 'ports': ['Sydney', 'Melbourne', 'Brisbane']},
    'Vietnam': {'currency': 'VND', 'ports': ['Ho Chi Minh', 'Haiphong', 'Da Nang']},
}

# Commodity Categories (HS Codes - Harmonized System)
COMMODITIES = {
    'Electronics': {
        'hs_codes': ['8471', '8517', '8528', '8542'],
        'description': ['Computers', 'Mobile Phones', 'LED Monitors', 'Semiconductors'],
        'avg_value_per_kg': [800, 1200, 300, 5000],
        'weight_range': (50, 5000)
    },
    'Machinery': {
        'hs_codes': ['8429', '8431', '8443', '8450'],
        'description': ['Bulldozers', 'Machine Parts', 'Printing Machinery', 'Washing Machines'],
        'avg_value_per_kg': [150, 200, 250, 180],
        'weight_range': (500, 20000)
    },
    'Textiles': {
        'hs_codes': ['6203', '6204', '6109', '6110'],
        'description': ['Men Suits', 'Women Dresses', 'T-Shirts', 'Sweaters'],
        'avg_value_per_kg': [50, 60, 30, 45],
        'weight_range': (100, 8000)
    },
    'Chemicals': {
        'hs_codes': ['2902', '2905', '2917', '3004'],
        'description': ['Petrochemicals', 'Industrial Alcohols', 'Polymers', 'Pharmaceuticals'],
        'avg_value_per_kg': [5, 8, 12, 500],
        'weight_range': (1000, 25000)
    },
    'Automotive': {
        'hs_codes': ['8703', '8708', '8706', '8711'],
        'description': ['Passenger Cars', 'Auto Parts', 'Chassis', 'Motorcycles'],
        'avg_value_per_kg': [200, 80, 150, 180],
        'weight_range': (800, 15000)
    },
    'Food Products': {
        'hs_codes': ['0901', '1001', '1006', '2009'],
        'description': ['Coffee', 'Wheat', 'Rice', 'Fruit Juices'],
        'avg_value_per_kg': [8, 0.5, 1.2, 3],
        'weight_range': (5000, 30000)
    },
    'Furniture': {
        'hs_codes': ['9403', '9401', '9404', '9405'],
        'description': ['Office Furniture', 'Seating', 'Bedding', 'Lamps'],
        'avg_value_per_kg': [40, 35, 25, 50],
        'weight_range': (500, 10000)
    },
    'Plastics': {
        'hs_codes': ['3920', '3923', '3926', '3917'],
        'description': ['Plastic Sheets', 'Containers', 'Plastic Articles', 'Pipes'],
        'avg_value_per_kg': [4, 3, 6, 5],
        'weight_range': (2000, 20000)
    }
}

# Shipping Methods
SHIPPING_MODES = {
    'Sea': {'base_cost_per_kg': 0.5, 'transit_days': (15, 45), 'reliability': 0.95},
    'Air': {'base_cost_per_kg': 8, 'transit_days': (2, 7), 'reliability': 0.98},
    'Rail': {'base_cost_per_kg': 1.5, 'transit_days': (10, 25), 'reliability': 0.92},
    'Road': {'base_cost_per_kg': 2, 'transit_days': (3, 15), 'reliability': 0.90}
}

# Container Types (for Sea freight)
CONTAINER_TYPES = ['20FT', '40FT', '40HC', 'Reefer 20', 'Reefer 40', 'Open Top', 'Flat Rack']

# Freight Forwarders
FREIGHT_FORWARDERS = [
    'DHL Global Forwarding', 'Kuehne + Nagel', 'DB Schenker', 'DSV Panalpina',
    'Expeditors', 'CEVA Logistics', 'Nippon Express', 'Hellmann Worldwide',
    'Agility Logistics', 'Bolloré Logistics', 'Geodis', 'Pantos Logistics',
    'Kerry Logistics', 'Yusen Logistics', 'CMA CGM', 'Maersk Logistics'
]

# Carriers
CARRIERS = {
    'Sea': ['Maersk', 'MSC', 'CMA CGM', 'COSCO', 'Hapag-Lloyd', 'ONE', 'Evergreen', 'Yang Ming'],
    'Air': ['Emirates SkyCargo', 'Lufthansa Cargo', 'Korean Air Cargo', 'Cathay Pacific Cargo',
            'Singapore Airlines Cargo', 'Qatar Airways Cargo', 'FedEx', 'UPS Airlines'],
    'Rail': ['DB Cargo', 'CRRC', 'Russian Railways', 'BNSF Railway'],
    'Road': ['J.B. Hunt', 'Schneider', 'Swift Transportation', 'Werner Enterprises']
}

# Customs Status
CUSTOMS_STATUS = ['Cleared', 'Cleared', 'Cleared', 'Cleared', 'Cleared', 'Cleared',  # 85% cleared
                  'Under Inspection', 'Pending Documentation', 'Detained', 'Released with Penalty']

# Payment Terms
PAYMENT_TERMS = ['FOB', 'CIF', 'CFR', 'EXW', 'DDP', 'DAP', 'FCA', 'CPT']

# Incoterms
INCOTERMS = ['EXW', 'FCA', 'CPT', 'CIP', 'DAP', 'DPU', 'DDP', 'FAS', 'FOB', 'CFR', 'CIF']

# Trade Types
TRADE_TYPES = ['Import', 'Export']

# Shipment Status
SHIPMENT_STATUS = [
    'Delivered', 'Delivered', 'Delivered', 'Delivered', 'Delivered',  # 60% delivered
    'In Transit', 'In Transit', 'In Transit',  # 30% in transit
    'At Origin', 'Customs Clearance', 'Delayed', 'Returned'  # 10% other
]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_trade_id(index):
    """Generate unique trade transaction ID"""
    return f"TRD{datetime.now().year}{str(index).zfill(8)}"

def generate_bill_of_lading():
    """Generate realistic Bill of Lading number"""
    prefix = random.choice(['MAEU', 'MSCU', 'CMDU', 'COSU', 'HLCU', 'ONEY', 'EVRG', 'YMLU'])
    return f"{prefix}{random.randint(100000000, 999999999)}"

def generate_container_number():
    """Generate ISO 6346 compliant container number"""
    owner_code = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=3))
    category = random.choice(['U', 'J', 'Z'])
    serial = random.randint(100000, 999999)
    # Calculate check digit (simplified)
    check_digit = random.randint(0, 9)
    return f"{owner_code}{category}{serial}{check_digit}"

def calculate_customs_duty(value, hs_code):
    """Calculate customs duty based on HS code"""
    # Simplified duty calculation (realistic ranges)
    duty_rates = {
        '84': 0.025,  # Machinery
        '85': 0.03,   # Electronics
        '62': 0.12,   # Textiles
        '29': 0.05,   # Chemicals
        '87': 0.025,  # Automotive
        '09': 0.08,   # Food
        '94': 0.06,   # Furniture
        '39': 0.045   # Plastics
    }
    rate = duty_rates.get(hs_code[:2], 0.05)
    return round(value * rate, 2)

def get_exchange_rate(currency):
    """Get realistic exchange rates to USD"""
    rates = {
        'USD': 1.0,
        'EUR': 1.08,
        'GBP': 1.26,
        'JPY': 0.0067,
        'CNY': 0.14,
        'INR': 0.012,
        'SGD': 0.74,
        'KRW': 0.00075,
        'AED': 0.27,
        'BRL': 0.20,
        'MXN': 0.058,
        'CAD': 0.72,
        'AUD': 0.65,
        'VND': 0.000040
    }
    return rates.get(currency, 1.0)

# ============================================================================
# MAIN DATA GENERATION FUNCTION
# ============================================================================

def generate_trade_data(num_records=2000000, batch_size=100000):
    """
    Generate realistic international trade data in batches

    Parameters:
    - num_records: Total number of records to generate (default: 2 million)
    - batch_size: Number of records per batch for memory efficiency
    """

    all_data = []
    num_batches = (num_records + batch_size - 1) // batch_size

    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, num_records)
        current_batch_size = end_idx - start_idx

        print(f"\nGenerating Batch {batch_num + 1}/{num_batches} ({current_batch_size:,} records)...")

        batch_data = []

        for i in range(start_idx, end_idx):
            # Random date in last 3 years
            shipment_date = fake.date_between(start_date='-3y', end_date='today')

            # Trade type
            trade_type = random.choice(TRADE_TYPES)

            # Select countries
            if trade_type == 'Export':
                origin_country = random.choice(list(COUNTRIES.keys()))
                destination_country = random.choice([c for c in COUNTRIES.keys() if c != origin_country])
            else:  # Import
                destination_country = random.choice(list(COUNTRIES.keys()))
                origin_country = random.choice([c for c in COUNTRIES.keys() if c != destination_country])

            # Ports
            origin_port = random.choice(COUNTRIES[origin_country]['ports'])
            destination_port = random.choice(COUNTRIES[destination_country]['ports'])

            # Commodity
            commodity_category = random.choice(list(COMMODITIES.keys()))
            commodity_data = COMMODITIES[commodity_category]

            idx = random.randint(0, len(commodity_data['hs_codes']) - 1)
            hs_code = commodity_data['hs_codes'][idx]
            product_description = commodity_data['description'][idx]
            value_per_kg = commodity_data['avg_value_per_kg'][idx]

            # Weight (in kg)
            weight_kg = round(random.uniform(*commodity_data['weight_range']), 2)

            # Quantity (units/packages)
            quantity = random.randint(1, 5000)

            # Shipping mode
            shipping_mode = random.choices(
                list(SHIPPING_MODES.keys()),
                weights=[60, 25, 10, 5],  # Sea is most common
                k=1
            )[0]

            mode_data = SHIPPING_MODES[shipping_mode]

            # Transit time
            transit_days = random.randint(*mode_data['transit_days'])
            estimated_delivery = shipment_date + timedelta(days=transit_days)

            # Actual delivery (add delays sometimes)
            if random.random() > mode_data['reliability']:
                delay_days = random.randint(1, 15)
                actual_delivery = estimated_delivery + timedelta(days=delay_days)
                is_delayed = True
            else:
                actual_delivery = estimated_delivery
                is_delayed = False

            # Status based on dates
            today = datetime.now().date()
            if actual_delivery < today:
                status = random.choice(['Delivered', 'Delivered', 'Delivered', 'Delivered', 'Returned'])
            elif shipment_date < today < estimated_delivery:
                status = random.choice(['In Transit', 'Customs Clearance'])
            else:
                status = 'At Origin'

            # Container info (for sea freight)
            if shipping_mode == 'Sea':
                container_number = generate_container_number()
                container_type = random.choice(CONTAINER_TYPES)
            else:
                container_number = None
                container_type = None

            # Bill of Lading
            bl_number = generate_bill_of_lading()

            # Carrier
            carrier = random.choice(CARRIERS[shipping_mode])

            # Freight forwarder
            freight_forwarder = random.choice(FREIGHT_FORWARDERS)

            # Value calculations
            fob_value = round(weight_kg * value_per_kg * random.uniform(0.85, 1.15), 2)

            # Freight cost
            base_freight = weight_kg * mode_data['base_cost_per_kg']
            freight_cost = round(base_freight * random.uniform(0.9, 1.3), 2)

            # Insurance (0.5% of FOB value)
            insurance_cost = round(fob_value * 0.005, 2)

            # Customs duty
            customs_duty = calculate_customs_duty(fob_value, hs_code)

            # Total landed cost
            total_value = fob_value + freight_cost + insurance_cost + customs_duty

            # Currency
            currency = COUNTRIES[origin_country]['currency']
            exchange_rate = get_exchange_rate(currency)
            value_usd = round(fob_value * exchange_rate, 2)

            # Customs status
            customs_status = random.choice(CUSTOMS_STATUS)

            # Payment term
            payment_term = random.choice(PAYMENT_TERMS)
            incoterm = random.choice(INCOTERMS)

            # Shipper and consignee
            shipper_name = fake.company()
            consignee_name = fake.company()

            # Record
            record = {
                'trade_id': generate_trade_id(i),
                'trade_type': trade_type,
                'shipment_date': shipment_date,
                'estimated_delivery_date': estimated_delivery,
                'actual_delivery_date': actual_delivery if status == 'Delivered' else None,
                'origin_country': origin_country,
                'destination_country': destination_country,
                'origin_port': origin_port,
                'destination_port': destination_port,
                'shipper_name': shipper_name,
                'consignee_name': consignee_name,
                'freight_forwarder': freight_forwarder,
                'carrier': carrier,
                'shipping_mode': shipping_mode,
                'container_number': container_number,
                'container_type': container_type,
                'bl_number': bl_number,
                'hs_code': hs_code,
                'commodity_category': commodity_category,
                'product_description': product_description,
                'quantity': quantity,
                'weight_kg': weight_kg,
                'fob_value': fob_value,
                'freight_cost': freight_cost,
                'insurance_cost': insurance_cost,
                'customs_duty': customs_duty,
                'total_landed_cost': total_value,
                'currency': currency,
                'exchange_rate_to_usd': exchange_rate,
                'value_usd': value_usd,
                'payment_term': payment_term,
                'incoterm': incoterm,
                'customs_status': customs_status,
                'shipment_status': status,
                'is_delayed': is_delayed,
                'transit_days_planned': transit_days,
                'transit_days_actual': (actual_delivery - shipment_date).days if status == 'Delivered' else None,
            }

            batch_data.append(record)

        all_data.extend(batch_data)
        print(f"✓ Batch {batch_num + 1} completed. Total records generated: {len(all_data):,}")

    return pd.DataFrame(all_data)

# ============================================================================
# EXECUTION
# ============================================================================

if __name__ == "__main__":

    # Generate data
    print("\n" + "=" * 80)
    print("STARTING DATA GENERATION")
    print("=" * 80)

    df = generate_trade_data(num_records=2000000, batch_size=100000)

    print("\n" + "=" * 80)
    print("DATA GENERATION COMPLETED!")
    print("=" * 80)
    print(f"Total Records: {len(df):,}")
    print(f"Total Columns: {len(df.columns)}")
    print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    # Display sample
    print("\n" + "=" * 80)
    print("SAMPLE DATA (First 5 Records)")
    print("=" * 80)
    print(df.head())

    # Data summary
    print("\n" + "=" * 80)
    print("DATA SUMMARY")
    print("=" * 80)
    print(df.info())

    print("\n" + "=" * 80)
    print("COLUMN DESCRIPTIONS")
    print("=" * 80)
    print(df.describe(include='all'))

    # Export options
    print("\n" + "=" * 80)
    print("EXPORT OPTIONS")
    print("=" * 80)
    print("1. CSV Format")
    print("2. Parquet Format")
    print("3. Both formats")

    choice = input("\nEnter your choice (1/2/3): ").strip()

    if choice in ['1', '3']:
        print("\nExporting to CSV...")
        csv_filename = 'international_trade_data_2M.csv'
        df.to_csv(csv_filename, index=False)
        print(f"✓ CSV file saved: {csv_filename}")

    if choice in ['2', '3']:
        print("\nExporting to Parquet...")
        parquet_filename = 'international_trade_data_2M.parquet'
        df.to_parquet(parquet_filename, index=False, compression='snappy')
        print(f"✓ Parquet file saved: {parquet_filename}")

    print("\n" + "=" * 80)
    print("PRACTICE SUGGESTIONS")
    print("=" * 80)
    print("""
    SQL Practice Ideas:
    1. Find total trade value by country
    2. Calculate average customs duty by commodity
    3. Identify delayed shipments and their carriers
    4. Analyze shipping mode preferences by route
    5. Track monthly trade volumes

    PySpark Practice Ideas:
    1. Aggregate freight costs by shipping mode
    2. Window functions for running totals by country
    3. Join operations (self-joins for route analysis)
    4. Partitioning data by year/month
    5. Calculate complex metrics (duty-to-value ratios)

    Pandas Practice Ideas:
    1. Time series analysis of trade volumes
    2. Groupby operations on multiple dimensions
    3. Pivot tables for cross-tabulation
    4. Data cleaning and validation
    5. Statistical analysis of delays

    Data Warehousing Practice Ideas:
    1. Design fact and dimension tables
    2. Create star/snowflake schema
    3. Build ETL pipelines
    4. Implement SCD Type 2 for changing data
    5. Create aggregated tables for reporting
    """)

    print("\n" + "=" * 80)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print("\nHappy Learning, Ritik! 🚀")
    print("Keep practicing and building your data engineering skills!")
    print("=" * 80)
