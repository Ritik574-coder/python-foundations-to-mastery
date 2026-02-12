!pip install pandas numpy faker pyarrow tqdm

#!/usr/bin/env python3
"""
Generate 2 million rows of realistic e-commerce data in a single master table.
Output formats: Parquet (default) or CSV.
Run: pip install pandas numpy faker pyarrow tqdm
"""

import os
import uuid
import numpy as np
import pandas as pd
from faker import Faker
from tqdm import tqdm
import argparse
from datetime import datetime, timedelta

# ---------------------------- Configuration ----------------------------
TOTAL_ROWS = 2_000_000           # target number of rows (line items)
NUM_CUSTOMERS = 100_000          # unique customers
NUM_PRODUCTS = 5_000            # unique products
NUM_ORDERS = 500_000            # will produce ~ TOTAL_ROWS line items (avg 4 per order)
CHUNK_SIZE = 100_000           # rows per chunk when writing final output
ORDER_DATE_START = datetime(2021, 1, 1)
ORDER_DATE_END = datetime(2024, 12, 31)

# Output settings
OUTPUT_FILE = "ecommerce_master"
OUTPUT_FORMAT = "parquet"       # "parquet" or "csv"
# ----------------------------------------------------------------------

# Set random seed for reproducibility
np.random.seed(42)
Faker.seed(42)
fake = Faker()

# ---------------------------- Helper functions ----------------------------
def generate_customers(n):
    """Generate n unique customers with realistic attributes."""
    customers = []
    for i in tqdm(range(1, n + 1), desc="Generating customers"):
        first = fake.first_name()
        last = fake.last_name()
        email = f"{first.lower()}.{last.lower()}@{fake.free_email_domain()}"
        city = fake.city()
        state = fake.state_abbr()
        zip_code = fake.zipcode()
        reg_date = fake.date_between(start_date="-5y", end_date="today")
        customers.append({
            "customer_id": i,
            "first_name": first,
            "last_name": last,
            "customer_name": f"{first} {last}",
            "customer_email": email,
            "customer_city": city,
            "customer_state": state,
            "customer_zip": zip_code,
            "registration_date": reg_date,
        })
    return pd.DataFrame(customers)

def generate_products(n):
    """Generate n unique products with categories, brands, and prices."""
    categories = {
        "Electronics": ["Smartphone", "Laptop", "Tablet", "Headphones", "Camera"],
        "Clothing": ["T-Shirt", "Jeans", "Jacket", "Shoes", "Dress"],
        "Home": ["Cookware", "Bedding", "Furniture", "Decor", "Tools"],
        "Books": ["Fiction", "Non-Fiction", "Children", "Textbook", "Comics"],
        "Sports": ["Footwear", "Apparel", "Equipment", "Accessories", "Outdoors"]
    }
    brands = {
        "Electronics": ["Sony", "Samsung", "Apple", "LG", "Bose"],
        "Clothing": ["Nike", "Adidas", "Levi's", "Zara", "Gap"],
        "Home": ["IKEA", "KitchenAid", "Crate&Barrel", "Black+Decker", "Dyson"],
        "Books": ["Penguin", "HarperCollins", "Simon&Schuster", "Macmillan", "Hachette"],
        "Sports": ["Nike", "Adidas", "UnderArmour", "Wilson", "Spalding"]
    }

    products = []
    cat_list = list(categories.keys())
    cat_weights = [0.4, 0.25, 0.15, 0.12, 0.08]  # Electronics most frequent

    for i in tqdm(range(1, n + 1), desc="Generating products"):
        category = np.random.choice(cat_list, p=cat_weights)
        subcategory = np.random.choice(categories[category])
        brand = np.random.choice(brands[category])
        price = round(np.random.uniform(10, 800), 2)
        # premium products for electronics
        if category == "Electronics" and np.random.rand() > 0.7:
            price = round(np.random.uniform(800, 2000), 2)
        cost = round(price * np.random.uniform(0.45, 0.75), 2)
        product_name = f"{brand} {subcategory} {fake.word().capitalize()}"
        products.append({
            "product_id": i,
            "product_name": product_name,
            "category": category,
            "subcategory": subcategory,
            "brand": brand,
            "price": price,
            "cost": cost,
        })
    return pd.DataFrame(products)

def generate_orders(n, customers_df):
    """
    Generate n orders by sampling customers.
    Customer attributes are denormalised into the order row.
    """
    # Sample customers with replacement – some customers order more frequently
    customer_ids = customers_df["customer_id"].values
    # Skewed probabilities: customers with smaller id (older) order more often
    probs = 1.0 / (customer_ids ** 0.5)
    probs /= probs.sum()
    chosen_idx = np.random.choice(customer_ids, size=n, p=probs)
    orders = customers_df.iloc[chosen_idx - 1].copy()  # -1 because customer_id starts at 1
    orders.reset_index(drop=True, inplace=True)

    # Add order-specific fields
    order_ids = [f"ORD-{uuid.uuid4().hex[:8].upper()}" for _ in range(n)]
    order_dates = pd.date_range(ORDER_DATE_START, ORDER_DATE_END, periods=n).to_series()
    order_dates = order_dates.sample(frac=1, random_state=42).reset_index(drop=True)

    statuses = ["Completed", "Pending", "Cancelled", "Processing", "Shipped"]
    status_weights = [0.65, 0.15, 0.05, 0.1, 0.05]
    payment_methods = ["Credit Card", "Debit Card", "PayPal", "Gift Card", "Cash on Delivery"]
    payment_weights = [0.5, 0.2, 0.15, 0.1, 0.05]

    orders["order_id"] = order_ids
    orders["order_date"] = order_dates
    orders["order_status"] = np.random.choice(statuses, size=n, p=status_weights)
    orders["payment_method"] = np.random.choice(payment_methods, size=n, p=payment_weights)

    # Rename address columns to shipping_*
    orders.rename(columns={
        "customer_city": "shipping_city",
        "customer_state": "shipping_state",
        "customer_zip": "shipping_zip"
    }, inplace=True)

    # Keep only relevant columns for orders
    keep_cols = ["order_id", "customer_id", "customer_name", "customer_email",
                 "shipping_city", "shipping_state", "shipping_zip",
                 "order_date", "order_status", "payment_method"]
    orders = orders[keep_cols]
    return orders

def generate_line_items(orders_chunk, products_df):
    """
    Take a chunk of orders, generate 1–5 line items per order,
    assign random products (skewed popularity), and compute line totals.
    Returns a DataFrame of denormalised master rows.
    """
    # Number of line items per order
    num_items = np.random.choice([1, 2, 3, 4, 5], size=len(orders_chunk),
                                 p=[0.1, 0.2, 0.4, 0.2, 0.1])
    total_lines = num_items.sum()

    # Repeat each order row according to num_items
    order_indices = np.repeat(np.arange(len(orders_chunk)), num_items)
    line_df = orders_chunk.iloc[order_indices].reset_index(drop=True)

    # Add line item id within each order
    line_df["order_line_id"] = line_df.groupby("order_id").cumcount() + 1

    # Assign product_id with skewed popularity (zipf-like)
    product_ids = products_df["product_id"].values
    # popularity ~ rank^(-1.2)
    rank = np.arange(1, len(product_ids) + 1)
    probs = rank ** -1.2
    probs /= probs.sum()
    chosen_products = np.random.choice(product_ids, size=total_lines, p=probs)
    line_df["product_id"] = chosen_products

    # Merge product details
    line_df = line_df.merge(products_df, on="product_id", how="left")

    # Quantity and discount
    line_df["quantity"] = np.random.randint(1, 6, size=total_lines)
    discount_options = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    discount_probs = [0.5, 0.2, 0.1, 0.07, 0.05, 0.05, 0.03]
    line_df["discount"] = np.random.choice(discount_options, size=total_lines, p=discount_probs)

    # Compute line total
    line_df["line_total"] = line_df["price"] * line_df["quantity"] * (1 - line_df["discount"])
    line_df["line_total"] = line_df["line_total"].round(2)

    # Add a global row identifier
    line_df["row_id"] = [f"ROW-{uuid.uuid4().hex[:12].upper()}" for _ in range(total_lines)]

    # Reorder columns for readability
    cols = [
        "row_id", "order_id", "order_line_id", "order_date", "order_status", "payment_method",
        "customer_id", "customer_name", "customer_email",
        "shipping_city", "shipping_state", "shipping_zip",
        "product_id", "product_name", "category", "subcategory", "brand",
        "price", "quantity", "discount", "line_total"
    ]
    return line_df[cols]

# ---------------------------- Main execution ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate e-commerce master data")
    parser.add_argument("--format", choices=["parquet", "csv"], default=OUTPUT_FORMAT,
                        help="Output file format")
    parser.add_argument("--rows", type=int, default=TOTAL_ROWS,
                        help="Number of rows to generate")
    parser.add_argument("--output", type=str, default=OUTPUT_FILE,
                        help="Output file name (without extension)")

    # In Colab, the kernel passes args like -f, which argparse doesn't recognize.
    # We pass an empty list to parse_args to avoid this.
    args = parser.parse_args([])

    print(f"🚀 Generating {args.rows:,} rows of realistic e-commerce data...")
    print("   This may take a few minutes. Chunked processing keeps memory low.\n")

    # 1. Generate reference tables
    customers_df = generate_customers(NUM_CUSTOMERS)
    products_df = generate_products(NUM_PRODUCTS)

    # 2. Generate all orders
    print("\n📦 Generating orders...")
    orders_df = generate_orders(NUM_ORDERS, customers_df)
    print(f"   Generated {len(orders_df):,} orders.")

    # 3. Process orders in chunks and write final table
    output_path = f"{args.output}.{args.format}"
    if args.format == "csv":
        # Write header first
        header = True
        mode = "w"
    else:
        # For Parquet we will write in chunked mode using pyarrow
        header = None  # not used
        mode = "w"

    chunk_size_orders = 50_000  # process this many orders at a time
    total_orders = len(orders_df)
    num_chunks = (total_orders + chunk_size_orders - 1) // chunk_size_orders

    print(f"\n🖊️  Generating line items and writing to {output_path}...")
    with tqdm(total=args.rows, desc="Writing rows") as pbar:
        for i in range(num_chunks):
            start_idx = i * chunk_size_orders
            end_idx = min((i + 1) * chunk_size_orders, total_orders)
            orders_chunk = orders_df.iloc[start_idx:end_idx]

            # Generate line items for this chunk of orders
            chunk_lines = generate_line_items(orders_chunk, products_df)

            # Write to file
            if args.format == "csv":
                chunk_lines.to_csv(output_path, mode=mode, header=header, index=False)
                mode = "a"
                header = False
            else:  # parquet
                # For first chunk, write file; for subsequent, append
                if i == 0:
                    chunk_lines.to_parquet(output_path, index=False)
                else:
                    existing = pd.read_parquet(output_path)
                    combined = pd.concat([existing, chunk_lines], ignore_index=True)
                    combined.to_parquet(output_path, index=False)
                # Note: For very large files, better to use pyarrow's write_to_dataset
                # with row groups, but this is simple and works for 2M rows.

            pbar.update(len(chunk_lines))

    print(f"\n✅ Successfully generated {args.rows:,} rows in '{output_path}'.")
    print("\nSample of generated data:")
    sample = pd.read_parquet(output_path).head(3) if args.format == "parquet" else pd.read_csv(output_path, nrows=3)
    print(sample.to_string(index=False))

if __name__ == "__main__":
    main()
