# %%
!pip install --force-reinstall "numpy<2"

# %%
!which python  # macOS/Linux

# %%
from openai import OpenAI
import json
import pandas as pd
from textblob import TextBlob
import time
from dotenv import load_dotenv
import os
from pathlib import Path

# %%
os.environ["OPENAI_API_KEY"] = "ENTER IT HERE"

# %%
## print(os.getenv("OPENAI_API_KEY")) 

# %%
# Load API Key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Missing OpenAI API key. Set it using the environment variable 'OPENAI_API_KEY'.")
client = OpenAI(api_key=api_key)

# Define domains and subdomains
domains = {
    "Consumer Goods and Retails": [
        "Electronics", "Fashion & Apparel", "Beauty & Skincare", "Home Appliances",
        "Furniture & Home Decor", "Food & Beverages", "Fast Food Chains",
        "Organic & Health Foods", "Coffee & Tea Brands", "Alcoholic Beverages",
        "Energy Drinks & Soft Drinks"
    ],
    "Technology & Software": [
        "Smartphones & Accessories", "Laptops & Workstations", "Cloud Services",
        "AI & Chatbots", "Cybersecurity & VPN Services"
    ],
    "Finance & Investment": [
        "Stock Trading Platforms", "Cryptocurrencies & Exchanges", "Banking & Credit Cards",
        "Personal Finance Management", "Business Loans & Funding"
    ],
    "Education & Learning": [
        "Online Courses", "Test Prep & Tutoring", "Language Learning Apps",
        "College & University Programs", "AI & Programming Bootcamps"
    ],
    "Entertainment & Media": [
        "Streaming Services", "Gaming Platforms", "Music Streaming",
        "Books & Audiobooks", "Live Events & Concerts"
    ],
    "Travel & Transportation": [
        "Airlines & Flights", "Hotels & Accommodation", "Car Rentals & Ridesharing",
        "Travel Booking Websites", "Luxury Cruises & Tours"
    ],
    "Health & Wellness": [
        "Gym Memberships & Fitness Apps", "Mental Health & Therapy Services",
        "Dietary Supplements & Vitamins", "Telemedicine & Online Doctors",
        "Health Insurance Plans"
    ],
    "Automotive & Transportation": [
        "Car Dealerships", "Auto Insurance", "Auto Parts & Accessories",
        "Electric Vehicles", "Motorcycle & Bicycle Brands"
    ],
    "Business & Productivity": [
        "Project Management Software", "Email & Communication Tools",
        "HR & Hiring Solutions", "Office Supplies & Workstations", "AI Productivity Tools"
    ]
}

# Load previously saved checkpoints if they exist
if Path("product_names_by_domain.json").exists():
    with open("product_names_by_domain.json") as f:
        products_by_domain = json.load(f)
else:
    products_by_domain = {}

if Path("global_generated_products.json").exists():
    with open("global_generated_products.json") as f:
        global_generated_products = json.load(f)
else:
    global_generated_products = []

if Path("checkpoint_ads.csv").exists():
    df_ads = pd.read_csv("checkpoint_ads.csv")
    ad_data = df_ads.to_dict(orient="records")
    ad_id = df_ads["ad_id"].max() + 1
else:
    ad_data = []
    ad_id = 1

# === Function to generate product names ===
def generate_product_names(domain, subdomain, count, already_generated):
    prompt = f"""
    Generate a JSON array of {count} unique and creative product names for fictional advertisements in the '{subdomain}' subdomain under the domain '{domain}'.
    Do not include any extra commentary.
    Avoid any names that might duplicate the following: {", ".join(already_generated)}.
    The output should be in JSON array format.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a creative assistant that outputs clean JSON arrays."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        response_text = response.choices[0].message.content.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:-3].strip()
        return json.loads(response_text)
    except Exception as e:
        print(f"Error generating product names for {domain} - {subdomain}: {e}")
        return []

# === Function to generate ad details ===
def generate_ad_details(domain, subdomain, product, all_generated):
    prompt = f"""
    Generate an advertisement for the product "{product}" in the '{subdomain}' subdomain under the domain '{domain}'.
    Do not generate duplicate products. Previously generated products: {", ".join(all_generated)}.
    Respond as a JSON object with the following fields:
    - product: name of the product
    - ad_key_words: list of 4-5 keywords
    - ad_description: 1 sentence description of the product
    - ad_benefits: list of 3-5 short bullet points

    Do not add any extra commentary. Just the raw JSON.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful ad generator that outputs clean JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        response_text = response.choices[0].message.content.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:-3].strip()
        return json.loads(response_text)
    except Exception as e:
        print(f"Error generating ad details for product '{product}' in {domain} - {subdomain}: {e}")
        return None

# === Phase 1: Generate or top up product names ===
for domain, subdomains in domains.items():
    if domain not in products_by_domain:
        products_by_domain[domain] = {}
    for subdomain in subdomains:
        existing = products_by_domain[domain].get(subdomain, [])
        needed = 100 - len(existing)
        if needed > 0:
            print(f"Generating {needed} more product names for: {domain} > {subdomain}")
            new_names = generate_product_names(domain, subdomain, needed, global_generated_products)
            if isinstance(new_names, list):
                products_by_domain[domain][subdomain] = existing + new_names
                global_generated_products.extend(new_names)
            time.sleep(1)

# Save Phase 1 results
with open("product_names_by_domain.json", "w") as f:
    json.dump(products_by_domain, f, indent=2)

with open("global_generated_products.json", "w") as f:
    json.dump(global_generated_products, f)

total_products = sum(len(p) for sub in products_by_domain.values() for p in sub.values())
print(f"Total unique products available: {total_products}")
print(f"Ads already generated: {len(ad_data)}")


# === Phase 2: Generate Ad Data ===
for domain, subdomains in products_by_domain.items():
    for subdomain, product_names in subdomains.items():
        for product in product_names:
            # Skip if already in ad_data
            if any(ad["product"] == product for ad in ad_data):
                continue
            ad = generate_ad_details(domain, subdomain, product, global_generated_products)
            if ad:
                ad_data.append({
                    "ad_id": ad_id,
                    "domain": domain,
                    "category": subdomain,
                    "product": ad["product"],
                    "ad_key_words": ad["ad_key_words"],
                    "ad_description": ad["ad_description"],
                    "ad_benefits": ad["ad_benefits"]
                })
                ad_id += 1

                # Save checkpoint every 10 ads
                if ad_id % 10 == 0:
                    pd.DataFrame(ad_data).to_csv("checkpoint_ads.csv", index=False)
                    print(f"Checkpoint saved after ad_id {ad_id}")
            time.sleep(1)

# Final save
df_ads = pd.DataFrame(ad_data)
output_path = Path.cwd() / "generated_ad_dataset.csv"
df_ads.to_csv(output_path, index=False)
print(f"Final ad dataset saved to: {output_path}")

# %%



