import time
import pandas as pd
from bs4 import BeautifulSoup, Comment
from selenium import webdriver
from selenium.webdriver.edge.service import Service
from webdriver_manager.microsoft import EdgeChromiumDriverManager
from selenium.webdriver.edge.options import Options
from io import StringIO

# Function to convert age to decimal format
def convert_age_to_decimal(age_str):
    try:
        if pd.isna(age_str) or age_str == "N/A":
            return "N/a"
        age_str = str(age_str).strip()
        if "-" in age_str:  # Format "years-days"
            years, days = map(int, age_str.split("-"))
            return round(years + (days / 365), 2)
        if "." in age_str:  # Decimal format
            return round(float(age_str), 2)
        if age_str.isdigit():  # Whole years
            return round(float(age_str), 2)
        return "N/a"
    except (ValueError, AttributeError):
        return "N/a"

# Function to extract country code from Nation
def extract_country_code(nation_str):
    try:
        if pd.isna(nation_str) or nation_str == "N/A":
            return "N/a"
        return nation_str.split()[-1]  # Extract country code (e.g., "eng ENG" → "ENG")
    except (AttributeError, IndexError):
        return "N/a"

# Function to clean player names
def clean_player_name(name):
    try:
        if pd.isna(name) or name == "N/A":
            return "N/a"
        if "," in name:  # Handle comma-separated names
            parts = [part.strip() for part in name.split(",")]
            return " ".join(parts[::-1]) if len(parts) >= 2 else name
        return " ".join(name.split()).strip()  # Normalize spaces
    except (AttributeError, TypeError):
        return "N/a"

# Initialize Selenium WebDriver
options = Options()
options.add_argument("--headless")  # Run in headless mode
driver = webdriver.Edge(service=Service(EdgeChromiumDriverManager().install()), options=options)

# Define URLs and table IDs
urls = [
    "https://fbref.com/en/comps/9/2024-2025/stats/2024-2025-Premier-League-Stats",
    "https://fbref.com/en/comps/9/2024-2025/keepers/2024-2025-Premier-League-Stats",
    "https://fbref.com/en/comps/9/2024-2025/shooting/2024-2025-Premier-League-Stats",
    "https://fbref.com/en/comps/9/2024-2025/passing/2024-2025-Premier-League-Stats",
    "https://fbref.com/en/comps/9/2024-2025/gca/2024-2025-Premier-League-Stats",
    "https://fbref.com/en/comps/9/2024-2025/defense/2024-2025-Premier-League-Stats",
    "https://fbref.com/en/comps/9/2024-2025/possession/2024-2025-Premier-League-Stats",
    "https://fbref.com/en/comps/9/2024-2025/misc/2024-2025-Premier-League-Stats",
]
table_ids = [
    "stats_standard", "stats_keeper", "stats_shooting", "stats_passing",
    "stats_gca", "stats_defense", "stats_possession", "stats_misc"
]

# Define required columns (78 columns as specified)
required_columns = [
    "Player", "Nation", "Team", "Position", "Age",
    "Matches Played", "Starts", "Minutes",
    "Goals", "Assists", "Yellow cards", "Red cards",
    "xG", "xAG",
    "PrgC", "PrgP/Progression", "PrgR/Progression",
    "Gls/90", "Ast/90", "xG/90", "xAG/90",
    "GA90", "Save%", "CS%", "Penalty kicks Save%",
    "SoT%", "SoT/90", "G/Sh", "Dist",
    "Cmp", "Cmp%", "TotDist", "ShortCmp%", "MedCmp%", "LongCmp%", "KP", "Pass into 1/3", "PPA", "CrsPA", "PrgP/Passing",
    "SCA", "SCA90", "GCA", "GCA90",
    "Tkl", "TklW",
    "Deff Att", "Lost",
    "Blocks", "Sh", "Pass", "Int",
    "Touches", "Def Pen", "Def 3rd", "Mid 3rd", "Att 3rd", "Att Pen",
    "Take-Ons Att", "Succ%", "Tkld%",
    "Carries", "ProDist", "ProgC", "Carries 1/3", "CPA", "Mis", "Dis",
    "Rec", "PrgR/Possession",
    "Fls", "Fld", "Off", "Crs", "Recov",
    "Aerl won", "Aerl Lost", "Aerl Won%"
]

# Define column renaming dictionaries (updated for new column names)
column_rename_dict = {
    "stats_standard": {
        "Unnamed: 1": "Player", "Unnamed: 2": "Nation", "Unnamed: 3": "Position", "Unnamed: 4": "Team",
        "Unnamed: 5": "Age", "Playing Time": "Matches Played", "Playing Time.1": "Starts",
        "Playing Time.2": "Minutes", "Performance": "Goals", "Performance.1": "Assists",
        "Performance.6": "Yellow cards", "Performance.7": "Red cards", "Expected": "xG", "Expected.2": "xAG",
        "Progression": "PrgC", "Progression.1": "PrgP/Progression", "Progression.2": "PrgR/Progression",
        "Per 90 Minutes": "Gls/90", "Per 90 Minutes.1": "Ast/90", "Per 90 Minutes.4": "xG/90",
        "Per 90 Minutes.5": "xAG/90"
    },
    "stats_keeper": {
        "Unnamed: 1": "Player", "Performance.1": "GA90", "Performance.4": "Save%", "Performance.9": "CS%",
        "Penalty Kicks.4": "Penalty kicks Save%"
    },
    "stats_shooting": {
        "Unnamed: 1": "Player", "Standard.3": "SoT%", "Standard.5": "SoT/90", "Standard.6": "G/Sh",
        "Standard.8": "Dist"
    },
    "stats_passing": {
        "Unnamed: 1": "Player", "Total": "Cmp", "Total.2": "Cmp%", "Total.3": "TotDist",
        "Short.2": "ShortCmp%", "Medium.2": "MedCmp%", "Long.2": "LongCmp%", "Unnamed: 26": "KP",
        "Unnamed: 27": "Pass into 1/3", "Unnamed: 28": "PPA", "Unnamed: 29": "CrsPA",
        "Progression.1": "PrgP/Passing"
    },
    "stats_gca": {
        "Unnamed: 1": "Player", "SCA": "SCA", "SCA.1": "SCA90", "GCA": "GCA", "GCA.1": "GCA90"
    },
    "stats_defense": {
        "Unnamed: 1": "Player", "Tackles": "Tkl", "Tackles.1": "TklW", "Challenges.1": "Deff Att",
        "Challenges.2": "Lost", "Blocks": "Blocks", "Blocks.1": "Sh", "Blocks.2": "Pass", "Unnamed: 20": "Int"
    },
    "stats_possession": {
        "Unnamed: 1": "Player", "Touches": "Touches", "Touches.1": "Def Pen", "Touches.2": "Def 3rd",
        "Touches.3": "Mid 3rd", "Touches.4": "Att 3rd", "Touches.5": "Att Pen", "Take-Ons": "Take-Ons Att",
        "Take-Ons.2": "Succ%", "Take-Ons.4": "Tkld%", "Carries": "Carries", "Carries.2": "ProDist",
        "Carries": "ProgC", "Carries.4": "Carries 1/3", "Carries.5": "CPA", "Carries.6": "Mis", "Carries.7": "Dis",
        "Receiving": "Rec", "Receiving.1": "PrgR/Possession"
    },
    "stats_misc": {
        "Unnamed: 1": "Player", "Performance.3": "Fls", "Performance.4": "Fld", "Performance.5": "Off",
        "Performance.6": "Crs", "Performance.12": "Recov", "Aerial Duels": "Aerl won",
        "Aerial Duels.1": "Aerl Lost", "Aerial Duels.2": "Aerl Won%"
    }
}

# Initialize dictionary to store tables
all_tables = {}

# Scrape and process each table
for url, table_id in zip(urls, table_ids):
    print(f"Processing {table_id} from {url}")
    driver.get(url)
    time.sleep(3)
    soup = BeautifulSoup(driver.page_source, "html.parser")
    comments = soup.find_all(string=lambda text: isinstance(text, Comment))
    table = None
    for comment in comments:
        if table_id in comment:
            comment_soup = BeautifulSoup(comment, "html.parser")
            table = comment_soup.find("table", {"id": table_id})
            if table:
                break
    if not table:
        print(f"Table {table_id} not found!")
        continue
    try:
        df = pd.read_html(StringIO(str(table)), header=0)[0]
        df = df.rename(columns=column_rename_dict.get(table_id, {}))  # Rename columns
        df = df.loc[:, ~df.columns.duplicated()]  # Remove duplicates
        if "Player" in df.columns:
            df["Player"] = df["Player"].apply(clean_player_name)  # Clean player names
        if "Age" in df.columns:
            df["Age"] = df["Age"].apply(convert_age_to_decimal)  # Convert age
        all_tables[table_id] = df
    except Exception as e:
        print(f"Error reading table {table_id}: {e}")
        continue

# Merge tables
merged_df = None
for table_id, df in all_tables.items():
    df = df[[col for col in df.columns if col in required_columns]]  # Select required columns
    df = df.drop_duplicates(subset=["Player"])  # Remove duplicates
    if merged_df is None:
        merged_df = df
    else:
        merged_df = pd.merge(merged_df, df, on="Player", how="outer")

# Reorder columns
merged_df = merged_df.loc[:, [col for col in required_columns if col in merged_df.columns]]

# Filter players with >90 minutes
merged_df["Minutes"] = pd.to_numeric(merged_df["Minutes"], errors="coerce")
merged_df = merged_df[merged_df["Minutes"] > 90]

# Define column types
int_columns = ["Matches Played", "Starts", "Goals", "Assists", "Yellow cards", "Red cards", "PrgC", 
               "PrgP/Progression", "PrgR/Progression", "Cmp", "Tkl", "TklW", "Deff Att", "Lost", "Blocks", 
               "Sh", "Pass", "Int", "Touches", "Def Pen", "Def 3rd", "Mid 3rd", "Att 3rd", "Att Pen", 
               "Take-Ons Att", "Carries", "Carries 1/3", "CPA", "Mis", "Dis", "Rec", 
               "PrgR/Possession", "Fls", "Fld", "Off", "Crs", "Recov", "Aerl won", "Aerl Lost", "SCA", "GCA"]
float_columns = ["Age", "xG", "xAG", "Gls/90", "Ast/90", "xG/90", "xAG/90", "GA90", "Save%", "CS%", 
                 "Penalty kicks Save%", "SoT%", "SoT/90", "G/Sh", "Dist", "Cmp%", "ShortCmp%", "MedCmp%", 
                 "LongCmp%", "KP", "Pass into 1/3", "PPA", "CrsPA", "SCA90", "GCA90", "Succ%", "Tkld%", 
                 "ProDist", "Aerl Won%"]
string_columns = ["Player", "Nation", "Team", "Position"]

# Convert and fill missing values
for col in int_columns + float_columns:
    if col in merged_df.columns:
        merged_df[col] = pd.to_numeric(merged_df[col], errors="coerce").fillna("N/a")
for col in string_columns:
    if col in merged_df.columns:
        merged_df[col] = merged_df[col].fillna("N/a")
if "Nation" in merged_df.columns:
    merged_df["Nation"] = merged_df["Nation"].apply(extract_country_code)

# Sort players alphabetically by first name
merged_df['First_Name'] = merged_df['Player'].apply(lambda x: x.split()[0] if x != "N/a" else "N/a")
merged_df = merged_df.sort_values('First_Name')
merged_df = merged_df.drop('First_Name', axis=1)

# Save to CSV
merged_df.to_csv("results.csv", index=False, encoding="utf-8-sig")
print(f"Saved data to results.csv with {merged_df.shape[0]} rows and {merged_df.shape[1]} columns.")

# Close WebDriver
driver.quit()
