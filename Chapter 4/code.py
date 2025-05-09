import pandas as pd
import numpy as np
import asyncio
import logging
import random
import time
from functools import wraps
from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.microsoft import EdgeChromiumDriverManager
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Configuration
CONFIG = {
    'input_csv': 'results.csv',
    'transfer_csv': 'transfer_values.csv',
    'predict_csv': 'transfer_predictions.csv',
    'transfermarkt_url': 'https://www.transfermarkt.com/schnellsuche/ergebnis/schnellsuche?query=',
    'footballtransfers_url': 'https://www.footballtransfers.com/en/search?q=',
    'features': [
        'Age', 'Minutes', 'Goals', 'Assists', 'xG', 'xAG', 'PrgC', 'PrgP/Progression',
        'PrgR/Progression', 'SoT%', 'SoT/90', 'Tkl', 'TklW', 'Blocks', 'Touches',
        'Succ%', 'Fls', 'Fld', 'Aerl Won%'
    ],
    'headless': True,
    'max_attempts': 2,
    'wait_timeout': 15,
    'page_timeout': 50
}

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

def log_errors(func):
    """Decorator to log errors for methods."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            raise
    return wrapper

class TransferScraper:
    """Class to handle data loading, scraping, and modeling."""
    
    def __init__(self, config: dict):
        self.config = config
        self.df = None
        self.players_df = None
        self.transfer_data = None

    def load_data(self):
        """Load and filter CSV data."""
        try:
            self.df = pd.read_csv(self.config['input_csv'], encoding='utf-8-sig')
            required_cols = ['Player', 'Minutes']
            team_col = next((col for col in ['Team', 'Squad', 'team', 'TEAM'] if col in self.df.columns), None)
            if not all(col in self.df.columns for col in required_cols) or not team_col:
                raise ValueError(f"Missing columns: {self.df.columns.tolist()}")

            self.df['Minutes'] = pd.to_numeric(self.df['Minutes'].astype(str).str.replace(',', ''), errors='coerce')
            self.df = self.df.dropna(subset=['Minutes'])
            self.players_df = self.df[self.df['Minutes'] > 900][['Player', team_col, 'Minutes']].rename(columns={team_col: 'Team'})
            logger.info(f"Loaded {len(self.players_df)} players with Minutes > 900")
        except FileNotFoundError:
            logger.error(f"'{self.config['input_csv']}' not found")
            raise
        except Exception as e:
            logger.error(f"CSV load failed: {e}")
            raise

    def setup_driver(self) -> webdriver.Edge:
        """Configure Edge WebDriver."""
        options = Options()
        if self.config['headless']:
            options.add_argument("--headless=new")
        options.add_argument("--ignore-certificate-errors")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) Edge/91.0.4472.124")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--allow-insecure-localhost")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.accept_insecure_certs = True
        driver = webdriver.Edge(service=Service(EdgeChromiumDriverManager().install()), options=options)
        driver.set_page_load_timeout(self.config['page_timeout'])
        return driver

    @log_errors
    async def scrape_player(self, player: str) -> dict:
        """Scrape transfer value for a player."""
        async with asyncio.Lock():  # Ensure thread-safe driver usage
            with self.setup_driver() as driver:
                value = None
                # Try Transfermarkt
                for attempt in range(self.config['max_attempts']):
                    try:
                        logger.info(f"Scraping {player} (Transfermarkt, attempt {attempt + 1})")
                        driver.get(f"{self.config['transfermarkt_url']}{player.replace(' ', '+')}")
                        link = WebDriverWait(driver, self.config['wait_timeout']).until(
                            EC.element_to_be_clickable((By.CSS_SELECTOR, "td.hauptlink a[href*='/profil/spieler/']"))
                        )
                        driver.get(link.get_attribute('href'))
                        value = WebDriverWait(driver, self.config['wait_timeout']).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, "a[href*='/marktwertverlauf/spieler/']"))
                        ).text
                        logger.info(f"Got {player}: {value} (Transfermarkt)")
                        return {'player_name': player, 'transfer_value': value}
                    except:
                        await asyncio.sleep(random.uniform(2, 4))

                # Fallback to FootballTransfers
                for attempt in range(self.config['max_attempts']):
                    try:
                        logger.info(f"Scraping {player} (FootballTransfers, attempt {attempt + 1})")
                        driver.get(f"{self.config['footballtransfers_url']}{player.replace(' ', '+')}")
                        value = WebDriverWait(driver, self.config['wait_timeout']).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, "div.market-value"))
                        ).text
                        logger.info(f"Got {player}: {value} (FootballTransfers)")
                        return {'player_name': player, 'transfer_value': value}
                    except:
                        await asyncio.sleep(random.uniform(2, 4))

                logger.warning(f"No value for {player}")
                return {'player_name': player, 'transfer_value': 'N/a'}

    async def scrape_all(self):
        """Scrape transfer values for all players."""
        tasks = [self.scrape_player(player) for player in self.players_df['Player'].head(5)]  # Limit to 5 for testing
        results = await asyncio.gather(*tasks, return_exceptions=True)
        self.transfer_data = pd.DataFrame([r for r in results if isinstance(r, dict)])
        self.transfer_data.to_csv(self.config['transfer_csv'], index=False, encoding='utf-8-sig')
        logger.info(f"Saved transfer data to {self.config['transfer_csv']}")

    def clean_value(self, value: str) -> float:
        """Convert transfer value to numeric."""
        if value == 'N/a' or pd.isna(value):
            return np.nan
        try:
            value = value.replace('€', '').replace('£', '').strip().lower()
            if 'm' in value:
                return float(value.replace('m', '')) * 1e6
            if 'k' in value:
                return float(value.replace('k', '')) * 1e3
            return float(value)
        except:
            return np.nan

    def prepare_data(self):
        """Merge and clean data."""
        merged_data = pd.merge(self.players_df, self.transfer_data, left_on='Player', right_on='player_name', how='left')
        merged_data['transfer_value'] = merged_data['transfer_value'].apply(self.clean_value)
        self.df['transfer_value'] = merged_data['transfer_value']

    def get_numeric_features(self) -> list:
        """Identify numeric features for modeling."""
        numeric_cols = []
        for col in self.config['features']:
            if col in self.df.columns:
                try:
                    self.df[col] = pd.to_numeric(self.df[col].astype(str).str.replace(',', ''), errors='coerce')
                    if self.df[col].notna().sum() > 0:
                        numeric_cols.append(col)
                except:
                    logger.debug(f"Cannot convert {col} to numeric")
        return numeric_cols

    def train_model(self):
        """Train RandomForest model and predict."""
        logger.warning("Small dataset may lead to unreliable predictions")
        numeric_features = self.get_numeric_features()
        if not numeric_features:
            logger.error("No valid features for modeling")
            return

        valid_df = self.df.dropna(subset=numeric_features + ['transfer_value'])
        if valid_df.empty:
            logger.error("No valid data after dropping NaNs")
            return

        X = valid_df[numeric_features].astype(float)
        y = valid_df['transfer_value']
        players = valid_df['Player']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        logger.info(f"Model - MSE: {mse:.2f}, R^2: {r2:.2f}")

        full_pred = model.predict(X_scaled)
        predictions = pd.DataFrame({
            'Player': players.values,
            'Actual_Value': y.values,
            'Predicted_Value': full_pred
        })
        predictions.to_csv(self.config['predict_csv'], index=False, encoding='utf-8-sig')
        logger.info(f"Saved predictions to {self.config['predict_csv']}")

    async def run(self):
        """Execute the full pipeline."""
        self.load_data()
        await self.scrape_all()
        self.prepare_data()
        self.train_model()

def main():
    """Main entry point."""
    scraper = TransferScraper(CONFIG)
    asyncio.run(scraper.run())

if __name__ == "__main__":
    main()
