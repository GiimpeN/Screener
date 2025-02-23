import logging
import yfinance as yf
import pandas as pd
import numpy as np
import time
import os
import signal
import sys
import threading
import subprocess
from math import log, sqrt
from datetime import datetime, timedelta
from scipy.stats import norm
from yfinance.exceptions import YFRateLimitError
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from requests.exceptions import ReadTimeout, ConnectionError
import json
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from webdriver_manager.chrome import ChromeDriverManager

# ============================
# Logging Configuration


# ============================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# ============================
# Configuration Parameters
# ============================
MIN_PRICE = 1.0                    # Minimum stock price
MIN_DTE = 20                       # Minimum days to expiration
MAX_DTE = 45                       # Maximum days to expiration
TARGET_DTE = 30                    # Target DTE (reference)

DELTA_LOWER = -0.40                # Adjusted delta range for sold puts
DELTA_UPPER = -0.25
TARGET_DELTA = -0.30               # Target delta for sold put

MIN_LIQ_SCORE = 1.0                # Minimum liquidity score
MAX_TICKERS_TO_PROCESS = 1000      # Maximum tickers to screen
REQUEST_DELAY = 10                  # Delay after each yfinance request (seconds)
MAX_WORKERS = 10                    # Number of workers
RISK_FREE_RATE = 0.05              # Risk-free rate (5%)
TIMEOUT = 60                       # Increased timeout for API requests (seconds)

# Stricter fundamental thresholds
MIN_EARNINGS_GROWTH = 0.05         # Earnings growth > 5%
MAX_DEBT_TO_EQUITY = 0.8           # Debt-to-equity < 0.8
MIN_ROE = 15                       # ROE > 15%

# Liquidity filters
MIN_OPEN_INTEREST = 50             # Minimum open interest
MAX_BID_ASK_SPREAD_RATIO = 0.05    # Max bid/ask spread ratio (5%)

# Trade sizing
ACCOUNT_SIZE = 50000               # Account size ($)
RISK_PER_TRADE_PERCENT = 1.0       # Risk 1% per trade

# Market scan flags
SCAN_OMX = True                    # Scan OMX tickers
SCAN_US = True                     # Scan US stocks/options
OMX_TICKER_LIST = "omxspi_tickers.txt"
MAX_SPREAD_WIDTH = 10              # Max spread width ($/share)

# Optional proxy (set to None if not using)
PROXY = None  # Example: {'http': 'http://proxy:port', 'https': 'https://proxy:port'}

# ============================
# Global Variables & Cache Setup
# ============================
results_list = []
stop_event = threading.Event()
progress_lock = threading.Lock()
total_tickers = 0

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# ============================
# Signal Handler for Clean Exit
# ============================
def signal_handler(sig, frame):
    logging.info("Exiting – saving results...")
    stop_event.set()
    if results_list:
        df_partial = pd.DataFrame(results_list)
        today_str = datetime.today().strftime('%Y-%m-%d')
        results_file = "pcs_results.txt"
        try:
            with open(results_file, "w" if not os.path.isfile(results_file) else "r+") as f:
                if os.path.isfile(results_file):
                    prev_results = f.read()
                    f.seek(0)
                else:
                    prev_results = ""
                f.write(f"\n=== Exited Credit Spreads {today_str} ===\n")
                f.write(df_partial.to_string(index=False))
                f.write(f"\n=== Previous Results ===\n{prev_results}")
        except Exception as e:
            logging.error(f"Error writing results: {e}")
    os._exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# ============================
# Caching Historical Data
# ============================
def get_history_with_cache(ticker_obj, period='12mo', interval='1d', max_retries=5):
    filename = os.path.join(CACHE_DIR, f"{ticker_obj.ticker}_history.pkl")
    if os.path.exists(filename) and datetime.fromtimestamp(os.path.getmtime(filename)).date() == datetime.now().date():
        try:
            return pd.read_pickle(filename)
        except Exception as e:
            logging.warning(f"Cache read error for {ticker_obj.ticker}: {e}")
    retries = 0
    while retries < max_retries:
        try:
            hist = ticker_obj.history(period=period, interval=interval, timeout=TIMEOUT, proxy=PROXY)
            if not hist.empty:
                pd.to_pickle(hist, filename)
                time.sleep(REQUEST_DELAY)  # Delay after fetching history
                return hist
            logging.warning(f"No historical data for {ticker_obj.ticker}")
            return pd.DataFrame()
        except (ReadTimeout, ConnectionError, YFRateLimitError) as e:
            wait_time = 60 * (retries + 1)
            logging.error(f"Fetch error for {ticker_obj.ticker}: {e}. Retrying in {wait_time}s (attempt {retries + 1}/{max_retries})")
            time.sleep(wait_time)
            retries += 1
        except Exception as e:
            logging.error(f"Unexpected error for {ticker_obj.ticker}: {e}")
            break
    return pd.DataFrame()

# New function to fetch info with retry logic
def get_info_with_retry(ticker_obj, max_retries=5):
    retries = 0
    while retries < max_retries:
        try:
            info = ticker_obj.info
            time.sleep(REQUEST_DELAY)  # Delay after successful fetch
            return info
        except YFRateLimitError:
            wait_time = 60 * (retries + 1)
            logging.error(f"Rate limit error fetching info for {ticker_obj.ticker}. Retrying in {wait_time}s (attempt {retries + 1}/{max_retries})")
            time.sleep(wait_time)
            retries += 1
    logging.error(f"Failed to fetch info for {ticker_obj.ticker} after {max_retries} attempts")
    return {}

# ============================
# Position Sizing
# ============================
def calculate_position_size(risk, account_size=ACCOUNT_SIZE, risk_pct=RISK_PER_TRADE_PERCENT):
    allowed_risk = account_size * (risk_pct / 100.0)
    return int(allowed_risk / risk) if risk > 0 else 0

# ============================
# Ticker Fetching Functions
# ============================
def get_nasdaq100_tickers():
    try:
        url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
        tables = pd.read_html(url)
        for table in tables:
            cols = [str(col).lower() for col in table.columns]
            if 'ticker' in cols or 'symbol' in cols:
                col_name = next(col for col in table.columns if str(col).lower() in ['ticker', 'symbol'])
                tickers = table[col_name].astype(str).str.replace(r'[^a-zA-Z.]', '', regex=True).unique().tolist()
                logging.info(f"Fetched Nasdaq-100: {len(tickers)} tickers")
                return [t for t in tickers if t not in ('nan', '')]
        return []
    except Exception as e:
        logging.error(f"Error fetching Nasdaq-100: {e}")
        return []

def get_sp500_tickers():
    try:
        df = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
        tickers = df['Symbol'].astype(str).str.replace(r'[^a-zA-Z.]', '', regex=True).unique().tolist()
        logging.info(f"Fetched S&P 500: {len(tickers)} tickers")
        return [t for t in tickers if t not in ('nan', '')]
    except Exception as e:
        logging.error(f"Error fetching S&P 500: {e}")
        return []

# ============================
# Options & Screening Helpers
# ============================
def get_options_chain_from_obj(ticker_obj, max_attempts=5):
    attempts = 0
    while attempts < max_attempts:
        try:
            puts = []
            options = ticker_obj.options
            time.sleep(REQUEST_DELAY)  # Delay after fetching options
            if not options:
                logging.info(f"{ticker_obj.ticker}: No options available")
                return pd.DataFrame()
            for exp in options:
                exp_date = datetime.strptime(exp, '%Y-%m-%d')
                dte = (exp_date - datetime.now()).days
                if MIN_DTE <= dte <= MAX_DTE:
                    ticker_exp = yf.Ticker(ticker_obj.ticker, proxy=PROXY) # Pass proxy here
                    chain = ticker_exp.option_chain(exp)
                    time.sleep(REQUEST_DELAY)  # Delay after fetching each option chain
                    if not chain.puts.empty:
                        chain.puts['dte'] = dte
                        chain.puts['expiration'] = exp
                        chain.puts['ticker'] = ticker_obj.ticker
                        puts.append(chain.puts)
            return pd.concat(puts).reset_index(drop=True) if puts else pd.DataFrame()
        except (ReadTimeout, ConnectionError, YFRateLimitError) as e:
            attempts += 1
            sleep_time = 60 * attempts
            logging.warning(f"Error fetching options for {ticker_obj.ticker}: {e}. Retrying in {sleep_time}s (attempt {attempts}/{max_attempts})")
            time.sleep(sleep_time)
        except Exception as e:
            logging.error(f"Unexpected error fetching options for {ticker_obj.ticker}: {e}")
            break
    logging.error(f"Failed to fetch options for {ticker_obj.ticker} after {max_attempts} attempts")
    return pd.DataFrame()

def filter_liquid_options(df):
    if df.empty:
        return df
    df = df.copy()
    df['liq_score'] = df.get('openInterest', 0) / (df.get('volume', 0) + 1)
    df = df[df['openInterest'] >= MIN_OPEN_INTEREST]
    df['bid_ask_spread_ratio'] = (df['ask'] - df['bid']) / ((df['ask'] + df['bid']) / 2)
    return df[df['bid_ask_spread_ratio'] <= MAX_BID_ASK_SPREAD_RATIO]

def calculate_historical_volatility(data):
    if len(data) < 10 or 'Close' not in data.columns:
        return 0.0
    returns = np.log(data['Close'] / data['Close'].shift(1)).dropna()
    return returns.std() * np.sqrt(252)

def check_technical_support(data):
    if data.empty or len(data) < 55 or 'Close' not in data.columns:
        return False
    window = 14
    delta = data['Close'].diff()
    rs = (delta.where(delta > 0, 0).rolling(window=window).mean() / 
          -delta.where(delta < 0, 0).rolling(window=window).mean())
    data['RSI'] = 100 - (100 / (1 + rs))
    ma_short = data['Close'].rolling(window=13).mean()
    ma_mid = data['Close'].rolling(window=21).mean()
    ma_long = data['Close'].rolling(window=55).mean()
    return (ma_short.iloc[-1] > ma_mid.iloc[-1] and 
            data['Close'].iloc[-1] > ma_long.iloc[-1] and 
            data['RSI'].iloc[-1] > 40)

def get_fundamentals(ticker_obj, info):
    try:
        required = ['profitMargins', 'ebitdaMargins', 'earningsGrowth', 'debtToEquity', 'returnOnEquity']
        if not all(key in info and info[key] is not None for key in required):
            return False
        net_income_ratio = float(info['profitMargins'])
        ebitda_ratio = float(info['ebitdaMargins'])
        earnings_growth = float(info['earningsGrowth'])
        debt_to_equity = float(info['debtToEquity']) / 100 if float(info['debtToEquity']) > 10 else float(info['debtToEquity'])
        roe = float(info['returnOnEquity']) * 100 if float(info['returnOnEquity']) < 1 else float(info['returnOnEquity'])
        return (net_income_ratio > 0 and ebitda_ratio > 0 and 
                earnings_growth > MIN_EARNINGS_GROWTH and 
                debt_to_equity < MAX_DEBT_TO_EQUITY and 
                roe > MIN_ROE)
    except Exception as e:
        logging.error(f"Fundamental error for {ticker_obj.ticker}: {e}")
        return False

def calculate_put_delta(S, strikes, dte, sigma):
    T = dte / 365.0
    d1 = (np.log(S / strikes) + (RISK_FREE_RATE + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) - 1

def calculate_put_price(S, strikes, dte, sigma):
    T = dte / 365.0
    d1 = (np.log(S / strikes) + (RISK_FREE_RATE + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return strikes * np.exp(-RISK_FREE_RATE * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def get_mid_price(row):
    return (row['bid'] + row['ask']) / 2 if pd.notnull(row['bid']) and pd.notnull(row['ask']) else row.get('lastPrice', 0)

# ============================
# Stock Scoring with ADX
# ============================
def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index (ADX) for trend strength."""
    plus_dm = high.diff().where(lambda x: x > 0, 0)
    minus_dm = -low.diff().where(lambda x: x < 0, 0)
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    
    atr = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
    adx = dx.rolling(window=period).mean()
    return adx

def calculate_stock_score(ticker_obj, hist_data, info):
    technical_score = fundamental_score = 0.0
    try:
        close = hist_data['Close']
        high = hist_data['High']
        low = hist_data['Low']
        if len(close) < 55:  # Ensure sufficient data for 55-day MA and ADX
            logging.info(f"{ticker_obj.ticker}: Insufficient data for technical scoring")
            return {'Technical Score': 0.0, 'Fundamental Score': 0.0, 'Total Score': 0.0}
        
        # Technical Score (max 5.0)
        price = close.iloc[-1]
        ma_55 = close.rolling(window=55).mean().iloc[-1]
        ma_short = close.rolling(window=13).mean().iloc[-1]
        ma_mid = close.rolling(window=21).mean().iloc[-1]
        if not np.isnan(ma_55) and price > ma_55:
            technical_score += 1.5  # Price above 55-day MA
        if not np.isnan(ma_mid) and ma_mid > 0 and ma_short > ma_mid:
            technical_score += 1.5  # Short-term trend strength
        
        delta = close.diff()
        if len(delta) >= 15:  # RSI component
            rs = delta.where(delta > 0, 0).rolling(window=14).mean().iloc[-1] / (delta.where(delta < 0, 0).rolling(window=14).mean().iloc[-1] or 1)
            rsi = 100 - (100 / (1 + rs)) if rs != 0 else 100
            if 40 <= rsi <= 70:  # Neutral to bullish momentum
                technical_score += 1.0
        
        if len(close) >= 28:
            adx = calculate_adx(high, low, close).iloc[-1]
            if not np.isnan(adx) and adx > 25:  # Strong trend
                technical_score += 1.0
        
        technical_score = min(technical_score, 5.0)  # Cap at 5.0
    except Exception as e:
        logging.error(f"Technical score error for {ticker_obj.ticker}: {e}")
    
    try:
        # Normalize fundamental metrics to 0-5 scale using pre-fetched info
        earnings_growth = float(info.get('earningsGrowth', 0))
        roe = float(info.get('returnOnEquity', 0)) * 100 if float(info.get('returnOnEquity', 0)) < 1 else float(info.get('returnOnEquity', 0))
        profit_margins = float(info.get('profitMargins', 0))
        
        eg_score = min(max(0, earnings_growth * 10), 5)  # 0.5 -> 5, cap at 5
        roe_score = min(max(0, roe / 20), 5)  # 100% ROE -> 5, cap at 5
        pm_score = min(max(0, profit_margins * 20), 5)  # 0.25 -> 5, cap at 5
        
        fundamental_score = (eg_score + roe_score + pm_score) / 3  # Average, max 5
    except Exception as e:
        logging.error(f"Fundamental score error for {ticker_obj.ticker}: {e}")
    
    total_score = (technical_score * 0.6) + (fundamental_score * 0.4)  # 60% technical, 40% fundamental
    return {
        'Technical Score': round(technical_score, 2),
        'Fundamental Score': round(fundamental_score, 2),
        'Total Score': round(total_score, 2)
    }

# ============================
# Ticker Processing
# ============================
def process_ticker(ticker_item):
    if stop_event.is_set():
        return None
    ticker, market = ticker_item['ticker'], ticker_item.get('market', 'US')
    use_options = market == 'US'
    logging.info(f"Processing {ticker} ({market})...")
    try:
        ticker_obj = yf.Ticker(ticker, proxy=PROXY)
        
        # Fetch all required data once
        hist_data = get_history_with_cache(ticker_obj)
        if hist_data.empty:
            logging.info(f"{ticker}: No historical data")
            return None
        if len(hist_data) < 55:
            logging.info(f"{ticker}: Insufficient historical data for analysis")
            return None
        
        # Fetch info with retry logic
        info = get_info_with_retry(ticker_obj)
        if not info:  # If info fetch fails after retries
            logging.info(f"{ticker}: Unable to fetch stock info")
            return None
        
        S = hist_data['Close'].iloc[-1]
        scores = calculate_stock_score(ticker_obj, hist_data, info)
        
        if scores['Technical Score'] <= 0:
            logging.info(f"{ticker}: Technical Score {scores['Technical Score']} is not greater than 0, skipping")
            return None

        if not use_options:  # OMX stocks: only technical/fundamental scoring
            if not check_technical_support(hist_data) or not get_fundamentals(ticker_obj, info):
                logging.info(f"{ticker}: Fails technical or fundamental criteria")
                return None
            result = {'Ticker': ticker, 'Market': market, 'Underlying Price': round(S, 2), **scores}
            logging.info(f"{ticker} qualifies with scores: {scores}")
            return result

        # US stocks: options analysis plus scoring
        puts = get_options_chain_from_obj(ticker_obj)
        if puts.empty:
            logging.info(f"{ticker}: No puts for {TARGET_DTE} DTE")
            return None
        puts = filter_liquid_options(puts)
        hist_vol = calculate_historical_volatility(hist_data.tail(30))
        if hist_vol == 0 or not check_technical_support(hist_data) or not get_fundamentals(ticker_obj, info):
            logging.info(f"{ticker}: Fails volatility, technical, or fundamental criteria")
            return None
        puts['sigma'] = puts.get('impliedVolatility', hist_vol).fillna(hist_vol)
        puts['delta'] = calculate_put_delta(S, puts['strike'], puts['dte'], puts['sigma'])
        puts['theoretical_price'] = calculate_put_price(S, puts['strike'], puts['dte'], puts['sigma'])
        candidates = puts[(puts['strike'] < S) & (puts['delta'].between(DELTA_LOWER, DELTA_UPPER))]
        if candidates.empty or (rich_candidates := candidates[candidates['lastPrice'] > candidates['theoretical_price']]).empty:
            logging.info(f"{ticker}: No rich candidates")
            return None
        sold_option = rich_candidates.assign(delta_diff=(rich_candidates['delta'] - TARGET_DELTA).abs()).sort_values('delta_diff').iloc[0]
        expiration_date = datetime.strptime(sold_option['expiration'], '%Y-%m-%d')
        current_time = datetime.now()
        for event, ts_key in [('Earnings', 'earningsTimestamp'), ('Dividend', 'exDividendDate')]:
            ts = info.get(ts_key)
            if ts and current_time <= (event_date := datetime.fromtimestamp(ts)) <= expiration_date:
                logging.info(f"{ticker}: {event} on {event_date:%Y-%m-%d} during DTE, skipping")
                return None
        buy_candidates = puts[(puts['strike'] < sold_option['strike']) & (puts['expiration'] == sold_option['expiration']) & 
                              ((sold_option['strike'] - puts['strike']) <= MAX_SPREAD_WIDTH)]
        if buy_candidates.empty:
            logging.info(f"{ticker}: No buy option within {MAX_SPREAD_WIDTH}")
            return None
        buy_option = buy_candidates.loc[buy_candidates['strike'].idxmax()]
        sold_mid, buy_mid = get_mid_price(sold_option), get_mid_price(buy_option)
        net_credit = (sold_mid - buy_mid) * 100
        risk = (sold_option['strike'] - buy_option['strike']) * 100
        if risk <= 0:
            logging.error(f"{ticker}: Invalid risk")
            return None
        annualized_return = (net_credit / risk) * (365 / sold_option['dte']) * 100
        prob_success = (1 - abs(sold_option['delta'])) * 100
        contracts = calculate_position_size(risk)
        result = {
            'Ticker': ticker, 'Market': market, 'Underlying Price': round(S, 2),
            **scores,
            'Sold Strike': sold_option['strike'], 'Sold Premium (Mid)': round(sold_mid, 2),
            'Buy Strike': buy_option['strike'], 'Buy Premium (Mid)': round(buy_mid, 2),
            'Expiration': sold_option['expiration'], 'DTE': sold_option['dte'],
            'Net Credit ($)': round(net_credit, 2), 'Risk ($)': round(risk, 2),
            'Annualized Return (%)': round(annualized_return, 1), 'Prob. Success (%)': round(prob_success, 1),
            'Contracts': contracts
        }
        logging.info(f"{ticker} qualifies, Return: {annualized_return:.1f}%, Contracts: {contracts}, Scores: {scores}")
        return result
    except Exception as e:
        logging.exception(f"Error processing {ticker}: {e}")
        return None

# ============================
# Main Screening
# ============================
def screen_credit_spreads(ticker_items):
    global results_list, total_tickers
    total_tickers = len(ticker_items)
    processed_count = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        # Stagger the start of workers
        for i, item in enumerate(ticker_items):
            futures[executor.submit(process_ticker, item)] = item
            if i < MAX_WORKERS - 1:  # Delay for all but the last worker to stagger starts
                time.sleep(REQUEST_DELAY / MAX_WORKERS)  # e.g., 3 / 6 = 0.5 seconds
        while futures:
            done, _ = wait(futures, timeout=1, return_when=FIRST_COMPLETED)
            for future in done:
                processed_count += 1
                with progress_lock:
                    logging.info(f"Processed {processed_count}/{total_tickers} tickers")
                if result := future.result():
                    results_list.append(result)
                del futures[future]
            if stop_event.is_set():
                break
    return pd.DataFrame(results_list)

# ============================
# Main Function
# ============================
def main():
    start_time = time.time()
    
    # Hämta användarinställningar och uppdatera globala variabler
    config = get_user_inputs()
    globals().update(config)
    
    # Uppdatera sökvägen för resultatfilen
    if getattr(sys, 'frozen', False):
        # Om vi kör från .exe
        script_dir = os.path.dirname(sys.executable)
    else:
        # Om vi kör från .py
        script_dir = os.path.dirname(os.path.abspath(__file__))
    
    results_file = os.path.join(script_dir, "results.txt")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    omx_ticker_file = os.path.join(script_dir, OMX_TICKER_LIST)
    combined_tickers = []

    if SCAN_OMX:
        if not os.path.isfile(omx_ticker_file):
            logging.info(f"Pulling OMX tickers...")
            try:
                omx_tickers = pull_omx_tickers()
                if omx_tickers:
                    combined_tickers.extend({"ticker": t.strip(), "market": "OMX"} for t in omx_tickers if t.strip())
            except Exception as e:
                logging.error(f"Error pulling OMX tickers: {e}")
        else:
            try:
                with open(omx_ticker_file, "r") as f:
                    combined_tickers.extend({"ticker": t.strip(), "market": "OMX"} for t in f.read().splitlines() if t.strip())
            except Exception as e:
                logging.error(f"Error reading OMX tickers: {e}")

    if SCAN_US:
        us_tickers = list(set(get_sp500_tickers() + get_nasdaq100_tickers()))
        combined_tickers.extend({"ticker": t, "market": "US"} for t in us_tickers if t.strip())

    if not combined_tickers:
        logging.error("No tickers to scan")
        return

    combined_tickers = combined_tickers[:MAX_TICKERS_TO_PROCESS]
    logging.info(f"Scanning {len(combined_tickers)} tickers")

    df = screen_credit_spreads(combined_tickers)
    today_str = datetime.today().strftime('%Y-%m-%d')

    if not df.empty:
        # Handle US data
        df_us = df[df['Market'] == 'US'].sort_values('Annualized Return (%)', ascending=False) if any(df['Market'] == 'US') else pd.DataFrame()
        
        # Handle OMX data
        if any(df['Market'] == 'OMX'):
            df_omx = df[df['Market'] == 'OMX'][["Ticker", "Market", "Underlying Price", "Technical Score", "Fundamental Score", "Total Score"]]
            if not df_omx['Total Score'].isna().all():
                df_omx = df_omx.sort_values('Total Score', ascending=False)
            else:
                logging.warning("No valid Total Score values for OMX data")
        else:
            df_omx = pd.DataFrame()

        logging.info(f"=== Best Results {today_str} ===")
        if not df_us.empty:
            logging.info(f"US Stocks/Options:\n{df_us.to_string(index=False)}")
        if not df_omx.empty:
            logging.info(f"OMX Stocks:\n{df_omx.to_string(index=False)}")

        try:
            # Skapa innehållet som ska skrivas
            new_content = f"=== Best Results {today_str} ===\n"
            if not df_us.empty:
                new_content += f"US Stocks/Options:\n{df_us.to_string(index=False)}\n\n"
            if not df_omx.empty:
                new_content += f"OMX Stocks:\n{df_omx.to_string(index=False)}\n\n"
            
            # Läs befintligt innehåll om filen finns
            old_content = ""
            if os.path.exists(results_file):
                try:
                    with open(results_file, 'r', encoding='utf-8') as f:
                        old_content = f.read()
                except Exception as e:
                    logging.warning(f"Kunde inte läsa befintlig results fil: {e}")

            # Skriv nytt innehåll
            with open(results_file, 'w', encoding='utf-8') as f:
                f.write(new_content)
                if old_content:
                    f.write(old_content)
            
            logging.info(f"Resultat sparade till: {results_file}")
            
        except Exception as e:
            logging.error(f"Fel vid skrivning till resultatfil: {e}")
            logging.error(f"Försökte skriva till: {results_file}")
            # Skriv ut mer information om felet
            import traceback
            logging.error(traceback.format_exc())
    else:
        logging.info("No opportunities found")

    logging.info(f"Processed {total_tickers} tickers in {time.time() - start_time:.1f}s")
    input("\nTryck Enter för att avsluta...")

# Lägg till denna funktion efter imports och före konfigurationsparametrar
def get_user_inputs():
    """Hämta användarinställningar med standardvärden"""
    print("\n=== Screener Inställningar ===")
    
    # Uppdaterad hantering av sökvägar för config-filen
    if getattr(sys, 'frozen', False):
        # Om vi kör från .exe
        base_path = os.path.dirname(sys.executable)
    else:
        # Om vi kör från .py
        base_path = os.path.dirname(os.path.abspath(__file__))
    
    config_file = os.path.join(base_path, "screener_config.json")
    
    # Standardkonfiguration
    default_config = {
        'MIN_PRICE': 1.0,
        'MIN_DTE': 20,
        'MAX_DTE': 45,
        'TARGET_DTE': 30,
        'DELTA_LOWER': -0.40,
        'DELTA_UPPER': -0.25,
        'TARGET_DELTA': -0.30,
        'MIN_LIQ_SCORE': 1.0,
        'MIN_OPEN_INTEREST': 50,
        'MAX_BID_ASK_SPREAD_RATIO': 0.05,
        'MIN_EARNINGS_GROWTH': 0.05,
        'MAX_DEBT_TO_EQUITY': 0.8,
        'MIN_ROE': 15.0,
        'ACCOUNT_SIZE': 50000,
        'RISK_PER_TRADE_PERCENT': 1.0,
        'SCAN_OMX': True,
        'SCAN_US': True,
        'MAX_WORKERS': 10,
        'REQUEST_DELAY': 10,
        'MAX_TICKERS_TO_PROCESS': 1000
    }
    
    # Försök läsa befintlig config
    try:
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                saved_config = json.load(f)
                logging.info(f"Läste in konfiguration från: {config_file}")
                
                # Använd sparade värden men behåll standardvärden för saknade nycklar
                default_config.update(saved_config)
                
                # Fråga användaren om de vill använda sparade inställningar
                use_saved = input("\nVill du använda sparade inställningar? (j/n) [j]: ").lower() != 'n'
                if use_saved:
                    logging.info("Använder sparade inställningar")
                    return default_config
                
    except Exception as e:
        logging.error(f"Kunde inte läsa konfigurationsfil: {e}")
        logging.error(f"Sökväg: {config_file}")

    # Om vi inte använder sparade inställningar, fråga användaren om nya
    def get_input(prompt, default, value_type=float):
        user_input = input(f"{prompt} [{default}]: ").strip()
        return value_type(user_input) if user_input else default

    config = {
        # Prisfilter
        'MIN_PRICE': get_input("Minimum aktiekurs", default_config['MIN_PRICE']),
        
        # DTE-filter
        'MIN_DTE': get_input("Minimum dagar till förfall", default_config['MIN_DTE'], int),
        'MAX_DTE': get_input("Maximum dagar till förfall", default_config['MAX_DTE'], int),
        'TARGET_DTE': get_input("Önskade dagar till förfall", default_config['TARGET_DTE'], int),
        
        # Delta-inställningar
        'DELTA_LOWER': get_input("Nedre deltagräns", default_config['DELTA_LOWER']),
        'DELTA_UPPER': get_input("Övre deltagräns", default_config['DELTA_UPPER']),
        'TARGET_DELTA': get_input("Önskat delta", default_config['TARGET_DELTA']),
        
        # Likviditetsfilter
        'MIN_LIQ_SCORE': get_input("Minimum likviditetsvärde", default_config['MIN_LIQ_SCORE']),
        'MIN_OPEN_INTEREST': get_input("Minimum open interest", default_config['MIN_OPEN_INTEREST'], int),
        'MAX_BID_ASK_SPREAD_RATIO': get_input("Max bid/ask spread (%)", default_config['MAX_BID_ASK_SPREAD_RATIO'] * 100) / 100,
        
        # Fundamentala filter
        'MIN_EARNINGS_GROWTH': get_input("Minimum vinsttillväxt (%)", default_config['MIN_EARNINGS_GROWTH'] * 100) / 100,
        'MAX_DEBT_TO_EQUITY': get_input("Max skuld/eget kapital", default_config['MAX_DEBT_TO_EQUITY']),
        'MIN_ROE': get_input("Minimum ROE (%)", default_config['MIN_ROE']),
        
        # Kontohantering
        'ACCOUNT_SIZE': get_input("Kontostorlek ($)", default_config['ACCOUNT_SIZE'], int),
        'RISK_PER_TRADE_PERCENT': get_input("Risk per trade (%)", default_config['RISK_PER_TRADE_PERCENT']),
        
        # Marknadsinställningar
        'SCAN_OMX': input("Skanna OMX? (j/n) [j]: ").lower() != 'n',
        'SCAN_US': input("Skanna US? (j/n) [j]: ").lower() != 'n',
        
        # Tekniska inställningar
        'MAX_WORKERS': get_input("Antal samtidiga processer", default_config['MAX_WORKERS'], int),
        'REQUEST_DELAY': get_input("API-fördröjning (sekunder)", default_config['REQUEST_DELAY'], int),
        'MAX_TICKERS_TO_PROCESS': get_input("Max antal aktier att skanna", default_config['MAX_TICKERS_TO_PROCESS'], int)
    }

    # Spara den nya konfigurationen
    try:
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        logging.info(f"Sparade konfiguration till: {config_file}")
    except Exception as e:
        logging.error(f"Kunde inte spara konfiguration: {e}")
        logging.error(f"Försökte spara till: {config_file}")

    return config

def pull_omx_tickers():
    """Hämtar OMX-tickers direkt från Nasdaq"""
    logging.info("Hämtar OMX-tickers från Nasdaq...")
    
    options = Options()
    options.add_argument("--headless")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    
    tickers = set()
    url = "https://indexes.nasdaqomx.com/index/Weighting/OMXSPI"
    
    try:
        driver.get(url)
        time.sleep(3)

        while True:
            time.sleep(2)
            soup = BeautifulSoup(driver.page_source, "html.parser")
            rows = soup.select("#weightingsTable tbody tr")

            for row in rows:
                columns = row.find_all("td")
                if len(columns) > 2:
                    raw_ticker = columns[2].text.strip()
                    formatted_ticker = raw_ticker.replace(" ", "-") + ".ST"
                    tickers.add(formatted_ticker)

            logging.info(f"Hämtade {len(tickers)} tickers hittills...")

            try:
                next_button = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((By.ID, "weightingsTable_next"))
                )
                if "disabled" in next_button.get_attribute("class"):
                    break
                driver.execute_script("arguments[0].scrollIntoView();", next_button)
                time.sleep(1)
                next_button.click()
            except:
                break

    except Exception as e:
        logging.error(f"Fel vid hämtning av OMX-tickers: {e}")
    finally:
        driver.quit()

    if tickers:
        with open(OMX_TICKER_LIST, "w", encoding="utf-8") as file:
            for ticker in sorted(tickers):
                file.write(ticker + "\n")
        logging.info(f"Sparade {len(tickers)} tickers i {OMX_TICKER_LIST}")
    
    return tickers

if __name__ == '__main__':
    main()