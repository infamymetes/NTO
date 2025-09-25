import yfinance as yf
import pandas as pd

print("--- Starting Data Test ---")

try:
    # Attempt to download data for SPY
    ticker = "SPY"
    print(f"Attempting to download data for {ticker}...")
    
    data = yf.download(
        ticker, 
        start="2023-01-01", 
        end="2023-12-31",
        progress=False
    )
    
    # Check if the download was successful
    if not data.empty:
        print("\n✅ SUCCESS: Data downloaded successfully!")
        print(f"Downloaded {len(data)} rows of data for {ticker}.")
        print("This means yfinance is working correctly on your system.")
    else:
        print("\n❌ FAILURE: Download returned an empty result.")
        print("This is likely the reason the main script is not working.")
        print("Possible causes: network issue, firewall, or a problem with the yfinance service.")

except Exception as e:
    print(f"\n❌ CRITICAL FAILURE: An error occurred during download.")
    print(f"Error details: {e}")
    print("\nThis confirms there is a problem with the data fetching.")
    print("Please check your internet connection and make sure yfinance is installed (`pip install yfinance`).")

print("\n--- Data Test Finished ---")