# Stock & Options Screener

En screener för aktier och optioner som analyserar både amerikanska (S&P 500, NASDAQ 100) och svenska (OMX) marknader. Programmet utför teknisk och fundamental analys för att identifiera potentiella handelsmöjligheter.

## Funktioner

- Screening av US aktier och optioner (S&P 500 och NASDAQ 100)
- Screening av svenska aktier (OMX)
- Teknisk analys (RSI, Moving Averages, ADX)
- Fundamental analys (ROE, Earnings Growth, Debt/Equity)
- Options analys för US-marknaden (Put Credit Spreads)
- Automatisk datahämtning från Yahoo Finance
- Konfigurerbara screeningparametrar
- Sparar resultat till fil
- Cachad datahämtning för bättre prestanda

## Systemkrav

- Python 3.8 eller senare
- Windows/Linux/MacOS

## Installation

1. Installera beroenden:
```
pip install -r requirements.txt
```
## Användning

1. Konfigurera screeningparametrar i `screener_config.json`

2. Kör programmet:

```
python screener.py
```

eller screener.exe

