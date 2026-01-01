import requests
from bs4 import BeautifulSoup
from pathlib import Path

SP_TICKER_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

OUTPUT_FILE_PATH = Path(__file__).parent / 'tickers.py'

agent_headers = {
    'User-Agent' : 'Portfolio optimizer scraping the tickets from raw HTML'
}

response = requests.get(SP_TICKER_URL, headers=agent_headers)

response_raw_html = response.text

soup = BeautifulSoup(response_raw_html, 'lxml')

TABLE_ID = "constituents" 

table = soup.find('table', id=TABLE_ID)
with open(OUTPUT_FILE_PATH, mode='w') as file:
    file.write('TICKERS=[\n')
    if table:
        rows = table.find_all('tr')

        for row in rows:
            cells = row.find_all(['td', 'th'])
            data = [cell.get_text(strip=True) for cell in cells]
            if data[0] == 'Symbol': continue
            file.write(
                f"\'{str(data[0])}\',\n" 
            )
        file.write(']')
    else:
        print("Table not found!")
