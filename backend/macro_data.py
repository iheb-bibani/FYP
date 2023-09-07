import requests
import pandas as pd
import psycopg2
from postgres import connection
from datetime import datetime


def main():
    """
    NGDP_RPCH: Real GDP growth rate (annual %),
    NGDPD: GDP (current Billion US$),
    NGDPDPC: GDP per capita (current US$),
    PPPGDP: GDP, PPP (current international $),
    PPPPC: GDP per capita, PPP (current international $),
    PPPSH: GDP Based on PPP, Share of World Total (%),
    PPPEX: Implied PPP Conversion Rate, GDP (current international $ per international $),
    PCPIPCH: Inflation, average consumer prices (annual %),
    PCPIEPCH: Inflation, end of period consumer prices (annual %),
    LP: Population (Persons),
    BCA: Current Account Balance (Billion US$),
    BCA_NGDPD: Current Account Balance (Percent of GDP),
    Unemployment Rate (Percent of total labor force),
    GGXCNL_NGDP: General government net lending/borrowing (Percent of GDP),
    GGXWDG_NGDP: General government gross debt (Percent of GDP)

    extensive: Extensive margin (Percent of GDP),
    intensive: Intensive margin (Percent of GDP),
    total_theil: Total Theil index (index),
    SITC1_0: Share of world exports in total exports (%),

    DirectAbroad: Direct investment abroad (Billion US$),
    DirectIn: Direct investment in the reporting economy (Billion US$),

    PrivInexDI: Private inward direct investment (Billion US$),
    PrivInexDIGDP: Private inward direct investment (Percent of GDP),

    ka_new: New capital expenditure (Billion US$),
    ka_in: Capital expenditure (Billion US$),
    ka_out: Capital expenditure (Billion US$),
    FM_ka: Foreign direct investment (Billion US$),

    GII_TC: Global Innovation Index (index),
    DEBT1: Debt service (Percent of exports of goods, services and primary income),
    PVD_LS: Public debt (Percent of GDP),
    HH_LS: Household debt (Percent of GDP),
    NFC_LS: Non-financial corporate debt (Percent of GDP),

    GGXCNL_G01_GDP_PT: General government net lending/borrowing (Percent of GDP),
    GGCB_G01_PGDP_PT: General government consolidated gross debt (Percent of GDP),
    FR_ind: Financial resources (Percent of GDP)
    """
    indicators = [
        "NGDP_RPCH",
        "NGDPD",
        "NGDPDPC",
        "PPPGDP",
        "PPPPC",
        "PPPSH",
        "PPPEX",
        "PCPIPCH",
        "PCPIEPCH",
        "LP",
        "BCA",
        "BCA_NGDPD",
        "LUR",
        "GGXCNL_NGDP",
        "GGXWDG_NGDP",
        "extensive",
        "intensive",
        "total_theil",
        "SITC1_0",
        "DirectAbroad",
        "DirectIn",
        "PrivInexDI",
        "PrivInexDIGDP",
        "ka_new",
        "ka_in",
        "ka_out",
        "FM_ka",
        "GII_TC",
        "DEBT1",
        "PVD_LS",
        "HH_LS",
        "NFC_LS",
        "GGXCNL_G01_GDP_PT",
        "GGCB_G01_PGDP_PT",
        "FR_ind",
    ]
    country_code = "SGP"  # Singapore
    years = list(range(2010, datetime.now().year + 1))

    all_data = fetch_imf_data(country_code, indicators, years)

    df = pd.DataFrame(all_data)
    df["Year"] = years
    df.dropna(axis=1, how="any", inplace=True)
    store_to_db(df, connection)


def fetch_imf_data(country_code, indicators, years):
    all_data = {}
    for indicator in indicators:
        url = f"https://www.imf.org/external/datamapper/api/v1/{indicator}/{country_code}?periods={','.join(map(str, years))}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            all_data[indicator] = data["values"][indicator][country_code]
        else:
            print(f"Failed to retrieve data for {indicator}: {response.status_code}")
    return all_data


def store_to_db(df: pd.DataFrame, connection: psycopg2.extensions.connection) -> None:
    cursor = connection.cursor()

    columns = ", ".join([f"{col} FLOAT" for col in df.columns if col != "Year"])
    cursor.execute(
        f"CREATE TABLE IF NOT EXISTS macroeconomic_data (id SERIAL PRIMARY KEY, year INTEGER UNIQUE, {columns})"
    )

    for _, row in df.iterrows():
        year = row["Year"]
        values = [row[col] for col in df.columns if col != "Year"]
        placeholders = ", ".join(["%s" for _ in values])

        cursor.execute(
            f"INSERT INTO macroeconomic_data (year, {', '.join(df.columns[df.columns != 'Year'])}) VALUES (%s, {placeholders}) ON CONFLICT (year) DO NOTHING",
            [year] + values,
        )

    connection.commit()
    cursor.close()


if __name__ == "__main__":
    main()
