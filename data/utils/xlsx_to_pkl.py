from dataclasses import dataclass
import argparse
import pickle

import pandas as pd

@dataclass
class Player:
    name: str
    prices : list[int]


def convert_xlsx_data_to_pkl(path: str, output_file: str = "players.pkl"):
    players: dict[str, list[int]] = {}

    print("TEAMS")
    dataframeTeams = pd.read_excel(path, index_col=None)
    # summarize shape
    print("Shape:" + str(dataframeTeams.shape))
    # summarize first few lines
    print("Summary Players")
    print(dataframeTeams)

    for _, row in dataframeTeams.iterrows():
        name = row['Name']
        price = row['Price']
        if pd.notna(price):
            price = int(price)
            if name not in players:
                players[name] = []
            players[name].append(price)

    with open(output_file, 'wb') as f:
        player_objects = [Player(name, prices) for name, prices in players.items()]
        pickle.dump(player_objects, f)

def convert_pkl_to_xlsx(pkl_file: str, output_file: str = "players.xlsx"):
    with open(pkl_file, 'rb') as f:
        player_objects = pickle.load(f)

    data = []
    for player in player_objects:
        for price in player.prices:
            data.append({'Name': player.name, 'Price': price})

    df = pd.DataFrame(data)
    df.to_excel(output_file, index=False)

def main():
    parser = argparse.ArgumentParser(description="Convert Excel file to PKL format")
    parser.add_argument('--excel_file', type=str, required=True, help='Path to the Excel file')
    parser.add_argument('--output_file', type=str, required=False, help='Path to the output PKL file')
    args = parser.parse_args()

    excel_file = args.excel_file
    print(f"Excel file provided: {excel_file}")

    process_teams_data(excel_file)

if __name__ == "__main__":
    main()