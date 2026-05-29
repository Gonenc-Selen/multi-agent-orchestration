import pandas as pd

df = pd.read_csv(r'C:\Users\selen\OneDrive\Desktop\IOTgent\Data\houseteh1_1day.csv')
print(f"Toplam satır: {len(df)}")
print(f"Sütunlar: {df.columns.tolist()}")
print(f"İlk 5 satır:")
print(df.head())
print(f"Son 5 satır:")
print(df.tail())
print(f"Boş olmayan kayıt sayısı: {df.iloc[:, 1].notna().sum()}")