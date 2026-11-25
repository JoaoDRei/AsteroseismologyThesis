import pandas as pd

df = pd.read_csv("Baseline/manifest.csv")

print(df[["kic", "nu_max", "teff", "metallicity", "evolstate"]].head(10))
print("\nEvolutionary states:")
print(df["evolstate"].value_counts(dropna=False))
print("\nMetallicity distribution:")
print(df["metallicity"].describe())