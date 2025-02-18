
import pandas as pd

# Wczytaj oryginalny plik
tax_df = pd.read_csv('wyniki_nameco_fungi/taxonomy/taxonomy_annotations.tsv', sep='\t')

# Utwórz kolumnę taksonomii w formacie QIIME2
def create_taxonomy_string(row):
    ranks = [
        f"k__{row['Kingdom']}", 
        f"p__{row['Phylum']}", 
        f"c__{row['Class']}", 
        f"o__{row['Order']}", 
        f"f__{row['Family']}", 
	f"g_{row['Genus']}", 
        f"s__{row['Species']}"
    ]
    return '; '.join(ranks)

# Utwórz nowy DataFrame w formacie QIIME2
qiime_tax = pd.DataFrame({
    'Feature ID': tax_df['query_id'],
    'Taxon': tax_df.apply(create_taxonomy_string, axis=1)
})

# Zapisz w formacie QIIME2
qiime_tax.to_csv('to_qiime2/taxonomy_qiime2.tsv', sep='\t', index=False)
