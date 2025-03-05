import matplotlib.font_manager
import pandas as pd

list_ttf = [f.name for f in matplotlib.font_manager.fontManager.ttflist]

list_afm = [f.name for f in matplotlib.font_manager.fontManager.afmlist]

list_font = list_ttf + list_afm

font_df = pd.DataFrame()
font_df['font_name'] = list(set(list_font))

font_df.to_csv('font_list.csv')
