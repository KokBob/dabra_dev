# -*- coding: utf-8 -*-
"""
https://www.youtube.com/watch?v=RCUrpCpGZ5o&t=1156s
https://www.youtube.com/watch?v=-Mx89Jcn2E4&list=PLh3I780jNsiTXlWYiNWjq2rBgg3UsL1Ub&index=5
stav: nejaky erro v dashi, neprehava tak jak je treba
https://www.youtube.com/playlist?list=PLh3I780jNsiTXlWYiNWjq2rBgg3UsL1Ub
https://community.plotly.com/t/slider-with-play-button-for-animations-independent-of-plotly/53188
https://dash.plotly.com/dash-core-components/slider
https://community.plotly.com/t/slider-with-play-button-for-animations-independent-of-plotly/53188/2
https://www.youtube.com/watch?v=d9SmpNfMg7U
https://towardsdatascience.com/how-to-create-animated-scatter-maps-with-plotly-and-dash-f10bb82d357a play button
https://stackoverflow.com/questions/71906091/python-plotly-dash-automatically-iterate-through-slider-play-button
"""
#******* CLEANED AND CLOSED *******

import sys
import pandas as pd
sys.path.append("..") 
# sys.path.append("../..") 
import insight
# %%
# ins = insight.emts()
ins = insight.emts(path2emts = '../../data/1201')
a_wi = ins.list_emts
# tady je potreba uz pocitat s nejakou hiarchickou rekurenci
if len(a_wi) > 17:
    print(f'hiearchicka mereni: {len(a_wi)/3}')
# ins.get_emt_list()
# %%
a_wi = ins.list_emts
# print(f'{a_wi}')
# %%

# %%
# ins.print_emts_stems()
# %%
# emt_file_path = '../../data/0409/3D point tracks.emt' # :dart:: putit to conf
# v1d_file_path = '../../data/0409/1D velocity track.emt' # :dart:: putit to conf
# df_3DU = pd.read_csv(emt_file_path, skiprows=9, delimiter=r"\s+")
# %%
# df_3DU = pd.read_csv(emt_file_path, skiprows=9, delimiter=r"\s+")
# df_3DU = pd.DataFrame(df_3DU.values.reshape(89,3), columns = ['X', 'Y','Z'])
# df_1Dv  = pd.read_csv(v1d_file_path, skiprows=9, delimiter=r"\s+")
# %%
# df_1Dv.Vel.plot()

# df_1Dv.Vel.plot()
