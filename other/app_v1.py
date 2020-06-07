
import pandas as pd
import seaborn as sns
import numpy as np
import streamlit as st
from PIL import Image

st.title('Power to the Meeple')
st.markdown('** MEANINGful recommendations for the novice board gamer**')



meeple_image = Image.open('other/meeple.png')
st.image(meeple_image, caption='meeple',width=200)

# LOAD DATA
allgamedata_df = pd.read_pickle('datasources/BGG_FINAL.pkl')
allgamesimilarities_df = pd.read_pickle('datasources/BGG_GameSimilarityMatrix.pkl')
finalgamelist_df = pd.read_pickle('datasources/BGG_GameSimilarityKey.pkl')

# FUNCTIONS
def get_similar_games_by_name(mygamename,allgamesimilarities_df):
    dum=allgamesimilarities_df.loc[:,mygamename]
    if min(dum.shape)<10: # There are multiple columns with the same game name?!
        mygamevect = dum.iloc[:,0]
    else:
        mygamevect = list(allgamesimilarities_df.loc[:,mygamename])
    mygamevect_df = pd.DataFrame({'gamename':allgamesimilarities_df.index,'cosinesimilarity':mygamevect})
    mygamevect_df.sort_values(by='cosinesimilarity',inplace=True,ascending=False)
    return mygamevect_df

def get_similar_games_by_name_fuzzy(mygamename):
    gamename_matchlist = [fuzz.token_sort_ratio(x,mygamename) for x in finalgamelist_df['gamename']]
    possiblegame_idx  = [i for i, x in enumerate(gamename_matchlist) if x == max(gamename_matchlist)]
    possiblegame_idx = possiblegame_idx[0] # Get first, make it number
    possiblegame_name = list(finalgamelist_df.loc[finalgamelist_df['idx']==possiblegame_idx,'gamename'])[0]
    print('Best match: {}'.format(possiblegame_name))
    mygamevect = list(allgamesimilarities_df.loc[:,possiblegame_name])
    mygamevect_df = pd.DataFrame({'gamename':allgamesimilarities_df.index,'cosinesimilarity':mygamevect})
    mygamevect_df.sort_values(by='cosinesimilarity',inplace=True,ascending=False)
    return mygamevect_df

def make_clickable(url,text): # Make liks in pd df for url in table
    return f'<a target="_blank" href="{url}">{text}</a>'

def streamlitify_df(df):
    # Get original URLS
    df['url'] = [list(allgamedata_df.loc[allgamedata_df['game_name']==x,'bgg_url'])[0] for x in list(df['gamename'])]
    df['Game_link'] = [make_clickable(a,b) for a,b in zip(list(df['url']),list(df['gamename']))]
    return df

# DO SOME STUFF
allgamedata_df['numeric_ranks']=[int(x) for x in allgamedata_df['game_rank']]
topranked_df = pd.DataFrame(allgamedata_df.loc[allgamedata_df['numeric_ranks']<=50,'game_name']) # To go back n forth
topranked_idx = topranked_df.index
demo_gamelist = tuple(list(finalgamelist_df.loc[topranked_idx,'gamename']))

# SHOW SOME STUFF
mygamename = st.selectbox('Choose a game',demo_gamelist)

# WHEN YOU CLICK THE BUTTON...
clicked = st.button('Go')
if clicked:
    with st.spinner('Looking for similar games...'):
        mygameid = allgamedata_df.loc[allgamedata_df['game_name']==mygamename,:].index[0]
        mygameurl=allgamedata_df.loc[mygameid,'bgg_url']
        st.write('Games closest to your chosen game:')
        st.write(mygameurl)

        mygamevect_df = get_similar_games_by_name(mygamename,allgamesimilarities_df)
        mygamevect_df = mygamevect_df.head(11)
        mygamevect_df = streamlitify_df(mygamevect_df)

        #mygamevect_df['st_URL'] = [make_clickable(a,b) for a,b in zip(list(mygamevect_df['url']),list(mygamevect_df['gamename']))]
        #st.table(mygamevect_df.head(11))
        #st.write(mygamevect_df.to_html(escape = False), unsafe_allow_html = True)
        st.write(mygamevect_df[['Game_link','cosinesimilarity']].to_html(escape = False), unsafe_allow_html = True)
    st.success('Done!')





#sns.heatmap(np.random.randn(10,10))
#st.pyplot()
