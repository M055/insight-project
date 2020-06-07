
# VERSION V2
import pandas as pd
import seaborn as sns
import numpy as np
import streamlit as st
import sklearn.metrics.pairwise as sklpw
from PIL import Image

st.title('Power to the Meeple')
st.markdown('** MEANINGful recommendations for the novice board gamer**')

meeple_image = Image.open('other/meeple.png')
st.image(meeple_image, caption='meeple',width=200)

# LOAD DATA
allgamedata_df = pd.read_pickle('datasources/BGG_FINAL.pkl')
finalgamelist_df = pd.read_pickle('datasources/BGG_GameSimilarityKey.pkl')
allgamedocvects = np.load('datasources/allgamedocvects.npz')['arr_0']

# FUNCTIONS
def getcompute_similar_games_by_name(mygameid,allgamedata_df,allgamedocvects,finalgamelist_df):
    myvectid = mygameid
    mygamevector = allgamedocvects[myvectid,:]
    mygamevector= mygamevector.reshape(-1,1)
    mysimilarities = []
    for t in range(0,allgamedocvects.shape[0]):
        currgamevect = allgamedocvects[t,:]
        currgamevect = currgamevect.reshape(-1,1)
        dum = sklpw.cosine_similarity(currgamevect.T,mygamevector.T)
        mysimilarities.append(dum[0][0])
    mycompletesimlist_df = pd.concat((finalgamelist_df['gamename'],pd.DataFrame({'Similarity':mysimilarities})),axis=1)
    mycompletesimlist_df.sort_values(by='Similarity',ascending=False,inplace=True)
    mytop10simlist_df = mycompletesimlist_df[1:11]
    # Create output list
    urllist=[]
    for gamename in mytop10simlist_df['gamename']:
        urllist.append(list(allgamedata_df.loc[allgamedata_df['game_name']==gamename,'bgg_url'])[0])
    mytop10simlist_df = pd.DataFrame({'Game':mytop10simlist_df['gamename'],'Similarity':mytop10simlist_df['Similarity'],'url':urllist})
    mytop10simlist_df.reset_index(drop=True,inplace=True)
    mytop10simlist_df.index = mytop10simlist_df.index+1
    return mytop10simlist_df


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
    df['Similar game'] = [make_clickable(a,b) for a,b in zip(list(df['url']),list(df['Game']))]
    return df


#  CREATES THE DEMO GAME LIST
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

        mygameid = list(finalgamelist_df.index[finalgamelist_df['gamename']==mygamename])[0] # Need INDEX, not idx
        mygameurl=list(allgamedata_df.loc[allgamedata_df['game_name']==mygamename,'bgg_url'])[0]
        mytop10simlist_df = getcompute_similar_games_by_name(mygameid,allgamedata_df,allgamedocvects,finalgamelist_df)
        mygamevect_df = streamlitify_df(mytop10simlist_df)

        #mygamevect_df['st_URL'] = [make_clickable(a,b) for a,b in zip(list(mygamevect_df['url']),list(mygamevect_df['gamename']))]
        #st.table(mygamevect_df.head(11))
        #st.write(mygamevect_df.to_html(escape = False), unsafe_allow_html = True)
        st.write(mygamevect_df[['Similar game','Similarity']].to_html(escape = False), unsafe_allow_html = True)
    st.success('Done!')





#sns.heatmap(np.random.randn(10,10))
#st.pyplot()
