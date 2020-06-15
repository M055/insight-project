# VERSION V3
import pandas as pd
import seaborn as sns
import numpy as np
import streamlit as st
import sklearn.metrics.pairwise as sklpw
from PIL import Image
from fuzzywuzzy import fuzz
import re

st.title('Meeple for People')
st.header('**MEANINGful recommendations for the novice board gamer**')

meeple_image = Image.open('other/meeple.png')
st.sidebar.image(meeple_image, caption='meeple',width=100)
st.sidebar.markdown('**Mee.ple** _noun_ \n a small figure used as a playing piece in certain board games, having a stylized human form.')
#st.sidebar.markdown('Early 21st century: apparently a blend of **my** and a phonetic respelling of **people** and first used with reference to the board game ''<a target="_blank" href="https://boardgamegeek.com/boardgame/822/carcassonne">Carcassonne</a>.', unsafe_allow_html=True)
#st.sidebar.markdown('_-Google_')

min_rating = st.sidebar.slider('Minimum BGG average rating', min_value=1, max_value=9, value=7, step=1) 
min_players = st.sidebar.slider('Minimum number of players', min_value=1, max_value=10, value=1, step=1) 
min_dur = st.sidebar.slider('Minimum time to play (min)', min_value=1, max_value=100, value=1, step=1) 


st.sidebar.header('What do these recommendations mean?')
st.sidebar.markdown('**Conceptually similar games**: These are games that, as a whole, are similar to your target game. For example, you would summarize them in very similar ways when describing them to your friend.')
st.sidebar.markdown('**Games with similar features**: These are games that are similar to your target game in terms of specific features such as the type and genre of the game, or the mechanics it employs.')


st.markdown('** Select a game from the dropdown list of top rated games on BoardGameGeek.com, or enter a game name in the text box. Press _Go_ to search **')


# LOAD DATA

allgamedata_df = pd.read_pickle('datasources/bgg_filters.pkl') # USE FOR URLS and FILTERS
allgamedata_df = allgamedata_df.astype({'game_rank':'int32'},copy=True)
allgamedocvects = np.load('datasources/allgamedocvects_v3.npz')['arr_0']
finalgamelist_df = pd.read_pickle('datasources/finalgamelist_df.pkl')
#finalgamelist_df =  finalgamelist_df.astype({'game_rank':'int32'},copy=True)
#finalgamelist_df['num_raters'] = [list(allgamedata_df.loc[allgamedata_df['game_name']==g,'num_raters'])[0] for g in #finalgamelist_df['game_name']]
#finalgamelist_df.reset_index(drop=True,inplace=True) # So that row ids are indices to gamevector array

# FOr gameplay-based vectors
bgg_gameplay_df = pd.read_pickle('datasources/bgg_gameplayfeatures.pkl')
bgg_gameplay_df.dropna(inplace=True) # Some not-NAs here...
bgg_gameplay_df.reset_index(drop=True,inplace=True)
allgamePLAYdocvects = np.array(bgg_gameplay_df.iloc[:,1:]) # Create right here

# FUNCTIONS

def getcompute_similar_by_gameplay(gamename,allgamedata_df, bgg_gameplay_df, allgamePLAYdocvects):
    # Get game rank from game name, and matrix index from rank
  
    gamerank = list(allgamedata_df.loc[allgamedata_df['game_name']==gamename,'game_rank'])[0]
    gamerank = int(gamerank)
    gamerank_idx = list(bgg_gameplay_df.index[bgg_gameplay_df['game_rank']==gamerank])[0]
    #print(gamename, gamerank_idx)
    
    mygamePLAYvector = allgamePLAYdocvects[gamerank_idx,:] 
    mygamePLAYvector = mygamePLAYvector.reshape(-1,1)

    mysimilarities_gp = []
    for t in range(0,allgamePLAYdocvects.shape[0]):
        currgamevect_gp = allgamePLAYdocvects[t,:]
        currgamevect_gp = currgamevect_gp.reshape(-1,1)
        dum = sklpw.cosine_similarity(currgamevect_gp.T,mygamePLAYvector.T)
        mysimilarities_gp.append(dum[0][0])
    mycompleteGPsimlist_df = pd.concat((pd.DataFrame({'game_rank':bgg_gameplay_df['game_rank']}),pd.DataFrame({'GameplaySimilarity':mysimilarities_gp})),axis=1)
    return mycompleteGPsimlist_df

def getcompute_similar_games(mygameid,mygamename,allgamedata_df,allgamedocvects,finalgamelist_df,bgg_gameplay_df, allgamePLAYdocvects,W1,W2,filt_dict):
    myvectid = mygameid
    mygamevector = allgamedocvects[myvectid,:]
    mygamevector= mygamevector.reshape(-1,1)
    mysimilarities = []
    for t in range(0,allgamedocvects.shape[0]):
        currgamevect = allgamedocvects[t,:]
        currgamevect = currgamevect.reshape(-1,1)
        dum = sklpw.cosine_similarity(currgamevect.T,mygamevector.T)
        mysimilarities.append(dum[0][0])
        
     # GET NUMRATERs
    dumnumraters = [allgamedata_df.loc[allgamedata_df['game_name']==g,'num_raters'] for g in finalgamelist_df['game_name']]
    mycompletesimlist_df = pd.concat((finalgamelist_df[['game_rank','game_name', 'num_raters']],pd.DataFrame({'Similarity':mysimilarities})),axis=1)

    #pd.concat((finalgamelist_df['game_rank'],finalgamelist_df['game_name'],pd.DataFrame({'Similarity':mysimilarities})),axis=1)
    # Get this also for GAMEPLAY data
    mycompleteGPsimlist_df = getcompute_similar_by_gameplay(mygamename,allgamedata_df, bgg_gameplay_df, allgamePLAYdocvects)

    # PUt sim lists together
    mycompletesimlist_df  = mycompletesimlist_df.astype({'game_rank':'int32'},copy=True)
    mycompleteGPsimlist_df = mycompleteGPsimlist_df.astype({'game_rank':'int32'},copy=True)
    mycompletesimlist_df.set_index('game_rank',inplace=True)
    mycompleteGPsimlist_df.set_index('game_rank',inplace=True)
    # Do it
    myFINALsimlist_df = mycompletesimlist_df.join(mycompleteGPsimlist_df,how='inner')
    
    
    weightedsimilarity = (np.array(myFINALsimlist_df['Similarity'])*W1) + (np.array(myFINALsimlist_df['GameplaySimilarity'])*W2)
    myFINALsimlist_df['WghtdSimilarity'] = weightedsimilarity
    myFINALsimlist_df.sort_values(by='WghtdSimilarity',ascending=False,inplace=True)
    
    # NEW: Jun 13
    # Create DF with all the filter data
    dum=allgamedata_df.loc[[list(allgamedata_df.index[allgamedata_df['game_rank']==n])[0] for n in myFINALsimlist_df.index],:]
    dum = dum[[ 'avg_rating', 'bgg_url', 'numplayersmin', 'gamedurmin','agemin']].copy()
    dum.reset_index(drop=True, inplace=True)
    # Add this into myFINALsimlist_df
    myFINALsimlist_df.reset_index(inplace=True)
    myFINALsimlist_df=pd.concat((myFINALsimlist_df,dum),axis=1)
    # Create filter
    myfilters = (myFINALsimlist_df['avg_rating']>=filt_dict.get('min_rating'))  & (myFINALsimlist_df['numplayersmin']>=filt_dict.get('min_players'))  &     (myFINALsimlist_df['gamedurmin']>=filt_dict.get('min_dur')) &    (np.log10(myFINALsimlist_df['num_raters'])>=filt_dict.get('min_numraters'))
    # UPDATE myFINALsimlist_df
    myFINALsimlist_df = myFINALsimlist_df.loc[myfilters,:].copy()
    
    
    # Create output list
    mytop10simlist_df = myFINALsimlist_df[1:11]
    urllist=[]
    for gamename in mytop10simlist_df['game_name']:
        urllist.append(list(allgamedata_df.loc[allgamedata_df['game_name']==gamename,'bgg_url'])[0])
    #mytop10simlist_df = pd.DataFrame({'Game':mytop10simlist_df['game_name'],'Similarity':mytop10simlist_df['WghtdSimilarity'],'url':urllist})
    mytop10simlist_df = pd.DataFrame({'Game':mytop10simlist_df['game_name'],'url':urllist})
    mytop10simlist_df.reset_index(drop=True,inplace=True)
    mytop10simlist_df.index = mytop10simlist_df.index+1
    return mytop10simlist_df
    
    
def get_real_name_fuzzy(usergamename,finalgamelist_df):
    # Clean up
    usergamename = re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", usergamename)
    #usergamename = re.sub(r"\d+", "", usergamename) # Maybe dont remove numbers?
    gamename_matchlist = [fuzz.token_sort_ratio(x,usergamename) for x in finalgamelist_df['game_name']]
    possiblegame_idx  = [i for i, x in enumerate(gamename_matchlist) if x == max(gamename_matchlist)]
    possiblegame_idx = possiblegame_idx[0] # Get first, make it number
    #possiblegame_name = list(finalgamelist_df.loc[finalgamelist_df['idx']==possiblegame_idx,'gamename'])[0]
    possiblegame_name = finalgamelist_df.iloc[possiblegame_idx,2]
    #print('Best match: {}'.format(possiblegame_name))
    
    return possiblegame_name,max(gamename_matchlist)

def make_clickable(url,text): # Make liks in pd df for url in table
    return f'<a target="_blank" href="{url}">{text}</a>'

def streamlitify_df(df):
    # Get original URLS
    df['Game link'] = [make_clickable(a,b) for a,b in zip(list(df['url']),list(df['Game']))]
    return df


# CREATES THE DEMO GAME LIST
allgamedata_df['numeric_ranks']=[int(x) for x in allgamedata_df['game_rank']]
topranked_df = pd.DataFrame(allgamedata_df.loc[allgamedata_df['numeric_ranks']<=50,'game_name']) # To go back n forth
topranked_df.sort_values(by='game_name',inplace=True)
topranked_idx = topranked_df.index
demo_gamelist = tuple(list(finalgamelist_df.loc[topranked_idx,'game_name']))


# SHOW SOME STUFF
mydemogamename = st.selectbox('Choose a game',demo_gamelist)

st.write('Or..')

# Setup
defaultnumraters = 2 # At least so many raters (log10)
usergamename = 'Gloomhaven'
usergamename = st.text_input('Enter game name', 'Gloomhaven',max_chars=30)
#usemygamename = st.checkbox('Use game selected from the list')
gamename_source = st.radio("Process game:",('from the dropdown', 'from the textbox'))
if gamename_source == 'from the dropdown':
    usemygamename = False
else:
    usemygamename = True
    
    

# WHEN YOU CLICK THE BUTTON...
clicked = st.button('Go')
if clicked:
    with st.spinner('Looking for similar games...'):
        if usemygamename:
            mygamename,qltynum = get_real_name_fuzzy(usergamename,finalgamelist_df)
        else:
            mygamename = mydemogamename
                
          # FILTERS
        filt_dict = {'min_rating':min_rating,'min_players':min_players,'min_dur':min_dur,'min_numraters':defaultnumraters}
 
        mygamename,qltynum = get_real_name_fuzzy(mygamename,finalgamelist_df)
        print('Best guess: {} (match score: {}/100)'.format(mygamename,str(qltynum)))
        mygameid = list(finalgamelist_df.index[finalgamelist_df['game_name']==mygamename])[0] # Need INDEX, not idx
        mygameurl=list(allgamedata_df.loc[allgamedata_df['game_name']==mygamename,'bgg_url'])[0]
        
        # PRepare and write out teh chosen game:
        mygamename_st_url = f'<a target="_blank" href="{mygameurl}">{mygamename}</a>'
        if usemygamename: # If text used, indicate this is a guess:
            isguesstext = ' (best guess) '
        else:
            isguesstext = ' '
        st.write('Games similar to' + isguesstext + mygamename_st_url, unsafe_allow_html=True)

      

        # Make two separate TOP lists:
        # FIRST: SEMANTIC
        W1=1 # Semantic
        W2=0 # Feature

        mytop10simlist_df = getcompute_similar_games(mygameid,mygamename,allgamedata_df,allgamedocvects,finalgamelist_df,bgg_gameplay_df, allgamePLAYdocvects,W1,W2,filt_dict)
        mygamevect_df_SEM = streamlitify_df(mytop10simlist_df)
        mygamevect_df_SEM = mygamevect_df_SEM.iloc[0:5,:]
        
        # SECOND: FEATURAL
        W1=0 # Semantic
        W2=1 # Feature
        mytop10simlist_df = getcompute_similar_games(mygameid,mygamename,allgamedata_df,allgamedocvects,finalgamelist_df,bgg_gameplay_df, allgamePLAYdocvects,W1,W2,filt_dict)
        mygamevect_df_FTR = streamlitify_df(mytop10simlist_df)
        mygamevect_df_FTR = mygamevect_df_FTR.iloc[0:5,:]
        
        mygamevect_df = pd.DataFrame({'Conceptually similar games':mygamevect_df_SEM['Game link'],'Games with similar features':mygamevect_df_FTR['Game link']})

        #st.write('**Conceptually closest games**') 
        #st.write(mygamevect_df_SEM[['Game link']].to_html(escape = False), unsafe_allow_html = True)
        #st.write('**Games with similar features**') 
        #st.write(mygamevect_df_FTR[['Game link']].to_html(escape = False), unsafe_allow_html = True)
        
        st.write(mygamevect_df.to_html(escape = False), unsafe_allow_html = True)
        
    st.success('Done!')

#sns.heatmap(np.random.randn(10,10))
#st.pyplot()
