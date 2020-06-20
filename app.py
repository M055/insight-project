# VERSION V4BK
# Testing tabs
import pandas as pd
import numpy as np
import streamlit as st
from streamlit.ReportThread import get_report_ctx
from streamlit.hashing import _CodeHasher
from streamlit.server.Server import Server
import sklearn.metrics.pairwise as sklpw
from PIL import Image
from fuzzywuzzy import fuzz
import re


### THE WHOLE STATE THING:
class _SessionState:

    def __init__(self, session):
        """Initialize SessionState instance."""
        self.__dict__["_state"] = {
            "data": {},
            "hash": None,
            "hasher": _CodeHasher(),
            "is_rerun": False,
            "session": session,
        }

    def __call__(self, **kwargs):
        """Initialize state data once."""
        for item, value in kwargs.items():
            if item not in self._state["data"]:
                self._state["data"][item] = value

    def __getitem__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)
        
    def __getattr__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)

    def __setitem__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value

    def __setattr__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value
    
    def clear(self):
        """Clear session state and request a rerun."""
        self._state["data"].clear()
        self._state["session"].request_rerun()
    
    def sync(self):
        """Rerun the app with all state values up to date from the beginning to fix rollbacks."""

        # Ensure to rerun only once to avoid infinite loops
        # caused by a constantly changing state value at each run.
        #
        # Example: state.value += 1
        if self._state["is_rerun"]:
            self._state["is_rerun"] = False
        
        elif self._state["hash"] is not None:
            if self._state["hash"] != self._state["hasher"].to_bytes(self._state["data"], None):
                self._state["is_rerun"] = True
                self._state["session"].request_rerun()

        self._state["hash"] = self._state["hasher"].to_bytes(self._state["data"], None)


def _get_session():
    session_id = get_report_ctx().session_id
    session_info = Server.get_current()._get_session_info(session_id)

    if session_info is None:
        raise RuntimeError("Couldn't get your Streamlit Session object.")
    
    return session_info.session


def _get_state():
    session = _get_session()

    if not hasattr(session, "_custom_session_state"):
        session._custom_session_state = _SessionState(session)

    return session._custom_session_state







############# REAL START HERE!
state = _get_state()

######## INITIALIZE SOME VARIABLES
demogamelist_manual = ['Azul', 'Catan','Carcassonne','Cards Against Humanity', 'Clue','Pandemic', 'Scrabble','Taboo' ]
demo_gamelist = tuple(demogamelist_manual)


####### INTRO
st.title('Meeple for People')
st.header('**MEANINGful recommendations for all board gamers**')
meeple_image = Image.open('other/meeple.png')


def page_first(state):
    st.header("Select a game from the menu:")
    # SHOW SOME STUFF
    #state.mydemogamename = st.selectbox('Choose a game',demo_gamelist)
    state.mygamename = st.selectbox('',demo_gamelist)
    state.usemygamename = False
    st.markdown('** Select a game from the dropdown list of popular games (change the selection method from the side panel). Press _Go_ to search **')

def page_second(state):
    st.header("Enter a game name:")
    #state.usergamename = 'Taboo'
    #state.usergamename = st.text_input('Enter game name', 'Taboo',max_chars=30)
    state.mygamename = 'Taboo'
    state.mygamename = st.text_input('', 'Taboo',max_chars=30)
    state.usemygamename = True
    st.markdown('** Enter a game name in the text box to launch a fuzzy search (change the selection method from the side panel). Press _Go_ to search **')
 
  
pages = {
    "Choose a game from the list": page_first,
    "Enter a game name": page_second,
}


##### INIT SIDEBAR
st.sidebar.image(meeple_image, caption='meeple',width=100)
st.sidebar.markdown('**Mee.ple** _noun_ \n a small figure used as a playing piece in certain board games, having a stylized human form.')

page = st.sidebar.radio("Select your input method", tuple(pages.keys()))
# Display the selected page with the session state
pages[page](state)

######## OTHER SIDEBAR STUFF
st.sidebar.header('What do these recommendations mean?')
st.sidebar.markdown('These recommendations balance **Conceptually similar games**:  games that, as a whole, are similar to your target game. For example, you would summarize them in very similar ways when describing them to your friend, and **Games with similar features**: games that are similar to your target game in terms of specific features such as the type and genre of the game, or the mechanics it employs.')



############# LOAD DATA
allgamedata_df = pd.read_pickle('datasources/bgg_filters.pkl') # USE FOR URLS and FILTERS
allgamedata_df = allgamedata_df.astype({'game_rank':'int32'},copy=True)
allgamedocvects = np.load('datasources/allgamedocvects_v3.npz')['arr_0']
finalgamelist_df = pd.read_pickle('datasources/finalgamelist_df.pkl')
#finalgamelist_df =  finalgamelist_df.astype({'game_rank':'int32'},copy=True)
#finalgamelist_df['num_raters'] = [list(allgamedata_df.loc[allgamedata_df['game_name']==g,'num_raters'])[0] for g in #finalgamelist_df['game_name']]
#finalgamelist_df.reset_index(drop=True,inplace=True) # So that row ids are indices to gamevector array
##### FOr gameplay-based vectors
bgg_gameplay_df = pd.read_pickle('datasources/bgg_gameplayfeatures.pkl')
bgg_gameplay_df.dropna(inplace=True) # Some not-NAs here...
bgg_gameplay_df.reset_index(drop=True,inplace=True)
allgamePLAYdocvects = np.array(bgg_gameplay_df.iloc[:,1:]) # Create right here




########## SET VARIABLES
min_rating = 2
min_players = 1 #st.sidebar.slider('Minimum number of players', min_value=1, max_value=10, value=1, step=1) 
min_dur = 1 #st.sidebar.slider('Minimum time to play (min)', min_value=1, max_value=100, value=1, step=1) 
defaultnumraters = 2 # At least so many raters (log10)





######### MAIN PART

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
    
     #MO: Jun 18: Add in SUPECOMBO. Sort right after
    avgratingfactor = myFINALsimlist_df.iloc[0,:]['avg_rating']/10 # SCALE by rating of game?
    dumsupercombo = np.array(myFINALsimlist_df['Similarity'] + myFINALsimlist_df['GameplaySimilarity'] + (myFINALsimlist_df['avg_rating']/10)*avgratingfactor + np.log10(myFINALsimlist_df['num_raters'])/5)/4
    myFINALsimlist_df['supercombo'] = dumsupercombo
    myFINALsimlist_df.sort_values(by='supercombo',inplace=True,ascending=False)
    myFINALsimlist_df.reset_index(drop=True,inplace=True)
    myFINALsimlist_df.drop(index=0,inplace=True) # DROP THE MAIN COMPARISON GAME ALREADY
    myFINALsimlist_df.reset_index(drop=True,inplace=True)
    
    # Create output list
    # MO: New Jun 13 - check to make sure there are enough games!
    # MO: June 18: return FULL LIST, compute later
    if len(myFINALsimlist_df)>11:    
        mytop10simlist_df = myFINALsimlist_df[:10].copy()
    else:
        mytop10simlist_df = myFINALsimlist_df.copy()
    urllist=[]
    for gamename in mytop10simlist_df['game_name']:
        urllist.append(list(allgamedata_df.loc[allgamedata_df['game_name']==gamename,'bgg_url'])[0])
    #mytop10simlist_df = pd.DataFrame({'Game':mytop10simlist_df['game_name'],'Similarity':mytop10simlist_df['WghtdSimilarity'],'url':urllist})
    mytop10simlist_df = pd.DataFrame({'Game':mytop10simlist_df['game_name'],'url':urllist})
    mytop10simlist_df.reset_index(drop=True,inplace=True)
    mytop10simlist_df.index = mytop10simlist_df.index+1
    
    return mytop10simlist_df,myFINALsimlist_df
    
    
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

    
    
    

############# WHEN YOU CLICK THE BUTTON...
clicked = st.button('Go')
if clicked:
    with st.spinner('Looking for similar games...'):
        #if state.usemygamename:
        #    mygamename,qltynum = get_real_name_fuzzy(state.usergamename,finalgamelist_df)
        #else:
        #    mygamename = state.mydemogamename
                
          # FILTERS
        
        mygamename = state.mygamename
        filt_dict = {'min_rating':min_rating,'min_players':min_players,'min_dur':min_dur,'min_numraters':defaultnumraters}
 
        mygamename,qltynum = get_real_name_fuzzy(mygamename,finalgamelist_df)
        print('Best guess: {} (match score: {}/100)'.format(mygamename,str(qltynum)))
        mygameid = list(finalgamelist_df.index[finalgamelist_df['game_name']==mygamename])[0] # Need INDEX, not idx
        mygameurl=list(allgamedata_df.loc[allgamedata_df['game_name']==mygamename,'bgg_url'])[0]
        
        # PRepare and write out teh chosen game:
        mygamename_st_url = f'<a target="_blank" href="{mygameurl}">{mygamename}</a>'
        # Prepare indication for match quality
        st.markdown("<style>great{color:green} fair{color:blue} poor{color:orange} terrible{color:red}</style>",unsafe_allow_html=True)
        qltys = ['great', 'fair', 'poor','terrible']
        #x = f"Match quality: <{qltys[1]}>{qltys[1]}</{qltys[1]}>"
        #st.markdown(x, unsafe_allow_html=True)

        if qltynum>95:
            qltytext = f". Match quality: <{qltys[0]}>{qltys[0]}</{qltys[0]}>"
        elif (qltynum>80) & (qltynum<=95):
            qltytext = f". Match quality: <{qltys[1]}>{qltys[1]}</{qltys[1]}>"
        elif (qltynum>50) & (qltynum<=80):
            qltytext = f". Match quality: <{qltys[2]}>{qltys[2]}</{qltys[2]}>"
        else:
            qltytext = f". Match quality: <{qltys[3]}>{qltys[3]}</{qltys[3]}>"
        
        
        if not state.usemygamename: # If text used, indicate this is a guess:
            qltytext = '.'
        st.markdown('Games similar to ' + mygamename_st_url + qltytext, unsafe_allow_html=True)
        

        # Make two separate TOP lists:
        # FIRST: SEMANTIC
        W1=1 # Semantic
        W2=0 # Feature

        mytop10simlist_df,myFINALsimlist_df = getcompute_similar_games(mygameid,mygamename,allgamedata_df,allgamedocvects,finalgamelist_df,bgg_gameplay_df, allgamePLAYdocvects,W1,W2,filt_dict)
        mygamevect_df = streamlitify_df(mytop10simlist_df)
        dumtop10 = myFINALsimlist_df.copy().reset_index(drop=True)[:10]
        dumtop10.index = dumtop10.index+1
        mygamevect_df = pd.concat((mygamevect_df['Game link'],dumtop10[['avg_rating','num_raters','numplayersmin','gamedurmin']]),axis=1)
        mygamevect_df['num_raters']=np.round((np.log10(mygamevect_df['num_raters'])/5)*10).astype('int32')
        mygamevect_df['numplayersmin']=mygamevect_df['numplayersmin'].astype('int32')
        mygamevect_df['gamedurmin']=mygamevect_df['gamedurmin'].astype('int32')
        mygamevect_df.rename(columns={'Game_link':'Game','avg_rating':'Avg. Rating*','num_raters':'Popularity**','numplayersmin':'Min. Number of Players','gamedurmin':'Min. Game Duration (min)'},inplace=True)
        
        st.write(mygamevect_df.to_html(escape = False), unsafe_allow_html = True)
        bggcom_url =  f'<a target="_blank" href="https://boardgamegeek.com/">BoardGameGeek.com </a>'
        st.write(f'*Average ratings from ' + bggcom_url + ', from 0 (worst) to 10 (best)', unsafe_allow_html = True)
        st.write(f'**Popularity goes from 0 (least popular) to 10 (most popular)', unsafe_allow_html = True)
        
    st.success('Done!')

#state.sync()