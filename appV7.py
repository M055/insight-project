# VERSION V6
# Adding SQL for everything
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
import os
import psycopg2
import requests


#Version note:
# Updated to bgg_filters_jun20.pkl from sqlify_data

### SET UP FUNCTIONS FOR HANDLING STATE-SPECIFIC VARIABLES:
# Via streamlit user Synode
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







############# STORE INITIAL STATE
state = _get_state()

######## INITIALIZE SOME VARIABLES
demogamelist_manual = ['Azul', 'Catan','Carcassonne','Cards Against Humanity', 'Clue','Pandemic', 'Scrabble','Taboo' , 'Wingspan']
demo_gamelist = tuple(demogamelist_manual)


####### INTRO
st.title('Meeple for People')
st.header('**MEANINGful recommendations for all board gamers**')
meeple_image = Image.open('other/meeple.png')


#### SETUP STATE-SPECIFIC CONTENT
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
st.sidebar.image(meeple_image, caption='meeple*',width=50)
st.sidebar.markdown('***Mee.ple** _noun_ \n a small figure used as a playing piece in certain board games, having a stylized human form.')
page = st.sidebar.radio("Select your input method", tuple(pages.keys()))
# Display the selected page with the session state
pages[page](state)


######## OTHER SIDEBAR STUFF
st.sidebar.header('What do these recommendations mean?')
st.sidebar.markdown('These recommendations balance **Conceptually similar games**:  games that, as a whole, are similar to your target game. For example, you would summarize them in very similar ways when describing them to your friend, and **Games with similar features**: games that are similar to your target game in terms of specific features such as the type and genre of the game, or the mechanics it employs.')


st.sidebar.markdown(f'[Codebase](https://github.com/M055/insight-project)')
st.sidebar.markdown(f'[Slides](https://docs.google.com/presentation/d/1WjwLGVVUO2Jj42QNX5rg2alfR9mkni4qfI9egbTCUZw/edit?usp=sharing)')
st.sidebar.markdown(f'[Connect with me on LinkedIn!](https://www.linkedin.com/in/mohinishs/)')



############# LOAD DATA
@st.cache(suppress_st_warning=True,show_spinner=False)
def load_boardgame_data():
    ## READ environ vars
    # Retrieve data from AWS db
    dbname = 'm4pdb' 
    username = os.environ["AWSUSR"]
    mypswd = os.environ["AWSPWD"]
    con = psycopg2.connect(database = dbname, user = username, password= mypswd,host='meeps4peeps-db.ckzlat62o0dz.us-east-1.rds.amazonaws.com')
    # query:
    sql_query = """
    SELECT * FROM bgg_filters_table;
    """
    allgamedata_df = pd.read_sql_query(sql_query,con)

    finalgamelist_df = pd.read_pickle('datasources/finalgamelist_df.pkl')


    return allgamedata_df, finalgamelist_df

allgamedata_df, finalgamelist_df = load_boardgame_data()


########## SET VARIABLES
min_rating = 2
min_players = 1 #st.sidebar.slider('Minimum number of players', min_value=1, max_value=10, value=1, step=1) 
min_dur = 1 #st.sidebar.slider('Minimum time to play (min)', min_value=1, max_value=100, value=1, step=1) 
defaultnumraters = 2 # At least so many raters (log10)





######### MAIN PART

# FUNCTIONS

sim_sql_dict = {
    'dbname': 'm4p_db',
    'username': os.environ["AWSUSR"],
    'mypswd': os.environ["AWSPWD"]
}

def get_cosims(n,sim_sql_dict, currtablename):
    # Define SQL query
    sql_query1 = "SELECT * FROM " + currtablename + " WHERE (grank1=" + str(n) +  "); "
    sql_query2 = "SELECT * FROM " + currtablename + " WHERE (grank2=" + str(n) +  "); "
    #sql_query = " SELECT * FROM " + currtablename + " WHERE (grank1=" + str(currgamerank4sql) + " AND grank2>=" + str(currgamerank4sql) + ") OR (grank1<" + str(currgamerank4sql) + " AND grank2=" + str(currgamerank4sql) + ");"
    # Make connection
    con = psycopg2.connect(database = sim_sql_dict.get('dbname'), user = sim_sql_dict.get('username'), password=sim_sql_dict.get('mypswd'), host='meeps4peeps-db.ckzlat62o0dz.us-east-1.rds.amazonaws.com')
    dumdf1 = pd.read_sql_query(sql_query1,con)
    dumdf2 = pd.read_sql_query(sql_query2,con)
    dumdf = pd.concat((dumdf1,dumdf2))
    dumdf.drop_duplicates(inplace=True)
    rankseq = pd.DataFrame([[a,c] if a<n else [b,c] for a,b,c in zip(list(dumdf['grank1']),list(dumdf['grank2']),list(dumdf['cosim']))])
    rankseq.rename(columns={0:'game_rank',1:'cosim'},inplace=True)
    #rankseq.sort_values(by='cosim',inplace=True,ascending=False)
    #rankseq.reset_index(drop=True, inplace=True)
    return rankseq


def getcompute_similar_by_gameplay(mygamerank,sim_sql_dict):
    # Get gameplay similarties
    mycompleteGPsimlist_df = get_cosims(mygamerank,sim_sql_dict,'ftrsim_table')
    mycompleteGPsimlist_df.rename(columns={'cosim':'GameplaySimilarity'},inplace=True)    

    return mycompleteGPsimlist_df

def getcompute_similar_games(mygamerank,sim_sql_dict,W1,W2,filt_dict):
    # Get semantic siilarities and fold in with other values
    simrankseq = get_cosims(mygamerank,sim_sql_dict,'semsim_table')
    mycompletesimlist_df=simrankseq.merge(allgamedata_df[['game_rank','game_name','num_raters']],how='left',on='game_rank')
    mycompletesimlist_df.rename(columns={'cosim':'Similarity'},inplace=True)
  
    # Get this also for GAMEPLAY data
    mycompleteGPsimlist_df = getcompute_similar_by_gameplay(mygamerank,sim_sql_dict)

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
    mygamerank = finalgamelist_df.iloc[possiblegame_idx,1]
    return possiblegame_name,mygamerank,max(gamename_matchlist)

def make_clickable(url,text): # Make liks in pd df for url in table
    return f'<a target="_blank" href="{url}">{text}</a>'

def streamlitify_df(df):
    # Get original URLS
    df['Game link'] = [make_clickable(a,b) for a,b in zip(list(df['url']),list(df['Game']))]
    return df

def get_random_question():
    qurl = 'https://opentdb.com/api.php?amount=1&category=16'
    response = requests.get(qurl)
    rr = response.json()
    rr = rr['results'][0]
    rqstn = rr['question']
    rqstn = re.sub('&quot;', '', rqstn)
    ranwr = rr['correct_answer']
    ranwr = re.sub('&quot;', '', ranwr)
    rincr = rr['incorrect_answers']
    rincr = [re.sub('&quot;', '', x) for x in rincr]
    rincr.insert(np.random.randint(0,len(rincr)),ranwr)

    return rqstn, ranwr, rincr # rincr has all answers
    
    
    
    

############# WHEN YOU CLICK THE BUTTON...
clicked = st.button('Go')
Qtext='RANDOM QUESTION WHILE YOU WAIT: '
Atext='ANSWER: '


if clicked:
    with st.spinner('Looking for similar games. (This may take up to half a minute)'):
        # Create random q/a
        rqstn, ranwr, rincr = get_random_question()
        #opttext = ['('+str(a+1)+') '+b for a,b in enumerate(rincr)]
        opttext = [b+'...' for a,b in enumerate(rincr)]
        myinfo1 = st.info(Qtext + rqstn + '  CHOICES: ' + ' '.join(opttext[:]))
        
        
        # FILTERS
        filt_dict = {'min_rating':min_rating,'min_players':min_players,'min_dur':min_dur,'min_numraters':defaultnumraters}
        # Get current name - whether from dropdown or entered text
        mygamename = state.mygamename
        mygamename,mygamerank,qltynum = get_real_name_fuzzy(mygamename,finalgamelist_df)
        #print('Best guess: {} (match score: {}/100)'.format(mygamename,str(qltynum))) # For testing
        mygameid = list(finalgamelist_df.index[finalgamelist_df['game_name']==mygamename])[0] # Need INDEX, not idx
        mygameurl=list(allgamedata_df.loc[allgamedata_df['game_name']==mygamename,'bgg_url'])[0]
        
        # PRepare and write out teh chosen game:
        mygamename_st_url = f'<a target="_blank" href="{mygameurl}">{mygamename}</a>'
        # Prepare indication for match quality
        st.markdown("<style>great{color:green} fair{color:blue} poor{color:orange} terrible{color:red}</style>",unsafe_allow_html=True)
        qltys = ['great', 'fair', 'poor','terrible']

        # Prepare match quality indicator text
        if qltynum>95:
            qltytext = f". Match quality: <{qltys[0]}>{qltys[0]}</{qltys[0]}>"
        elif (qltynum>80) & (qltynum<=95):
            qltytext = f". Match quality: <{qltys[1]}>{qltys[1]}</{qltys[1]}>"
        elif (qltynum>50) & (qltynum<=80):
            qltytext = f". Match quality: <{qltys[2]}>{qltys[2]}</{qltys[2]}>"
        else:
            qltytext = f". Match quality: <{qltys[3]}>{qltys[3]}</{qltys[3]}>"
        
        
        if not state.usemygamename: # ONLY if text used, indicate this is a guess:
            qltytext = '.'
            
        # Give the anwer to the question first:
        myinfo2 = st.info(Atext + ranwr)
        
        # Show bestmatch to entered game
        st.markdown('Games similar to ' + mygamename_st_url + qltytext, unsafe_allow_html=True)
        
        
        

        # Weights (NB: compute but do not show now)
        W1=1 # Semantic
        W2=0 # Feature

        mytop10simlist_df,myFINALsimlist_df = getcompute_similar_games(mygamerank,sim_sql_dict,W1,W2,filt_dict)
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
        
        myinfo1.empty()
        myinfo2.empty()
    st.success('Done!')

#state.sync()