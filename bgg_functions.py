def sanitize_text(descr):
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        from nltk.tokenize import word_tokenize
        from nltk.tokenize import word_tokenize
        import re

        lemmatizer = WordNetLemmatizer()

        #stop_words = set(stopwords.words('english')) 

        # CLEAN
        descr = descr.lower()

        # TOKENIZE
        descr = word_tokenize(descr)

        descr = [re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem) for elem in descr]
        #descr.apply(lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))  
        # remove numbers
        #descr = descr.apply(lambda elem: re.sub(r"\d+", "", elem))

        # DISINFECT - aka TOKENIZATION
        descr = [word_tokenize(elem) for elem in descr]

        # Remove STOP words
        descr = [ww for ww in descr if not ww in stopwords.words()]

        # Re-form the list, remoaving empty cells because of stopwprd removal
        descr = [dum[0] for dum in descr if len(dum)>0]

        # CAUTERIZE - aka LEMMATIZATION
        descr = [lemmatizer.lemmatize(ww) for ww in descr]
        
        return descr
    
def description2vector(desc):
    import pandas as pd
    from gensim.models import Word2Vec
    from nltk.corpus import stopwords
    

    def desc2docvect(currdoc,words):
        ''' Get the document vector from a description'''
        gamedocvects_df = pd.DataFrame({})
        numw = len(currdoc)
        docvect_df = pd.DataFrame({})
        for w in range(0,numw): # Over all words in the doc
            if currdoc[w] in words:
                docvect_df = pd.concat((docvect_df,pd.DataFrame(bgg_model.wv[currdoc[w]])),axis=1)
        gamedocvects_df = pd.concat((gamedocvects_df,docvect_df.mean(axis=1)),axis=1)
        return gamedocvects_df

    ##### INITIALIZE ####
    # Get the word2vec model
    bgg_model = Word2Vec.load('datasources/bgg_w2v_model.bin')
    words = list(bgg_model.wv.vocab)
    # Sanitize text
    desc = sanitize_text(desc)
    # Get the vector
    dv = desc2docvect(desc,words)
    
    return dv


def vector2similargameurls(docvect):
    import numpy as np
    import pandas as pd
    import sklearn.metrics.pairwise as sklpw
    
    ##### GET THE DATA
    # Game data
    allgamedata_df = pd.read_pickle('datasources/BGG_FINAL.pkl') # USE ONLY FOR URLS
    allgamedata_df = allgamedata_df.astype({'game_rank':'int32'},copy=True)
    finalgamelist_df = pd.read_pickle('datasources/BGG_GameSimilarityKey.pkl')
    finalgamelist_df.reset_index(drop=True,inplace=True) # So that row ids are indices to gamevector array
    # Semantic game vectors
    allgamedocvects = np.load('datasources/allgamedocvects.npz')['arr_0']

    # Fix the input vector
    mygamevector = np.array(docvect.iloc[:,0])
    mygamevector= mygamevector.reshape(-1,1)


    ##### SEMANTICS BASED
    mysimilarities = []
    for t in range(0,allgamedocvects.shape[0]):
        currgamevect = allgamedocvects[t,:]
        currgamevect = currgamevect.reshape(-1,1)
        dum = sklpw.cosine_similarity(currgamevect.T,mygamevector.T)
        mysimilarities.append(dum[0][0])
    mycompletesimlist_df = pd.concat((finalgamelist_df['game_rank'],finalgamelist_df['game_name'],pd.DataFrame({'Similarity':mysimilarities})),axis=1)
    mytop10simlist_df = mycompletesimlist_df.copy()
    mytop10simlist_df.sort_values(by='Similarity',ascending=False)
    mytop10simlist_df = mytop10simlist_df[1:11]


    # SORT Etc
    mycompletesimlist_df = pd.concat((finalgamelist_df['game_rank'],finalgamelist_df['game_name'],pd.DataFrame({'Similarity':mysimilarities})),axis=1)
    mytop10simlist_df = mycompletesimlist_df.copy()
    mytop10simlist_df.sort_values(by='Similarity',ascending=False,inplace=True)
    mytop10simlist_df = mytop10simlist_df[1:11]

    # Create output list
    urllist=[]
    for gamename in mytop10simlist_df['game_name']:
        urllist.append(list(allgamedata_df.loc[allgamedata_df['game_name']==gamename,'bgg_url'])[0])
    mytop10simlist_df = pd.DataFrame({'Game':mytop10simlist_df['game_name'],'url':urllist})
    mytop10simlist_df.reset_index(drop=True,inplace=True)
    mytop10simlist_df.index = mytop10simlist_df.index+1

    return mytop10simlist_df