
# Meeple for People


In this project I built a board game recommendation system that combines a more "traditional" feature-based recommendation with an NLP-driven, *semantics-based* recommendation.

[Click here to try out some recommendations here!](https://meeple4people.herokuapp.com/)


#### What are meeple?
![A meeple](https://github.com/M055/insight-project/blob/master/other/meeple_sm.png)
[Meeple:](https://en.wiktionary.org/wiki/meeple) - are small person-shaped figures used as players' tokens in some board games.


# The context of this project.

Quite by chance, you and your friends recently discovered the game [Wingspan](https://www.boardgamegeek.com/boardgame/266192/wingspan) and enjoyed playing it. You want to find other games that are similar to Wingspan. What does *similar* mean?

What you are looking for is a **recommender**. Traditionally, recommenders rely on patterns of engagements between users and products (movies, books, etc), and on surface features of products such as genre, length, or author.

In this project, I explore a new kind of recommender, which looks at similarities between products -- games -- in terms of similarities in how you would describe a game to someone else. That is, I am interested in building a recommender that can understand something about the semantics (meaning) of descriptions of different games, and determine similarity in relation to the similarity of these descriptions.


# Calculating semantic similarity between games.

1. I scraped [BoardGameGeek](https://boardgamegeek.com/) and retrieved over 18,000 games – essentially, all games with a rank on BoardGameGeek.

2. I retrieved both surface features (themes, mechanics, clarifications) for each game, and game descriptions.

3. I used a combination of the python packages [NLTK](https://www.nltk.org/) and [Gensim](https://radimrehurek.com/gensim/) to create an NLP-ready version of game descriptions (normalizing case, removing non-alphanumeric characters, tokenization, stop-word removal, lemmatization). Then, I used Genesim to create a 100-element word2vec embedding of all the descriptions.

4. Each game was converted into a single 100-element vector by averaging across all the words in its description, and the similarity between games was calculated as the cosine similarity between them.


## Accounting for other game features.

I extracted features relating to gameplay, mechanics, themes, etc., and cleaned these to get a set of 127 'surface' features across all the games. Games were also projected into this feature-space.    
Additionally, I found that the average game score influences the likelihood that a user knows a game at all, or is interested in a game, so I included two measures of a game's popularity and success – its average game score, and its popularity, as measured by the (log10 of the) number of users who rated a game.


# Making recommendations.

For a given target game, I can calculate the (cosine) similarity of the target and the remaining games. This can be done for both the semantic-vector of the target game and its feature-vector. These similarities can then be combined. to determine a ranking of games from most to least similar.    
The combined similarity value was then modulated by normalized average score and log10(number of reviewers). Games were ranked based on this final metric, and the top ranking games are the ones that the app presents.

## Testing the validity of these recommendations.

A survey of the boardgames subreddit found that self-identified boardgamers judged the output of both the semantic and the feature model as appropriate for the target game.

I also scraped user information from BoardGameGeek, and found that the top 3 games of an individual user predicted their ratings of their remaining games, compared to a random model.

# More information.

If you would like to know more, you can check out the [slides](https://docs.google.com/presentation/d/1WjwLGVVUO2Jj42QNX5rg2alfR9mkni4qfI9egbTCUZw/edit?usp=sharing), or watch a [video presentation](https://www.youtube.com/watch?v=S-LlWeb5nK4) of the slides.
#### Connect with me on [LinkedIn](https://www.linkedin.com/in/mohinishs/)
