########################################
# Toxicity
########################################

toxic_prefixes = dict()

toxic_prefixes["neg"] = "The following text does not perpetuate negative stereotypes, is not threatening or sexually explicit, and does not contain profane language.\n\n"
toxic_prefixes["pos"] = 'The following text perpetuates negative stereotypes, is threatening or sexually explicit, or contains profane language.\n\n'

########################################
# Sentiment
########################################

sentiment_prefixes = dict()
sentiment_prefixes["neg"] = "The following text exhibits a very negative sentiment and/or opinion.\n\n"
sentiment_prefixes["pos"] = "The following text exhibits a very positive sentiment and/or opinion.\n\n"

########################################
# General Prefix Parameters
########################################

all_prefixes = {}
all_prefixes["toxicity"] = toxic_prefixes
all_prefixes["sentiment"] = sentiment_prefixes

all_prefix_settings = list(toxic_prefixes.keys()) + list(sentiment_prefixes.keys())
# strengths = [-4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]