## Addendum (post-implementation)

get_comps was implemented against a new synthetic sold-comps dataset
(data/sold_comps.json), not the active listings corpus — "comparable"
in real estate means comparable SOLD prices, not comparable current
asking prices. This required creating new synthetic data, since
listings_corpus.md only contains active listings with no sale history.
