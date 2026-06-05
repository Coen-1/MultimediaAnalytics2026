



# How to use
- .venv/bin/python precompute.py
- .venv/bin/python app.py 





# Project description

This project is a visualisation of a lat lon location in embedding spaces
- we do this my having a map in the middle of the application
- on the map we can pick a location (the map shows the pdok areal images)

- there are scatter plots of embeddings on the right side of the screen:
    - once this happens we see a scatter plot of the PCA components in 2d of the clip embedding(the one finetuned for earth observation stuff)
    - we also have a dataset of textual descriptions of all the neighborhoods and we want to embed these and do the same as the clip embeddings (pca scatter plot)
    - maybe also a scatter plot for the cbs learned dense vector if possible.

- on the left side, to give the user trust, we show a table and a spider plot with the cbs data in it.

- when the user clicks on one of the embedding spaces, we see the dots highlight in the other embeddingspace that is most similar and in the current highlighted map,
 we see all the dots in top k places that are similar according to embedding space (both clip and textual)