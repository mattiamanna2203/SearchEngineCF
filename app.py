from flask import Flask, render_template, request, flash
# region Pacchetti
import pandas as pd # Lavorare su dataframes

import json         # Lavorare con file json
import re           # Lavorare con regular expression
import numpy as np  # Lavorare con matrici 

# Pacchetti per calcolare i valori  tf-idf
from sklearn.feature_extraction.text import TfidfVectorizer # TF IDF
from sklearn.metrics.pairwise import cosine_similarity      # Cosine similarity


from collections import Counter # Dizionario speciale
from functools import reduce

import math

# Pacchetti per lavorare sul testo 
import spacy                              # Per lemmatizzazione
nlp = spacy.load('it_core_news_sm') # Caricare lemmatizzazione italiana

from nltk.stem import SnowballStemmer # Stemming
stemmer_snowball = SnowballStemmer('italian') # Definire lo stemmer per la lingua italiana

import unicodedata # Questo pacchetto serve per normalizzare le stringhe.  
                   # Ad esempio le due stringhe "l'amore è cieco"=="l'amore è cieco"
                   # risultavano diverse, alcuni caratteri nascosti le facevano sembrare diverse in termini di significato
                   # per un umano fossero uguali (si possono capire le differenze tramite il seguente for loop)
                   # for a, b in zip(r"l'amore è cieco", r"l'amore è cieco"):
                   #    print(a, b, a == b)  
                   # Tramite il pacchetto unicodedata si possono normalizzare le stringhe e renderle uguali
# Importare le funzioni

import sys
import os


from lemmatizzatore import lemmatizzazione
from SearchEngine_class import SearchEngine

# endregion


search_class = SearchEngine("/Dati")



# Create a class for our APP
app = Flask(__name__)
app.secret_key = "una_password_random"



@app.route("/main")
def hello():
    return render_template("index.html")


# Select route for the app
@app.route("/search", methods=["POST","GET"])
def search():
    richiesta = request.form.get('submit_action')
    print(richiesta)
    if richiesta == "reset":
        tabella_html = "<p id='reset_message' class='center-text'>Effettua una ricerca per vedere i risultati.</p>"
        return render_template("index.html",seasons_selected=[], 
                                            characters_selected=[],
                                            output_length_selected = 1765,
                                            campi_ricerca_selected = "ALL", 
                                            guest_stars_selected = [], 
                                            all_match_or_ranking_selected ="ranking_matching",
                                            tabella_html=tabella_html)





 
    query = request.form.get('query','')  #Ottieni il valore dell'input
    seasons = request.form.getlist('season')  # Liste delle stagioni selezionate
    guest_stars = request.form.getlist('guest_star')  # Liste delle guest star selezionate
    characters = request.form.getlist('character')  # Liste dei personaggi selezionati
    num_results = int(request.form.get('num_results', 20))  # Numero di risultati
    search_type = request.form.get('search_type', 'ALL')  # Tipo di ricerca
    all_match_or_ranking = request.form.get('all_match_or_ranking', 'all_matching')  # Tipo di ricerca
    print(query,seasons,characters,guest_stars,num_results,search_type,all_match_or_ranking)


 

    # Trasformazione degli output html per renderli compatibili alle funzioni python.
    if (len(seasons) < 1):
        seasons = None
        seasons_selected = []


    elif len(seasons) >= 1:
        seasons_selected = [str(i) for i in seasons]
        seasons = [int(i) for i in seasons]

    if len(characters) < 1:
        characters = None
        characters_selected = []
    else:   
        characters_selected = characters


    if len(guest_stars) < 1:
        guest_stars = None
        guest_stars_selected = []
    else:
        guest_stars_selected=guest_stars

    print(query,seasons,characters,guest_stars,num_results,search_type,all_match_or_ranking)
   # Con questo metodo di ricerca si consiglia di inserire i personaggi tramite l'apposito menu e non scrivergli
    if all_match_or_ranking == "all_matching":
        risultato_ricerca = search_class.all_matching( query_str= query,
                            campi_ricerca= search_type, # ALL, title_trama, copione
                            output_length = num_results,     # Numeri interi, None
                            season=seasons, # 1,2,3,4,5,6 [1,2], [4,5], None
                            personaggi_apparsi = characters, #["Luca","De Marinis","Ilaria","Gaia"], # Lista dei personaggi, ["Luca","Paolo","Giovanna"]
                            gueststar_apparse =  guest_stars#["Massimo Ranieri"] # Lista delle guest star ["",""]
                        ) 

    elif all_match_or_ranking == "ranking_matching":
        risultato_ricerca = search_class.ranking_matching( query_str= query,
                    campi_ricerca= search_type, # ALL, title_trama, copione
                    output_length = num_results,     # Numeri interi, None
                    season=seasons, # 1,2,3,4,5,6 [1,2], [4,5], None
                    personaggi_apparsi = characters, #["Luca","De Marinis","Ilaria","Gaia"], # Lista dei personaggi, ["Luca","Paolo","Giovanna"]
                    gueststar_apparse =  guest_stars#["Massimo Ranieri"] # Lista delle guest star ["",""]
                ) 

    if isinstance(risultato_ricerca, pd.DataFrame) and not risultato_ricerca.empty:
            # Converte il DataFrame in una tabella HTML con classi Bootstrap
            tabella_html = risultato_ricerca.to_html(
                index=False,
                classes='table table-striped table-bordered',  # Aggiungi classi per lo stile
                escape=False
            )
    else:
        # Messaggio di fallback in caso di risultati vuoti
        tabella_html = "<p class='center-text'>Nessun risultato trovato</p>"


  
    return render_template("index.html",seasons_selected=seasons_selected, 
                                        characters_selected=characters_selected,
                                        output_length_selected = num_results,
                                        campi_ricerca_selected = search_type, 
                                        guest_stars_selected =guest_stars_selected,
                                        all_match_or_ranking_selected = all_match_or_ranking,
                                        tabella_html=tabella_html)


