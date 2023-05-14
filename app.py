import json
import os

import pandas as pd
import spacy
import streamlit as st
from spacy_langdetect import LanguageDetector
from transformers import (BartForConditionalGeneration, BartTokenizer)

from simplification_model import SimplificationModel
from table_style import styles
from text import LEVELS, Text

checkpoint = "bart-base"
tokenizer_type = BartTokenizer
model_type = BartForConditionalGeneration
model_dir = os.path.join("models", checkpoint)


@spacy.language.Language.factory("language_detector")
def language_detector(nlp, name):
    return LanguageDetector()


@st.cache_resource
def load_wordlists(word2level_file):
    with open(word2level_file, "r") as f:
        word2level = json.load(f)
    return word2level


@st.cache_resource
def load_preprocessing_model():
    try:
        nlp = spacy.load("en_core_web_lg")
    except IOError:
        spacy.cli.download("en_core_web_lg")
        nlp = spacy.load("en_core_web_lg")
    nlp.add_pipe("language_detector", last=True)
    return nlp


@st.cache_resource
def load_simplification_model(tokenizer_type, model_type, model_dir):
    return SimplificationModel(tokenizer_type, model_type, model_dir)


def wordlist2string(wordlist):
    if wordlist:
        return ", ".join(wordlist)
    return "-"


def get_text_info(text):
    word_levels = text.get_words_from_level_lists(word2level)
    text_info = {
        "Mean sentence length (in words)": text.count_mean_sentence_len(),
        "Mean word length (in symbols)": text.count_mean_word_len(),
    }

    for level in LEVELS:
        text_info[level] = wordlist2string(word_levels[level])

    return text_info


def create_df_with_text_info():
    hard_text = Text(st.session_state.text, nlp)
    hard_text_info = get_text_info(hard_text)
    simple_text = Text(st.session_state.simplified, nlp)
    simple_text_info = get_text_info(simple_text)

    df = pd.DataFrame.from_dict(
        {"Original text": hard_text_info, "Simplified text": simple_text_info}
    )

    df_styled = (
        df.style.set_properties(**{"text-align": "left"})
        .set_table_styles(styles)
        .format(precision=2)
    )
    return df_styled


def on_click_simplify():
    st.session_state.text = st_text_area
    if st_text_area:
        if nlp(st.session_state.text)._.language["language"] == "en":
            st.session_state.simplified = simplification_model.simplify(st_text_area)
            right.write(st.session_state.simplified)
            df_styled = create_df_with_text_info()
            bottom.subheader("Text characteristics")
            bottom.table(df_styled)
        else:
            bottom.info("The text must be in English!")
    else:
        bottom.info("Empty field!")


st.set_page_config(page_title="Simplification", layout="wide")
st.header("Simplify English texts")

st_load = st.text("Loading preprocessing model...")
nlp = load_preprocessing_model()

st_load.text("Loading simplification model...")
simplification_model = load_simplification_model(tokenizer_type, model_type, model_dir)

st_load.text("Loading wordlists...")
word2level = load_wordlists(os.path.join("wordlists", "word2level.json"))
st_load.success("Models and wordlists loaded!")

if "text" not in st.session_state:
    st.session_state.text = ""

if "simplified" not in st.session_state:
    st.session_state.simplified = ""


top = st.container()
bottom = st.container()
left, right = top.columns(2, gap="large")

left.subheader("Text to simplify")
st_text_area = left.text_area(
    "Text to simplify",
    placeholder="Input some text to simplify",
    value=st.session_state.text,
    height=240,
    label_visibility="collapsed",
)
right.subheader("Simplified text")

_, _, col, _, _, _, _, _, _, _ = bottom.columns(10)
st_simplify_button = col.button(
    "Simplify", type="primary", use_container_width=True, on_click=on_click_simplify
)
