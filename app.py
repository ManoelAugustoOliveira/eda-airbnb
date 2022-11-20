# ==================================================  PROJETO EDA - AIRBNB ===================================================#

# imports
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
from wordcloud import WordCloud, STOPWORDS

# Configuracoes cabecalho da pagina
st.set_page_config(layout='wide', page_icon='bar_chart', page_title='AIRBNB - EDA')

# main
col1, col2 = st.columns([1.5, 8])

st.markdown('## EDA - Exploratory Data Analysis (AIRBNB)')

st.sidebar.markdown('### Contexto:')
st.sidebar.write("""Desde 2008, hóspedes e anfitriões usam o Airbnb para expandir as possibilidades de viagem e apresentar uma maneira mais única e personalizada de experimentar o mundo. 
                    Este conjunto de dados descreve a atividade de listagem e as métricas em NYC, NY para 2019.""")

st.sidebar.markdown('### Objetivo:')
st.sidebar.markdown(
    """Este trabalho teve como objetivo realizar o EDA (Exploratory Data Analysis) do conjunto de dados especificado para download logo abaixo.""")
st.sidebar.write('fonte:https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data')


# ======================================================= PREPARAÇÃO DOS DADOS ==================================================#

# load data
@st.cache(allow_output_mutation=True)
def get_data(__path__):
    df = pd.read_csv(__path__, sep=',')
    df['id'] = df['id'].astype(str)
    df['host_id'] = df['host_id'].astype(str)
    return df


data = get_data('datasets/AB_NYC_2019.csv')


def convert_df(df):
    return df.to_csv().encode('utf-8')


csv = convert_df(data)

st.sidebar.download_button(
    label="Download data as CSV",
    data=csv,
    file_name='AB_NYC_2019.csv',
    mime='text/csv',
)

# Descrição dos dados
st.markdown('---')
st.write('Total de linhas e colunas:', data.shape)
st.write('Descrição dos dados:', data.describe())

# Variáveis Qualitativas e Quantitativas

col1, col2 = st.columns([4, 4])

qualitativas = data[['id', 'name',
                     'host_id', 'host_name', 'neighbourhood_group',
                     'neighbourhood', 'room_type', 'last_review']]

with col1:
    st.write('Variáveis qualitativas:', qualitativas)

quantitativas = data[['latitude', 'longitude', 'price',
                      'minimum_nights', 'number_of_reviews',
                      'reviews_per_month', 'calculated_host_listings_count',
                      'availability_365']]

with col2:
    st.write('Variáveis quantitativas:', quantitativas)

# Data overview
st.write('Data Overview:', data);

st.markdown("---")

# ========================================= ANÁLISE GRÁFICA DAS VARIÁVEIS =========================================#

# Box plot variáveis quantitativas
st.write('Box Plot - Variáveis quantitativas:')
fig = plt.figure(figsize=(12, 8))
for i, col in enumerate(quantitativas):
    plt.subplot(3, 3, i + 1);
    data.boxplot(col)
    plt.tight_layout()

st.pyplot(fig)

st.markdown("---")

# Matriz de correlação
st.write('Matriz de correlação entre as variáveis quantitativas:')

fig = plt.figure(figsize=(10, 7))
sns.heatmap(data[['latitude', 'longitude', 'price',
                  'minimum_nights', 'number_of_reviews', 'reviews_per_month',
                  'calculated_host_listings_count', 'availability_365']].corr(),
            cmap="Blues", annot=True, vmax=1)

st.pyplot(fig)

st.markdown('----')
# ============================================ Análise das variáveis ========================================#
col1, col2 = st.columns(2)

with col1:
    DataRoomType = data['room_type'].value_counts()
    FigRoomType = px.bar(DataRoomType,
                         x=DataRoomType.index,
                         y='room_type', text_auto=True,
                         color=DataRoomType.index,
                         title='Contagem room_type:')

    FigRoomType.update_layout(plot_bgcolor="rgba(0,0,0,0)",
                              xaxis=(dict(showgrid=False)))

    st.plotly_chart(FigRoomType, use_container_width=True)

with col2:
    DataRoomPrice = data.groupby(['room_type']).mean()[['price']].sort_values('price')
    FigRoomPrice = px.bar(DataRoomPrice,
                          x=DataRoomPrice.index,
                          y='price', text_auto=True,
                          color=DataRoomPrice.index,
                          title='Preço médio por room_type:')

    FigRoomPrice.update_layout(plot_bgcolor="rgba(0,0,0,0)",
                               xaxis=(dict(showgrid=False)))

    st.plotly_chart(FigRoomPrice, use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    DataRoomMinimunNights = data.groupby(['room_type']).mean()[['minimum_nights']].sort_values('minimum_nights')
    FigRoomMinimumNights = px.bar(DataRoomMinimunNights,
                                  x=DataRoomMinimunNights.index,
                                  y='minimum_nights', text_auto=True,
                                  color=DataRoomMinimunNights.index,
                                  title='Média de minimum_nights por room_type:')

    FigRoomMinimumNights.update_layout(plot_bgcolor="rgba(0,0,0,0)",
                                       xaxis=(dict(showgrid=False)))

    st.plotly_chart(FigRoomMinimumNights, use_container_width=True)

with col2:
    DataRoomReviews = data.groupby(['room_type']).sum()[['number_of_reviews']].sort_values('number_of_reviews')
    FigRoomReviews = px.bar(DataRoomReviews,
                            x=DataRoomReviews.index,
                            y='number_of_reviews', text_auto=True,
                            color=DataRoomReviews.index,
                            title='Total de reviews por room_type:')

    FigRoomReviews.update_layout(plot_bgcolor="rgba(0,0,0,0)",
                                 xaxis=(dict(showgrid=False)))

    st.plotly_chart(FigRoomReviews, use_container_width=True)

# ========================================================== classificação price ==========================================#

pricelist = [
    (data['price'] <= 150),
    (data['price'] > 150) & (data['price'] <= 300),
    (data['price'] > 300)]

choicelist = ['low_price', 'regular_price', 'hight_price']

data['price_status'] = np.select(pricelist, choicelist)

col1, col2 = st.columns(2)

with col1:
    DataPriceStatus = data['price_status'].value_counts()
    FigPriceStatus = px.bar(DataPriceStatus,
                            x=DataPriceStatus.index,
                            y='price_status', text_auto=True,
                            color=DataPriceStatus.index,
                            title='Contagem de room por price_status:')

    FigPriceStatus.update_layout(plot_bgcolor="rgba(0,0,0,0)",
                                 xaxis=(dict(showgrid=False)))

    st.plotly_chart(FigPriceStatus, use_container_width=True)

with col2:
    DataPriceStatusMean = data.groupby(['price_status']).mean()[['price']].sort_values('price')
    FigPriceStatusMean = px.bar(DataPriceStatusMean,
                                x=DataPriceStatusMean.index,
                                y='price', text_auto=True,
                                color=DataPriceStatusMean.index,
                                title='Média price_status:')

    FigPriceStatusMean.update_layout(plot_bgcolor="rgba(0,0,0,0)",
                                     xaxis=(dict(showgrid=False)))

    st.plotly_chart(FigPriceStatusMean, use_container_width=True)

Host_number_reviews = data.groupby('id').sum()[['number_of_reviews']].sort_values('number_of_reviews',
                                                                                  ascending=False).head(15)
Host_number_reviews2 = Host_number_reviews.sort_values('number_of_reviews', ascending=False)
FigPHostNumberReviews = px.bar(Host_number_reviews2,
                               x=Host_number_reviews2.index,
                               y='number_of_reviews', text_auto=True,
                               color_discrete_sequence=["#0083B8"] * len(Host_number_reviews2),
                               template="plotly_white",
                               title='15 host com maior número de avaliações:')

FigPHostNumberReviews.update_layout(plot_bgcolor="rgba(0,0,0,0)",
                                    xaxis=(dict(showgrid=False)))
st.plotly_chart(FigPHostNumberReviews, use_container_width=True)

st.markdown('----')

# ============================================== wordcloud =============================================#

st.markdown("#### WordCloud for listing names:")

summary = data.dropna(subset=['name'], axis=0)['name']

# concatenar as palavras
all_summary = " ".join(s for s in summary)

# lista de stopword
stopwords = set(STOPWORDS)
stopwords.update(["da", "meu", "em", "você", "de", "ao", "os"])

# gerar uma wordcloud
wordcloud = WordCloud(stopwords=stopwords,
                      background_color="white",
                      width=1000, height=500).generate(all_summary)

# mostrar a imagem final
fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(wordcloud, interpolation='bilinear')
ax.set_axis_off()
plt.imshow(wordcloud),
st.pyplot(fig)

st.markdown('----')

# ============================================================ Total de host por cidade ====================================#

neighbourhood_host_counts = data.groupby('neighbourhood_group').count()[['host_id']].sort_values('host_id',
                                                                                                 ascending=False)
Figneighbourhood_host_counts = px.bar(neighbourhood_host_counts,
                                      x=neighbourhood_host_counts.index,
                                      y='host_id', text_auto=True,
                                      color=neighbourhood_host_counts.index,
                                      title='Total de host por neighbourhood:')

Figneighbourhood_host_counts.update_layout(plot_bgcolor="rgba(0,0,0,0)",
                                           xaxis=(dict(showgrid=False)))
st.plotly_chart(Figneighbourhood_host_counts, use_container_width=True)

st.markdown('----')

# ========================================== mapas =======================================================#

# filtro dos mapas
col1, col2 = st.columns([2, 3])

with col1:
    price_status = st.multiselect('Status de preço:', options=data['price_status'].unique(),
                                  default=data['price_status'].unique())

with col2:
    neighbourhood_group = st.multiselect('neighbourhood_group:', options=data['neighbourhood_group'].unique(),
                                         default=data['neighbourhood_group'].unique())

data_selection = data.query(
    "price_status == @price_status & neighbourhood_group == @neighbourhood_group"

)

# Construção e plotagem dos mapas

col1, col2 = st.columns([2, 2])

with col1:
    st.markdown('Localização host_id por price:')
    data_map = data_selection[['host_id', 'latitude', 'longitude', 'price']]
    fig_map = px.scatter_mapbox(data_selection, lat='latitude', lon='longitude',
                                hover_name='host_id',
                                hover_data=['price'],
                                color='price',
                                zoom=10,
                                size='price',
                                size_max=15,
                                height=500, )

    fig_map.update_layout(mapbox_style='open-street-map')
    fig_map.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    st.plotly_chart(fig_map, use_container_width=True)

with col2:
    st.markdown('Localização neighbourhood_group:')
    data_map = data_selection[['host_id', 'latitude', 'longitude', 'price', 'neighbourhood_group']]
    fig_map = px.scatter_mapbox(data_selection, lat='latitude', lon='longitude',
                                hover_name='host_id',
                                hover_data=['price'],
                                color='neighbourhood_group',
                                zoom=10,
                                size='price',
                                size_max=15,
                                height=500, )

    fig_map.update_layout(mapbox_style='open-street-map')
    fig_map.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    st.plotly_chart(fig_map, use_container_width=True)
