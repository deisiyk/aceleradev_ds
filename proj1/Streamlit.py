import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from datetime import datetime

# converte o valor recebido para float com duas casas decimais
def converte_valor(valor):
    return (float(valor) / 100)


# converte "AAAAMMDD" em data
def converte_data(data):
    return datetime.strptime(data, '%Y%m%d').strftime('%d/%m/%Y')


@st.cache()  # faz a carga dos dados uma unica vez
def carga(arquivo):
    # No arquivo nao estao descritos o cabecalho
    colunas = ['TIPREG', 'DATPRE', 'CODBDI', 'CODNEG', 'TPMERC', 'NOMRES',
               'ESPECI', 'PRAZOT', 'MODREF', 'PREABE', 'PREMAX', 'PREMIN',
               'PREMED', 'PREULT', 'PREOFC', 'PREOFV', 'TOTNEG', 'QUATOT',
               'VOLTOT', 'PREEXE', 'INDOPC', 'DATVEN', 'FATCOT', 'PTOEXE',
               'CODISI', 'DISMES']
    # As colunas estao separadas posicionalmente, com seus respectivos tamanhos
    tamanho = [2, 8, 2, 12, 3, 12, 10, 3, 4, 13, 13, 13, 13, 13,
               13, 13, 5, 18, 18, 13, 1, 8, 7, 13, 12, 3]

    # Especifica parametro dtype para carga do csv
    parm_dtype = {
        'TOTNEG': np.int32
    }

    # Use the functions defined above to convert data while loading
    # Faz as conversoes durante a carga do csv
    parm_convert = {
        'DATPRE': converte_data,
        'PREABE': converte_valor,
        'PREMAX': converte_valor,
        'PREMIN': converte_valor,
        'PREMED': converte_valor,
        'PREULT': converte_valor,
        'PREOFC': converte_valor,
        'PREOFV': converte_valor,
        'DATVEN': converte_data,
    }

    # Como os dados estao separados com formatacao de tamanho fixo
    # a leitura será ccom read_fwf
    df = pd.read_fwf(
        arquivo,
        widths=tamanho,
        names=colunas,
        dtype=parm_dtype,
        converters=parm_convert,
        skiprows=1,  # desconsidera o header
        skipfooter=1  # desconsidera o footer
    )
    return df

def get_y_vars(dataset, x, variables):
    corrs = dataset.corr()[x]
    remaining_variables = [v for v in variables if v != x]
    sorted_remaining_variables = sorted(
        remaining_variables, key=lambda v: corrs[v], reverse=True
    )
    format_dict = {v: f"{v} ({corrs[v]:.2f})" for v in sorted_remaining_variables}
    return sorted_remaining_variables, format_dict

def main():


    df = carga('COTAHIST_2020BI.TXT') # carga('Teste.txt')

    add_selectbox = st.sidebar.selectbox(
        'Dados para apresentação:',
        ('Sobre mim', 'Visão Geral dos Dados', 'Visão Detalhada'))

    if add_selectbox == 'Sobre mim':
        st.title('Seja bem vindo o meu projeto streamlit!')
        st.header("Deisi Yuki Kaetsu")
        st.markdown("**Formação:** Ciência da Computação")
        st.markdown("**Linkedin:** https://www.linkedin.com/in/deisi-kaetsu-9175282a/")
        st.markdown("**Github:** https://github.com/deisiyk")

    elif add_selectbox == 'Visão Geral dos Dados':
        st.title('Ações da Bovespa - 2019')
        st.header("** Exploração dos dados **")
        st.subheader("**Informações sobre o dataset:**")
        st.markdown("Total de Linhas: " + str(df.shape[0]))
        st.markdown("Total de Colunas: " + str(df.shape[1]))

        st.subheader("**Visualização do dataset:**")
        linhas = st.slider('Quantidade de linhas para visualizar', min_value=1, max_value=20,value=5 )
        st.dataframe(df.head(linhas))

        st.subheader("**Estrutura do dataset:**")
        expl = pd.DataFrame({
            'COLUNA': df.columns,
            'TIPO': df.dtypes,
            'NAN': df.isna().sum(),
            'NAN %': (df.isna().sum() / df.shape[0]) * 100
        })
        st.dataframe(expl)

    else:
        st.title('Ações da Bovespa - Primeiro bimestre 2020')
        lista = ['Numéricos', 'Não Numérico']
        dado = st.sidebar.selectbox('Tipo de dados para visualizar?', lista)

        if dado == 'Não Numérico':
            expl = pd.DataFrame({
                'COLUNA': df.columns,
                'TIPO': df.dtypes,
                'NAN': df.isna().sum(),
                'NAN %': (df.isna().sum() / df.shape[0]) * 100
            })
            st.header("** Não Numérico **")
            st.markdown('Colunas com datas:')
            st.table(expl.loc[(['DATPRE', 'DATVEN']), ['NAN', 'NAN %']])

            st.markdown('Colunas com dados categóricos:')
            st.table(expl.loc[(['TPMERC','CODNEG', 'NOMRES',
                                'ESPECI', 'MODREF', 'CODISI']),
                              ['NAN', 'NAN %']])

        else:
            numerico = ['DISMES', 'PREABE', 'PREMAX', 'PREMED', 'PREMIN', 'PREOFC',
                        'PREOFV', 'PREULT', 'QUATOT', 'TOTNEG', 'VOLTOT']
            expl = pd.DataFrame({
                'COLUNA': df[numerico],
                'NAN': df[numerico].isna().sum(),
                'NAN %': (df[numerico].isna().sum() / df[numerico].shape[0]) * 100,
                'MÉDIA': df[numerico].mean(),
                'MEDIANA': df[numerico].median(),
                'DESVIO_PADRAO': df[numerico].std()
            })
            st.header("** Numérico **")
            st.markdown('Colunas com dados numéricos:')
            st.table(expl.loc[numerico,
                              ['MÉDIA', 'MEDIANA', 'DESVIO_PADRAO','NAN', 'NAN %']])

            acoes = df.CODNEG.unique()
            selec = st.selectbox('Selecione ação para análise de preço médio:', acoes)

            df_aux = df[df.CODNEG == selec]
            #st.markdown('Histograma:')
            #bins = list(np.arange(0, 20, 0.5))
            #hist_values, hist_indexes = np.histogram(df.PREMED[df.CODNEG == selec], bins=bins)
            #st.bar_chart(pd.DataFrame(data=hist_values, index=hist_indexes[0:-1]), width=1000)
            #st.write('Target value min: {0:.2f}%; max: {1:.2f}%; mean: {2:.2f}%; std: {3:.2f}'.format(
            #    np.min(df.PREMED[df.CODNEG == selec]), np.max(df.PREMED[df.CODNEG == selec]), np.mean(df.PREMED[df.CODNEG == selec]), np.std(df.PREMED[df.CODNEG == selec])))

            st.subheader('Preço médio da ação X Dias')

            period = st.slider('Média móvel por período (dias)', min_value=2, max_value=30,
                               value=7, step=1)
            data = df_aux.copy()
            data2 = df_aux.PREMED.to_frame('Preco Medio')
            data[f'Media {period}'] = df_aux.PREMED.rolling(period).mean()
            data2[f'Media {period} dias'] = data[f'Media {period}'].reindex(data2.index)
            st.line_chart(data2)


            st.subheader("Correlation")
            numerico = [ 'DISMES', 'PREABE', 'PREMAX', 'PREMED', 'PREMIN', 'PREOFC',
                        'PREOFV', 'PREULT', 'QUATOT', 'TOTNEG', 'VOLTOT']
            x = st.selectbox("x", numerico)
            y_options, y_formats = get_y_vars(df_aux, x, numerico)
            y = st.selectbox(
                f"y (correlacao com {x})", y_options, format_func=y_formats.get
             )
            plot = alt.Chart(df_aux).mark_circle().encode(x=x, y=y)
            st.altair_chart(plot)


if __name__ == '__main__':
    main()
