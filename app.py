import os.path
from flask import Flask
from flask import render_template
import os
import joblib as jb
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

@app.route('/')
def main_page():

    #base_dds = pd.DataFrame()
    # Scrapando os dados do site do
    #base_dds = scrapa_site(base_dds)
    #data = trata_dados(base_dds)

    #data['m_quadrado'] = pd.to_numeric(data['m_quadrado'])pi
    #data['qntd_quartos'] = pd.to_numeric(data['qntd_quartos'])
    #data['aluguel'] = pd.to_numeric(data['aluguel'],errors='coerce')
    #data['preco_total'] = pd.to_numeric(data['preco_total'],errors='coerce')
    
    data = pd.read_csv('data_display_site.csv', sep=',', error_bad_lines=False)
    print(data.head())

    lgbm = jb.load('Modelos/lgbm.pkl.z')
    rf = jb.load('Modelos/randodm_forest.pkl.z')

    data2 = data.drop(["img_imovel", "Unnamed: 0"], axis=1,)


    lgm = jb.load('Modelos/lgbm.pkl.z')
    rf = jb.load('Modelos/randodm_forest.pkl.z')

    le_bairro = jb.load('Modelos/label_bairro_imovel.pkl.z')
    le = LabelEncoder()
    le_tipo = jb.load('Modelos/label_tipo_imovel.pkl.z')

    data2['tipo_imovel'] = le_tipo.transform(data2['tipo_imovel'])
    data2['rua_imovel'] = le.fit_transform(data2['rua_imovel'])
    data2['bairro_imovel'] = le_bairro.transform(data2['bairro_imovel'])

    resultados_lgb = lgm.predict_proba(data2)
    resultados_rf = rf.predict_proba(data2)

    p = 0.6 * resultados_lgb[:, 1] + 0.4* resultados_rf[:, 1]
    
    data['previsao'] = p

    display = data[['m_quadrado', 'img_imovel', 'previsao']]
    display = display.sort_values('previsao', ascending=False)
    display['previsao'] = pd.Series([round(val, 4) for val in display['previsao']], index = display.index)

    n_registros = len(display)
    
    display.style.set_table_attributes('class="table-style"')
    pd.set_option('display.max_colwidth', 5)
    return render_template('main.html',tables=[display.to_html(classes='table table-striped',index=False,table_id='table-style')], titles=display.columns.values, n_registros=n_registros)
        

if __name__ == '__main__':
    app.run(debug=True)