システムの使用方法
動作環境
Python 3.12 以上
Streamlit
scikit-learn
pandas
matplotlib
プロジェクト構成
diabetes-risk-screening
│
├── README.md
├── requirements.txt
│
├── data
│
├── src
│   ├── train.py
│   ├── predict.py
│   └── model.pkl
│
├── app
│   └── streamlit_app.py
│
└── venv
Windows環境での実行方法
1. プロジェクトフォルダへ移動
cd diabetes-risk-screening
2. 仮想環境の有効化
venv\Scripts\activate

成功すると以下のように表示される。

(venv)
3. 必要ライブラリのインストール
pip install -r requirements.txt
4. モデルの学習
python src/train.py

学習完了後、以下のファイルが生成される。

src/model.pkl
5. Webアプリケーションの起動
python -m streamlit run app/streamlit_app.py

ブラウザで以下のURLを開く。

http://localhost:8501
macOS環境での実行方法
1. プロジェクトフォルダへ移動
cd diabetes-risk-screening
2. 仮想環境の有効化
source venv/bin/activate

成功すると以下のように表示される。

(venv)
3. 必要ライブラリのインストール
pip install -r requirements.txt
4. モデルの学習
python src/train.py

学習完了後、以下のファイルが生成される。

src/model.pkl
5. Webアプリケーションの起動
python -m streamlit run app/streamlit_app.py

ブラウザで以下のURLを開く。

http://localhost:8501
使用手順
年齢を入力する
BMIを入力する
高血圧の有無を選択する
高コレステロールの有無を選択する
喫煙歴の有無を選択する
運動習慣の有無を選択する
果物摂取状況を選択する
野菜摂取状況を選択する
現在の健康状態を選択する
「Predict」ボタンを押す
出力結果

システムは以下の結果を表示する。

糖尿病リスク（%）
リスクレベル
リスク判定基準
リスク確率	判定
0 ～ 30%	低リスク
30 ～ 70%	中リスク
70%以上	高リスク
注意事項

本システムは研究・教育目的で開発されたものであり、医療診断を行うものではない。

実際の診断や治療については、必ず医療機関や医師へ相談すること。

また、本システムの予測結果は参考情報であり、医学的判断を代替するものではない。