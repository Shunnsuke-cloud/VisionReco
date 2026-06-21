# VisionReco

VisionRecoは、画像をアップロードするとAIが画像内の物体を判定するStreamlit製の画像認識Webアプリです。

## 主な機能

- JPG / JPEG / PNG画像のアップロード
- PyTorch + torchvisionの学習済みMobileNetV3による画像認識
- 上位5候補と信頼度の表示
- 英語ラベルの日本語表示
- 信頼度グラフ
- AI判定コメント
- セッション内の判定履歴
- 履歴をもとにしたカテゴリ分析

## セットアップ

```bash
pip install -r requirements.txt
streamlit run app.py
```

初回実行時はtorchvisionの学習済みモデルをダウンロードします。モデルを読み込めない環境では、画面確認用のデモ判定にフォールバックします。

## 使用技術

| 分類 | 使用技術 |
| --- | --- |
| 開発言語 | Python |
| Webアプリ | Streamlit |
| 画像認識 | PyTorch |
| 学習済みモデル | torchvision / MobileNetV3 |
| 画像処理 | Pillow |
| データ処理 | pandas |
| グラフ | matplotlib |

## 公開

Streamlit Community Cloudで公開する場合は、このリポジトリをGitHubへpushし、`app.py` をエントリーポイントとして指定します。
