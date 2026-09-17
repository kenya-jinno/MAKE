# MAKE48 — 編集部問い合わせ（tortured phrases / 参考文献チェック）への対応記録

対象: `main_paper_MAKE47.pdf` への編集部指摘 → `main_paper_MAKE48.tex` / `.pdf` で対応
（参考文献の追加・削除は行っていないため、引用番号 [1]–[37] は MAKE47 と完全に同一）

---

## 1. Tortured phrases への対応

### 指摘 1: "actually occur"
本文中に 3 箇所存在（§2 Related Work、§3.2.2 Step 2、§5.7 冒頭）。すべて書き換えた。

| 箇所 | 修正前 | 修正後 |
|------|--------|--------|
| §2（Active units and posterior collapse） | whether truncation **actually occurred** is diagnosed in Step 3 | whether truncation **has taken place** is diagnosed in Step 3 |
| §3.2.2（Step 2, VAE） | whether truncation **actually occurred** is checked by V1 | whether truncation **has taken place** is checked by V1 |
| §5.7（Applicability Boundary 冒頭） | failure modes ... **actually occur** | failure modes ... **arise in practice** |

さらに予防的に、他の "actually" も 5 箇所削減した（"coordinates actually used" → "coordinates in active use"（2箇所）、"were actually truncated" → "were indeed truncated"、"actually requires" → "truly requires"、"actually observed" → "observed empirically"、"actually in use" → "in use"）。

### 指摘 2: "multi-see"
**本文に "multi-see" という語句は存在しない。** これは標準的な機械学習用語 **"multi-seed"**（複数の乱数シードで実験を反復すること）の部分文字列を自動検出ツールが誤検出したもの。"multi-seed" は本論文の再現性主張の中核用語（multi-seed validation 等）であり、NeurIPS/ICLR 等の文献でも標準的に用いられるため**変更しない**。

### 全文チェック
全文を通読し、既知の tortured-phrase パターン（Problematic Paper Screener 系リスト）との照合も実施。上記以外に該当なし。

---

## 2. 参考文献チェックへの対応

### 番号対応表（bibitem 順 = 引用番号）

| # | キー | # | キー | # | キー |
|---|------|---|------|---|------|
| 1 | kingma2014vae | 14 | jinno2023multimodal | 27 | hatcher2002algebraic |
| 2 | bengio2013representation | 15 | dai2024latent | 28 | lecun1998 |
| 3 | lucas2019elbo | 16 | birdal2021intrinsic | 29 | xiao2017fashion |
| 4 | fefferman2016manifold | 17 | dai2026island | 30 | kornblith2019cka |
| 5 | facco2017twonn | 18 | levina2005mle | 31 | johnsson2015ess |
| 6 | pope2021intrinsic | 19 | ceruti2014danco | 32 | fu2019cyclical |
| 7 | ansuini2019intrinsic | 20 | camastra2016intrinsic | 33 | krizhevsky2009learning |
| 8 | valeriani2023geometry | 21 | allegra2020data | 34 | netzer2011reading |
| 9 | obata2025manifold | 22 | tippingbishop1999ppca | 35 | sonderby2016ladder |
| 10 | higgins2017betavae | 23 | bishop1999bayesian | 36 | cover2006elements |
| 11 | alain2014dae | 24 | ghahramani2005infinite | 37 | hoffman2016elbo |
| 12 | rifai2011cae | 25 | nazari2023geometric | | |
| 13 | okamoto2023latent | 26 | kato2020rdae | | |

### (a) Out-of-scope: [27], [28] → **維持（正当化）**
- [27] Hatcher『Algebraic Topology』: 命題1（位相的下界）の証明で埋め込み次元の定義の典拠として引用。理論主張に不可欠。
- [28] LeCun et al. 1998: 主実験データセット MNIST の原典。Data Availability 節でも必須。

### (b) Outdated: [2], [4], [18]–[20], [22], [27], [28], [31], [35], [36] → **維持（正当化）**
いずれも本論文で使用する手法・定義の**原典**（MLE 推定器 2005、DANCo 2014、ESS 2015、PPCA 1999、water-filling/情報理論、ladder VAE、多様体仮説、表現学習レビュー）。原典引用は意図的であり、最新動向は [6], [8], [16], [25]（2021–2023）および [9], [17]（2025–2026）でカバー済み。

### (c) 自己引用: [13]–[15], [17]（＋[9] も自グループ） → **維持（正当化）**
37 件中 5 件（約 14%）。[9] は本論文が拡張する NOLTA2025 予備研究で §1.2 に明示的に宣言済み。[13]–[15], [17] は §2 で実質的に議論される先行研究。編集部が削減を求める場合は [14]（multimodal, HCII2023）が最も周辺的で削除候補。

### (d) Overcited authors: [5], [7], [13]–[15], [17], [21] → **維持（正当化）**
[5], [7], [21] は A. Laio らのグループ（TwoNN 開発元）。TwoNN は本論文の主推定器であり、その原典 [5]・深層表現への適用 [7]・不均一 ID の注意点 [21] の引用は不可避。[13]–[15], [17] は (c) と同じ。

### (e) 要検証: [1], [9]–[12], [23]–[26], [33], [34], [37] → **全件検証・書誌情報補完**

| # | 検証結果と修正 |
|---|----------------|
| [1] kingma2014vae | 正しい。会議情報を補完（2nd ICLR, Banff, 2014.4.14–16）＋ arXiv:1312.6114 追記 |
| [9] obata2025manifold | 自グループの NOLTA2025 論文（沖縄, 2025.10, pp. 32–35）。記載どおり（著者確認済み） |
| [10] higgins2017betavae | 正しい。会議情報を補完（5th ICLR, Toulon, 2017.4.24–26） |
| [11] alain2014dae | 正しい（JMLR 15, 3743–3773, 2014）。変更なし |
| [12] rifai2011cae | 「Vol.~28」の紛らわしい表記を「28th ICML, Bellevue, WA, pp. 833–840」に修正（dblp で確認） |
| [23] bishop1999bayesian | 正しい（NIPS 11, pp. 382–388）。変更なし |
| [24] ghahramani2005infinite | **著者順の誤りを修正**: Griffiths, T.L.; Ghahramani, Z. が正（NeurIPS proceedings で確認）＋ pp. 475–482 追加 |
| [25] nazari2023geometric | 年・ページ欠落を補完: 40th ICML, PMLR 202, 2023, pp. 25834–25857（PMLR で確認） |
| [26] kato2020rdae | 年・ページ欠落を補完: 37th ICML, PMLR 119, 2020, pp. 5166–5176（PMLR で確認） |
| [33] krizhevsky2009learning | 正しい。出版者住所を MDPI 書式に整形 |
| [34] netzer2011reading | 正しい。開催地（Granada, Spain）を補完 |
| [37] hoffman2016elbo | 正しい。開催地（Barcelona, Spain）を補完 |

加えて、ジャーナル論文の書誌エントリに DOI を追加: [2], [4], [5], [19], [20], [21], [22], [28], [31]（自己引用 [13], [14], [17] は既に DOI あり）。

---

## 3. 編集部への返信文案（英語）

> Dear Editor,
>
> Thank you for the language and reference check. We have addressed all points in the revised manuscript as follows.
>
> **Tortured phrases.** We removed all three occurrences of "actually occur(red)" (Sections 2, 3.2.2, and 5.7) and additionally reduced other uses of "actually" throughout. Regarding "multi-see": this string does not occur in the manuscript; it is a substring of "multi-seed", the standard machine-learning term for experiments repeated over multiple random seeds, which is central to our reproducibility claims and is used unchanged. We have carefully re-checked the entire manuscript and found no other tortured phrases.
>
> **References.** No reference was added or removed, so the numbering [1]–[37] is unchanged.
> - *Out-of-scope [27, 28]*: Ref. [27] (Hatcher, Algebraic Topology) is the standard source for the topological embedding dimension used in the proof of Proposition 1; Ref. [28] (LeCun et al., 1998) is the original reference for the MNIST dataset used in all main experiments. Both are essential and retained.
> - *Outdated [2, 4, 18–20, 22, 27, 28, 31, 35, 36]*: These are the original sources of the estimators, models, and definitions we use (e.g., the MLE estimator [18], DANCo [19], ESS [31], PPCA [22], water-filling [36]); citing the originals is deliberate. Recent developments are covered by Refs. [6, 8, 16, 25] (2021–2023) and [9, 17] (2025–2026).
> - *Self-citations [13–15, 17]*: 5 of 37 references (≈14%) are by the authors' group. Ref. [9] is the preliminary study that this paper explicitly extends (declared in Section 1.2), and Refs. [13–15, 17] are discussed substantively in Related Work. We are willing to trim these if the editor prefers.
> - *Overcited authors [5, 7, 13–15, 17, 21]*: Refs. [5, 7, 21] share a co-author (A. Laio) because TwoNN [5] is the primary estimator of this paper, and [7, 21] are its key applications and caveats; these citations are methodologically unavoidable.
> - *References requiring validation [1, 9–12, 23–26, 33, 34, 37]*: We verified all of them against the publishers' records and completed the entries: corrected the author order of Ref. [24] (Griffiths; Ghahramani), replaced the ambiguous "Vol. 28" of Ref. [12] with "28th ICML, pp. 833–840", added the missing years and page ranges to Refs. [25] (PMLR 202, 2023, pp. 25834–25857) and [26] (PMLR 119, 2020, pp. 5166–5176), and added venue details to Refs. [1, 10, 33, 34, 37]. Ref. [9] is a NOLTA2025 proceedings paper (Okinawa, October 2025, pp. 32–35). DOIs were added to the journal-article entries where available.

---

## 4. 作業メモ

- `main_paper_MAKE48.tex` → pdflatex ×3 でコンパイル済み。エラー 0、未解決引用 0、Overfull hbox 4 件（MAKE47 と同一の既存分）。
- 今回の修正は英語表現と書誌情報のみのため、日本語 NN 版の改稿は行っていない（内容・数値・構成・引用番号に変更なし）。
- 書誌検証ソース: [PMLR v202 (Nazari)](https://proceedings.mlr.press/v202/nazari23a.html), [PMLR v119 (Kato)](https://proceedings.mlr.press/v119/kato20a.html), [NeurIPS 2005 (Griffiths & Ghahramani)](https://proceedings.neurips.cc/paper/2005/file/2ef35a8b78b572a47f56846acbeef5d3-Paper.pdf), [dblp (Rifai et al.)](https://dblp.uni-trier.de/rec/conf/icml/RifaiVMGB11.html)
