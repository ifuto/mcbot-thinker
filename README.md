# mcbot-thinker

マイクラ Java Edition 1.9+ の**ソードPvP戦闘BOT**。物理コアはCで書き、その上で
自己対戦強化学習(PPO)を走らせ、BOT同士を戦わせて学習・評価するためのプロジェクトです。

## 何をするものか

- **Cで書かれた高速PvPシミュレータ**（ヘッドレス、3次元、マイクラ準拠の物理）。
  2コアのオンボロPCでも ~700万 agent-ticks/sec を超える処理量で並列自己対戦を回せます。
- **自己対戦強化学習**(PPO)。learner が徐々に更新される opponent(EMA) と戦い、
  勝敗・ダメージ差・クリティカル・コンボなどの報酬で方策を最適化します。
- **学習後の行動設定スイッチ**（プリセット）。「コンボしかしない」「adtapなし」
  「クリティカルだけ」「疾走なし」等を、再学習なしで推論時に適用できます。
- **BOT同士の対戦**と **HTMLリプレイ**。

### Kit（あなたの指定どおり）
- 武器: **エンチャなしのダイヤ剣**（攻撃力7、攻撃速度1.6）
- 装備: **エンチャなしのダイヤフル**（アーマーポイント20 / タフネス0 / KB耐性0）

## 物理の忠実度（実数値は公式Wiki等を検索して確定）

| 項目 | 値 | 出典 |
| --- | --- | --- |
| 歩行 / 疾走 / スニーク速度 | 4.317 / 5.612 / 1.295 m/s | minecraft.wiki/Player |
| ジャンプ高 | 1.2522 ブロック（重力0.08 + 縦ドラッグ0.98） | minecraft.wiki |
| ヒットボックス | 幅0.6 × 高1.8（スニーク1.5）、アイ高1.62 | minecraft.wiki |
| 攻撃リーチ | 3.0（自分の足元中心→相手AABBの最近点距離） | minecraft.wiki/Sword |
| ダイヤ剣ダメージ/速度 | 7 / 1.6（クールダウン12.5tick） | minecraft.wiki/Diamond_Sword |
| クールダウン乗算 | 0.2+((t+0.5)/T)²×0.8（[0.2,1.0]） | フォーラム記載の公式 |
| クリティカル | 落下中＆非疾走で +50% | minecraft.wiki |
| 疾走ノックバック攻撃 | 充填≥84.8%、命中で疾走キャンセル+縦KB 0.5 | minecraft.wiki/Knockback |
| **Wタップ（疾走リセット）** | 疾走KB攻撃後に疾走ロック3tick。握りっぱなしでは自動再疾走できず、Wを離すタイミングが必要（=Wタップ） | minecraft.wiki/Tutorial:PvP |
| **ジャンプリセット** | 被弾時に空中だと受ける横KB ×0.7 | minecraft.wiki/Tutorial:PvP |
| 被弾後無敵 | 10 tick (0.5秒) | minecraft.wiki/Knockback |
| ヒットセレクト | 相手攻撃直後の攻撃で受ける横KB ×0.6 | minecraft.wiki/Tutorial:PvP |
| ダイヤ防具の減算 | `reduction=20-4dmg/8`、最大80%カット | 1.9+ の防具式 |

**3次元リーチの垂直非対称**を再現しています（下の相手は攻撃しやすく、上は届きにくい）。
さらに **Wタップ**（疾走KB攻撃後の疾走ロックにより「Wを離して再疾走する」タイミング）と
**ジャンプリセット**（被弾時に空中なら受ける横KB減）を物理に実装してあるため、
**AIが強化学習で自主的にこれらの操作テクニックを学べます**（学習結果そのものはRLが自動で見つけます）。
KB絶対値などは「実測に近い値」をチューニング可能な定数として C の `core.c` 冒頭に置いています。
**バイト単位で完全一致するものではない**こと（あくまで動きに直結する軸を実数準拠にした近似）を
ご理解ください。細部は定数を書き換えれば調整できます。

## クイックスタート

```bash
# 1) 依存（CPU版torch。オンボロでも可）
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2) Cコアをビルド（gcc/clangが必要）
.venv/bin/python mcbot/sim/build.py

# 3) 物理ユニットテスト
.venv/bin/python -m pytest tests/ -q
```

### とりあえずBOT同士を戦わせる（学習済み同梱BOT vs スクリプトBOT）

```bash
# 同梱のデフォルトBOT（ウォームスタートでスクリプト剣士を行動クローニングしたもの）
.venv/bin/python -m mcbot.duel --a models/sword1v1_default.pt --b scripted:aggro --games 20

# スクリプトBOT同士
.venv/bin/python -m mcbot.duel --a scripted:aggro --b scripted:strafer --games 20

# ミラー戦
.venv/bin/python -m mcbot.duel --a models/sword1v1_default.pt --b models/sword1v1_default.pt --games 10
```

対戦は **サンプリング（非greedy）** で行われます。このタイミング依存タスクでは
`--greedy` の毎tick argmax はクールダウンに合わせた攻撃を選びにくいため通常は使いません。

### HTMLリプレイ

```bash
.venv/bin/python -m mcbot.duel --a scripted:aggro --b models/sword1v1_default.pt \
    --games 1 --replay replay.html
# ブラウザで replay.html を開く（再生/スライダー/スペースキー）
```

### 学習（1日放置・自動再開）

**推奨: 強スクリプト相手と直接PPO対戦**（`mcbot.rl.train_vs_scripted`）。
ウォームスタートの行動クローニングは、クールダウンに合わせた攻撃タイミングを再現できず
「連打か不攻撃」に崩れるため、固定の強いスクリプト剣士を相手に報酬で直接タイミングを
学ばせる方式に置き換えました。`--iterations 0` で無制限（Ctrl+Cで保存終了）。

```bash
.venv/bin/python -m mcbot.rl.train_vs_scripted --iterations 0 \
    --nmatches 512 --rollout 128 --ckpt-dir checkpoints --run sword1v1 \
    --style aggro --save-every 100 --log-every 20 \
    --eval-every 50 --eval-matches 8 --max-minutes 340
```

- `--style` は相手のスクリプト剣士のスタイル（`aggro` / `strafer`）。
- `--eval-every` で「スクリプト相手」への固定ベースライン評価を行い、
  **実力が最も高い時点を `*_best.pt`** として保存します。
- `--max-minutes N` で「N分経ったらきれいに保存して停止」。

従来の自己対戦PPO（`mcbot.rl.train`）も `mcbot/rl/ppo.py` に残っていますが、自己対戦は
相手が適応して勝率が振動するため不安定で、推奨しません。同梱デフォルトBOT
（`models/sword1v1_default.pt`）は vs-scripted 学習でランダム相手に ~0.98 勝ちます。

- 途中経過は `checkpoints/sword1v1_log.csv` に記録、Ctrl+C で保存終了、次回起動で自動再開。
- `--max-minutes N` で「N分経ったらきれいに保存して停止」できます（CI用）。

### GitHub Actions で6時間まるごと学習

`.github/workflows/train.yml` を同梱しています。

1. このリポジトリを GitHub にプッシュする。
2. リポジトリの **Actions** タブ → **train-sword-1v1** → **Run workflow** を押す。
   （workflowのTrainステップのコマンドは、下記の vs-scripted 版を使ってください）
3. ジョブが **~5時間40分** 学習し（512並列アリーナ、自動でウォームスタート→PPO自己対戦）、
   終了時に **`sword1v1-checkpoints` というアーティファクト** として以下をアップロード:
   - `checkpoints/sword1v1_best.pt`（固定ベースライン評価で最強の時点）
   - `checkpoints/sword1v1_it*.pt`（最新）
   - `checkpoints/sword1v1_log.csv`（履歴）

   Actions のジョブ上限が6時間(=360分)なので、`--max-minutes 340` でクリーンに停止させ、
   アップロード時間を確保しています。
- `--eval-every` で「ランダム相手」への固定ベースライン評価を定期的に行い、
  **実力が最も高い時点を `*_best.pt` として保存**します（自己対戦相手は適応するため、
  自己対戦勝率は振動しますが、固定ベースライン評価が信頼できます）。
- デフォルトは2コアで数万 agent-steps/sec。1日放置で数十億ステップ = 数百本のアリーナでの
  **究極並行自己対戦**になります。ただし「人間のPvP何年分」という表現は定量的に約束できず、
  あくまで並列ステップ量の話です。

### 学習後の行動プリセット（再学習不要）

`--a-presets` / `--b-presets` にカンマ区切りで指定します。

```bash
# A: コンボ専用（ジャンプ禁止・常時疾走）、かつ adtap(A/Dストレイフ)なし
.venv/bin/python -m mcbot.duel --a models/sword1v1_best.pt --a-presets combo_only,no_adtap \
    --b scripted:aggro --games 20
```

利用可能なプリセット: `combo_only`（常時疾走+ジャンプ禁止）, `crit_only`（常時ジャンプ+非疾走）,
`no_jump`, `always_jump`, `no_sprint`, `always_sprint`, `no_sneak`, `no_attack`,
`always_attack`, `no_adtap`(A/Dなし), `strafe_only`(A/Dのみ), `walk_only`, `back_pedal`,
`charge`, `passive`。定義は `mcbot/behavior/presets.py`。複数は論理積で合成されます。

### 報酬のデフォルトと調整

デフォルト報酬（スロット順: dmg_dealt, dmg_taken, crit, sprint_kb, combo, miss, win, lose, draw）:

```
[1.0, 1.0, 0.3, 0.4, 0.5, 0.005, 5.0, 8.0, 2.0]
```

- **ダメージ与えたら+1×量 / 受けたら−1×量**、クリティカル・疾走KB・コンボで少量加点、
  空振りで微小減点、勝利+5 / 敗北−8 / 引き分け−2。
- 敗北(−8)を勝利(+5)より重くして「負けることと引き分けを避けて倒し切る」方向にしています。
- `--reward '1,1,0.3,0.4,0.5,0.005,5,10,2'` のように `--reward` で9個の値を直接指定できます。

## プロジェクト構成

```
mcbot/sim/core.{h,c}   Cシミュレータ（物理・報酬・自己対戦エンジン）
mcbot/sim/build.py     gccでcore.soにビルド
mcbot/sim/env.py       ctypesラッパー（ベクトル化環境）
mcbot/sim/consts.py    観測/行動/報酬の定数
mcbot/rl/network.py    小型ActorCritic(マルチディスクリート行動ヘッド)
mcbot/rl/ppo.py        PPO自己対戦トレーナー（EMAオッポーネント）
mcbot/rl/train.py      学習CLI（ウォームスタート/再開/best保存）
mcbot/rl/warmstart.py  スクリプト剣士の行動クローニング(BC)
mcbot/behavior/presets.py  学習後の行動プリセット
mcbot/duel.py          BOT vs BOT 対戦CLI
mcbot/viz/replay_html.py    HTMLリプレイ生成
mcbot/bots.py          Bot(方策) / ScriptedBot(スクリプト剣士)
config.py              訓練設定（並列数・報酬・ネット等）
tests/test_physics.py  物理ユニットテスト
```

## 誠実な現状と注意（RLについて）

- **シミュレータは完成・テスト済み**（`tests/` が13項目通ります。速度・リーチ・アーマー・
  Wタップ疾走ロック・ジャンプリセット等）。
- **学習パイプラインは動きます**：ウォームスタートBOTはランダム相手に ~75% 勝ち、
  PPO自己対戦でダメージ差を伸ばし、キル(勝敗)も発生します。
- 一方、**純自己対戦は本質的に不安定**です（オッポーネントが適応するため勝率が振動し、
  場合によってはウォームスタートより弱化することも）。これが強化学習一般の難しさです。
  そのため `--eval-every` で固定ベースライン評価による best チェックポイント保存を
  組み込み、安心して放置学習できるようにしました。
- 同梱の `models/sword1v1_default.pt` は「行動クローニング済みのスクリプト剣士」で、
  即座に安定して戦えるbaselineです。より強い相手(スクリプトaggro等)への対抗や
  高度な戦術(コンボ/クリティカル)は、長期間の学習とプリセット調整で育成してください。
- 物理の完全再現は目指しておらず、動きに直結する軸を実数値準拠にした近似です。

## License

MIT (see LICENSE)
