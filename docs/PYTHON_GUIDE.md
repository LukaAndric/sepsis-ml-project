# Python i ML: šta stvarno treba da znaš (kad AI piše kod)

Cilj: ne kucati iz glave, nego **razumeti, proveriti i voditi**. Seniori guglaju stalno — sintaksa se ne pamti.
Ono što se ne guglje i što AI ne može da proveri umesto tebe jeste **teorija i procena da li je rezultat veran**.

## 1. Pravilo podele rada
| AI radi | Ti radiš |
|---|---|
| Piše sintaksu, boilerplate, plotove | Postavljaš tačno pitanje i definišeš metriku |
| Predlaže modele i parametre | Proveravaš leakage, split i da li rezultat ima smisla |
| Ispravlja greške u kodu | Čitaš kod i znaš šta svaka linija radi |

Primer iz ovog projekta: AI (i originalni kod) je napravio `shift(-6)` preko celog dataframe-a. Kod je radio bez greške,
ali je mešao pacijente. **Kod koji radi nije kod koji je tačan.** To si uhvatio samo teorijom.

## 2. Python minimum za ML (to je sve što je potrebno za 90% posla)
1. **Osnove:** liste, rečnici, `for`, list comprehension `[f(x) for x in xs if uslov]`, funkcije, f-stringovi, `pathlib.Path`.
2. **Moduli i paketi:** `import`, `from . import config`, šta je `src/` paket, zašto `PYTHONPATH=src`.
3. **Okruženje:** `python -m venv .venv`, `pip install -r requirements.txt`. Jedan projekat = jedno okruženje
   (u ovom projektu je globalni `pip install` pokvario NumPy; venv to sprečava).
4. **NumPy:** `shape`, indeksiranje maskom `a[a > 0]`, `np.where`, `broadcasting`, vektorizacija (nikad petlja preko 1M redova).
5. **Pandas (najvažnije):**
   - `df.loc[maska, "kolona"]`, `iloc`, boolean maske `df[df.x > 3]`
   - `groupby(...).agg(...)`, `groupby(...).transform(...)`, `merge`, `concat`
   - vremenske operacije **po grupi**: `df.groupby("id")["x"].shift(1)`, `.ffill()`, `.rolling(6).mean()`
   - `isna()`, `dtypes`, `astype`, `clip`, `value_counts`
   - **zamke:** duplikati u indeksu (stari bug iz ovog projekta), `SettingWithCopy`, operacije preko granica grupa.
6. **scikit-learn obrazac:** `fit` / `predict` / `predict_proba`, `Pipeline`, `train_test_split`, `GroupKFold`, `cross_val_score`, `metrics`.
7. **Plotovi:** `plt.subplots`, `ax.plot/bar/hist`, `plt.savefig`. Ovo uvek guglaj/traži od AI.
8. **Debug:** `print(df.shape, df.head())`, `df.info()`, `assert` za pretpostavke, čitanje traceback-a **odozdo prema gore**.

## 3. Teorija ML-a koja se uvek pita na intervjuu
- **Bias/variance, overfitting, underfitting**; regularizacija; learning krive.
- **Train/validation/test:** zašto test dirneš jednom. Prag i hiperparametre biraš na **validaciji**.
- **Data leakage** (najčešća greška u praksi): (a) target u featurima, (b) statistike (mean, scaler) fitovane na svim podacima,
  (c) isti entitet (pacijent) u train i test, (d) budućnost u prošlosti (shift preko granica), (e) artefakt prikupljanja
  podataka (ovde: `ICULOS`, jer su zdravi pacijenti odsečeni na ~59 h).
- **Split strategije:** random, stratified, **grouped** (isti pacijent), **time-based** (budućnost nikad u treningu).
- **Metrike za neuravnotežene klase:** accuracy je beskorisna; koristi precision, recall, F1/F2, **PR-AUC**, ROC-AUC.
  Zašto je PR bitniji kad je prevalencija 1% (ROC može izgledati dobro, a precision je 2%).
- **Threshold ≠ model:** model daje verovatnoću, prag je poslovna odluka (cena lažnog alarma vs propuštenog slučaja).
- **Kalibracija:** da li "20% rizik" znači da se u 20% slučajeva desi? (calibration curve, Brier score).
- **Imputacija:** mean vs ffill vs NaN + indikator; zašto se računa samo na train.
- **Modeli:** linearni vs stabla (RF, gradient boosting); zašto GBM često pobeđuje na tabelarnim podacima; kako radi `early_stopping`.
- **Cross-validation** i interval poverenja (bootstrap po pacijentu, ne po redu).
- **Interpretabilnost:** feature importance vs SHAP; **korelacija nije uzročnost** (u ovom projektu top featurei su ponašanje lekara).
- **Drift i monitoring:** data drift, concept drift, PSI.

## 4. Kako proveriti kod koji je napisao AI (kontrolna lista)
1. **Pročitaj** svaku liniju koja menja podatke (`shift`, `ffill`, `merge`, `split`). Pitaj: *radi li ovo po pacijentu?*
2. **Veličine:** koliko redova/pacijenata ulazi i izlazi iz svakog koraka? Se li sve poklapa?
3. **Test prvog reda pacijenta:** ima li prvi sat nove osobe vrednosti iz prethodne? (`tests/test_features.py`)
4. **Prelep rezultat = sumnja.** AUROC 0.99 na ovakvom problemu je skoro uvek leakage.
5. **Baseline:** šta daje glupo pravilo ili model sa 1 featureom? Slučajno mešanje labela mora dati AUC ≈ 0.5.
6. **Ablacija:** izbaci sumnjivu kolonu i vidi koliko pada (ovde `ICULOS`: 0.818 → 0.797).
7. **Ponovi pokretanje:** isti seed daje iste brojke? (`random_state`/`seed` svuda).
8. **Pitaj "kako ovo može da bude pogrešno?"**, ne "da li je dobro?".
9. **Testovi:** za svaku grešku koju nađeš napiši mali test da se ne vrati.

## 5. Kako čitati greške (primeri iz ovog projekta)
| Poruka / simptom | Uzrok | Rešenje |
|---|---|---|
| `cannot reindex on an axis with duplicate labels` | Indeks DataFrame-a ima duplikate (prva kolona CSV-a ponavlja se po pacijentu) | `reset_index(drop=True)` |
| `Axes.boxplot() got an unexpected keyword argument 'legend'` | Verzije seaborn/matplotlib se ne slažu | nadogradi/uskladi pakete |
| `pyarrow requires NumPy 2.0 or newer` / `_ARRAY_API not found` | Paketi kompajlirani za različite verzije NumPy-ja | poseban `venv`, ne globalni pip |
| Jedan fold traje 10 min, a trening 6 s | `np.isin` nad 1.5M stringova | pretvori ID u int (`pd.factorize`) |
| Nema greške, ali rezultat predobar | Leakage | testovi + ablacija (odeljak 4) |

Pravilo: traceback čitaj **od poslednje linije**, kopiraj tačnu poruku u pretragu/AI, daj i verzije paketa.

## 6. Intervju pitanja (kratki odgovori)
- *Zašto grouped split?* Isti pacijent u train i test → model "pamti" pacijenta, procena je optimistična.
- *Zašto PR-AUC?* Kod retkog pozitivnog razreda ROC-AUC je preoptimističan; PR direktno pokazuje precision.
- *Kako biraš prag?* Na validaciji, prema ciljnom recall-u/ceni greške; test samo za finalnu procenu.
- *Šta je leakage? Primer?* Informacija iz budućnosti/targeta u featurima (npr. `ffill` preko pacijenata, `ICULOS`).
- *Zašto GBM, ne neuronska mreža?* Tabelarni podaci, NaN podrška, brzina, malo tuninga.
- *Šta radiš kad su podaci 95% prazni?* NaN + indikator merenja/„sati od poslednjeg merenja“, ne mean imputacija.
- *Kako znaš da model radi u produkciji?* Monitoring drifta ulaza i skora, logovanje predikcija, periodično ponovno vrednovanje.

## 7. Šta ne moraš da pamtiš
Tačna sintaksa plotova, argumenti `groupby`, imena parametara LightGBM-a, `git` komande napamet.
**Moraš da znaš:** šta problem zahteva, koji split je ispravan, koju metriku koristiš i zašto, i kako bi rezultat mogao biti lažan.

## 8. Plan učenja (2–3 nedelje, 1 h dnevno)
1. Pandas `groupby/merge/shift/rolling` na ovom datasetu (Colab notebook).
2. `sklearn` obrazac + `GroupKFold` + metrike za neuravnotežene klase.
3. Čitanje i pisanje testova protiv leakage-a (`pytest`).
4. Jedan projekat od početka do kraja: podaci → model → API → monitoring.
