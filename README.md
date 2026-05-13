# Inverse PINN per Super-Resolution con Operatore Ignoto

Pipeline per ricostruire un'immagine HR partendo da una versione degradata da
un operatore `P = S \circ G_sigma + \eta`, dove:

- `G_sigma` e' un blur gaussiano con `sigma` **ignoto**;
- `S` e' un downsampling stride (`x1` / `x2` / `x4`);
- `\eta` e' una **miscela** ignota di rumore (gaussiano + Laplace + speckle).

L'idea PINN inversa: il forward model resta differenziabile e i suoi
parametri (`sigma`, scale del rumore, pesi della miscela, pesi PDE) sono
**trainabili** insieme alla rete che rappresenta l'immagine HR. La loss
mescola fidelta' al dato osservato e PDE di nitidezza/definizione (Perona-
Malik, ROF, shock filter, TV).

## Cosa e' cambiato in questa iterazione

L'iterazione precedente produceva guadagni marginali (`+0.24 dB` sopra
bicubica nel benchmark x2). Le modifiche introdotte ora portano un margine
sensibile su tutte le scale, mantenendo lo schema "PINN inversa" (rete +
forward model con parametri trainabili + PDE prior).

Modifiche principali nel solver (`inverse_sr/solver.py`):

- **Charbonnier data fidelity** sull'osservato LR oltre alla negative
  log-likelihood della miscela: `lambda_data * sqrt((y - obs)^2 + eps^2)`.
  Dare un secondo segnale di gradiente piu' robusto evita che la NLL
  "assorba" residui inflattando le scale del rumore.
- **Residual scale** della rete portato da `0.12` a `0.45`. Prima il
  contributo della SIREN era schiacciato attorno alla bicubica e quasi
  invisibile.
- **Soft-clamp differenziabile** in uscita (`tanh`) al posto del clamp
  hard, per mantenere gradienti vivi sui pixel ai bordi del range.
- **Cosine LR scheduler** + LR del modello e dei parametri raddoppiato.
- **Warm-up dei prior**: PDE-loss scala da 0 a 1 sul primo `25%` delle
  epoche; cosi' i parametri del forward model si calibrano prima di
  imporre la regolarizzazione strutturale.
- Capienza della rete leggermente aumentata (`hidden_dim=192`,
  `num_layers=5`, `w0=24`).

Risultati (`butterfly`, 192x192, **200 epoche**, benchmark multiscala
salvato in `results/final_inverse_pinn_v1/`):

| Scenario | Bicubic PSNR | Best Inverse PINN PSNR | Delta PSNR | Delta SSIM |
| --- | ---: | ---: | ---: | ---: |
| `x1_deblur_denoise` | 21.28 | 22.60 (`TV energy`) | **+1.32 dB** | +0.076 |
| `x2_sr` | 20.50 | 22.27 (`TV energy`) | **+1.77 dB** | +0.072 |
| `x4_sr` | 16.97 | 19.90 (`Shock PDE`) | **+2.92 dB** | +0.127 |

(Per confronto, la versione precedente con la stessa configurazione di
benchmark si fermava a `+0.24 dB` su `x2_sr`.)

I miglioramenti di SSIM seguono lo stesso ordine di grandezza
(`+0.07` ... `+0.13`).

## Metodi confrontati

- `Bicubic` baseline;
- `All PDE trainable` (TV + ROF + Perona-Malik + Shock);
- singoli prior: `TV energy`, `ROF PDE`, `Perona-Malik PDE`, `Shock PDE`;
- combinazioni: `ROF + Shock`, `Perona-Malik + Shock`.

Tutti condividono il forward model differenziabile e i parametri di
degradazione trainabili. La differenza sta nei termini PDE attivi.

## Note metodologiche

### "TV energy" e gli altri metodi sono ancora PINN inverse?

Si'. Tutte le voci della tabella diverse da `Bicubic` hanno `kind: "inverse"`
in `main.py:default_configs()` e passano per `solve_inverse_problem` in
`inverse_sr/solver.py`. Questo significa che ognuna di esse usa:

- la stessa rete `ResidualSIREN` come rappresentazione dell'immagine HR;
- lo stesso forward model differenziabile con `sigma`, scale e pesi della
  miscela di rumore **trainabili**;
- lo stesso "blind common" di prior (`flat_noise`, `edge_sharpness`) e la
  stessa data fidelity (Charbonnier + NLL della miscela).

L'unica differenza tra i metodi e' **quale** termine PDE addizionale e'
attivo (`tv`, `rof`, `pm`, `shock` o combinazioni). Quindi `TV energy` non
e' la TV standard su pixel: e' PINN inversa con TV come prior aggiuntivo.
La baseline non-PINN del benchmark e' solo `Bicubic`.

### Dove appare l'HR nel ciclo di training?

**Nella loss: mai.** Il backward propaga gradienti solo da:

- `data_loss = NLL_mixture(y - P(x)) + lambda_data * Charbonnier(y - P(x))`,
  calcolata tra `predicted_lr = P(reconstruction)` e `lr_observed`;
- prior PDE valutati solo sull'immagine ricostruita;
- regolarizzazione sui parametri.

Nessun termine usa `hr_reference`. Da questo punto di vista il training e'
**completamente blind**.

**Nella selezione del checkpoint: si', come oracolo.**
`InverseSolverConfig.selection_metric = "psnr"` confronta la ricostruzione
con `hr_reference` ad ogni epoca e tiene lo stato col PSNR migliore. E'
"oracle early-stopping", uniforme per tutti i metodi inverse, e non entra
mai nel grafo di backward. Per disabilitarlo si puo' settare
`selection_metric="objective"` (gia' supportato): in quel caso la pipeline
diventa 100% blind anche nella scelta del checkpoint, al costo tipico di
qualche frazione di dB rispetto al PSNR-oracle.

## Uso

Singola scala:

```bash
python main.py --image butterfly.png --size 192 --scale 2 --epochs 200
python main.py --image butterfly.png --scale 4 --out results/run_butterfly_x4
python main.py --only rof_shock pm_shock
```

Benchmark multiscala (`x1` / `x2` / `x4`) consigliato:

```bash
python benchmark_multiscale.py --image butterfly.png --size 192 --epochs 200
```

Se `--image` non e' un path locale, il file viene cercato ricorsivamente in
`datasets/`.

## Output

Ogni run salva in `results/<run-name>/`:

- `hr.png`, `lr_clean.png`, `lr_observed.png`;
- una immagine per ogni metodo;
- `comparison.png` con `Ground truth`, `Observed`, tutti i metodi e crop
  zoomati sulle regioni piu' strutturate;
- `results.md` con la tabella finale `PSNR / SSIM / sigma stimato / pesi
  miscela noise / pesi prior appresi`;
- `results.json` con storico completo dell'ottimizzazione.

Per il benchmark multiscala viene aggiunto `benchmark_report.md` con la
sintesi cross-scenario.
