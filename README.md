# PINN Super-Resolution

Implementazione modulare di una PINN (Physics-Informed Neural Network) per
Super-Resolution di una singola immagine. Il problema e' formulato come
problema inverso mal-posto

```
y = P u + eta                    con  P = S . B
```

dove `u` e' l'immagine HR cercata, `y` e' la LR osservata, `B` e' un blur
gaussiano, `S` il sottocampionamento stride, `eta` del rumore. La rete
`u_theta : [0,1]^2 -> R^3` parametrizza la soluzione; l'ottimizzatore
minimizza

```
L = lambda_data * || P u_theta - y ||^2 + lambda_pde * || N[u_theta] ||^2 + ...
```

dove `N[.]` e' l'operatore differenziale di una PDE di diffusione che
regolarizza la soluzione. La PDE e' il **prior adattivo**: sceglie *quali*
alte frequenze introdurre quando `P` ha un kernel non banale.

---

## Struttura

```
upscaling/
├── config.py             # tutto il config (dataclass)
├── main.py               # entry point CLI
│
├── data/
│   ├── download.py       # scarica Set5 / DIV2K (fallback su immagini singole)
│   ├── dataset.py        # split 80/10/10 + loader PIL
│   └── single_image.py   # coordinate HR + punti-dato riposizionati da LR
│
├── degradation/
│   └── operator.py       # P = S . B, aggiunto P*, dot-product test
│
├── models/
│   ├── siren.py          # SIREN w0=30 (Sitzmann 2020)
│   └── fourier_mlp.py    # Fourier-feature MLP (Tancik 2020)
│
├── pde/
│   ├── operators.py      # grad / div via torch.autograd
│   └── diffusion.py      # Perona-Malik + tensore anisotropo vettoriale
│
├── losses/
│   ├── registry.py       # registry estensibile (@register("nome"))
│   ├── data_loss.py      # data_lr, data_points
│   ├── pde_loss.py       # pde_perona_malik, pde_anisotropic
│   ├── bc_loss.py        # bc_neumann
│   └── sharpness_loss.py # reg_tv, pde_shock
│
├── training/trainer.py   # loop + curriculum + bicubic init + snapshot
└── utils/metrics.py      # PSNR, save_image, save_triptych
```

---

## Termini di loss disponibili

I termini si selezionano da CLI tramite `--loss` e i pesi si regolano in
[config.py](config.py) (`LossConfig.weights`).

### Fedelta' ai dati (obbligatorio almeno uno)

| nome | formula | note |
|---|---|---|
| `data_lr` | `|| P u_theta - y_LR ||^2` su griglia HR completa | **il termine principale**: forza `u` ad avere struttura sub-pixel tale che blur+downsample riproduca `y` |
| `data_points` | MSE sui pixel LR riposizionati ai centri di cella `(i+0.5)/w` | utile come helper, ma da solo e' debole: lascia libera l'interpolazione fra i punti |

### Prior PDE (edge-preserving)

| nome | PDE | effetto |
|---|---|---|
| `pde_perona_malik` | `div( g(|grad u|^2) * grad u ) = 0` | baseline scalare, canali indipendenti |
| `pde_anisotropic` | `div( D(u) * grad u_c ) = 0` con `D(u) = Q diag(g1,g2) Q^T` e `G(u) = sum_c grad u_c grad u_c^T` | vettoriale: accoppia R/G/B tramite il tensore di struttura |

Questi termini **diffondono lungo le isofote** e *frenano* la diffusione
trasversale ai bordi. Preservano i bordi, non li inventano.

### Termini per la nitidezza

| nome | tipo | effetto |
|---|---|---|
| `reg_tv` | prior energetico `E[ sqrt(sum_c |grad u_c|^2) ]` | Total Variation vettoriale. Sparsifica i gradienti -> interni piatti, bordi netti (ROF / Rudin-Osher-Fatemi) |
| `pde_shock` | residuo `sign(u_eta_eta) * |grad u|` | Osher-Rudin shock filter. "Anti-diffusione" sui bordi: spinge verso jump netti |

### Condizioni al bordo

| nome | effetto |
|---|---|
| `bc_neumann` | `partial_n u = 0` sui 4 bordi di `[0,1]^2` (nessun flusso) |

---

## Dettagli numerici importanti

### Normalizzazione del residuo PDE

Le coordinate sono in `[0,1]^2`, ma l'immagine ha `H x W` pixel. Lo spacing
fisico e' `1/H`. Il residuo `div(D . grad u)` calcolato in coord `[0,1]^2`
e' **`H^2` volte** quello in coord-pixel. Per avere magnitudini O(1) e pesi
interpretabili, [losses/pde_loss.py](losses/pde_loss.py) divide il residuo
per `coord_scale^2 = max(H,W)^2`. Senza questa normalizzazione i pesi PDE
dovrebbero essere O(1e-8) per non esplodere.

### Curriculum

Il peso PDE e' moltiplicato da un fattore lineare 0->1 fra
`pde_warmup_epochs` e `pde_warmup_epochs + pde_ramp_epochs`. Serve per non
destabilizzare il training all'inizio quando la rete e' ancora lontana
dalla soluzione.

### Bicubic init

La rete viene pre-addestrata a replicare l'upsampling bicubico della LR
(2000 step, Adam con cosine annealing). Da questa inizializzazione la
PINN parte da un PSNR gia' ragionevole; senza pre-training converge molto
piu' lentamente e spesso si ferma in minimi pessimi.

### Gradient clipping

`torch.nn.utils.clip_grad_norm_(..., 1.0)` per evitare blow-up durante le
prime epoche del curriculum PDE.

### Regolarizzazione del tensore di struttura

Prima di `torch.linalg.eigh(G)` si aggiunge `eps * I` a `G`: evita che la
decomposizione in autovalori sia instabile quando i gradienti locali sono
~0.

---

## Come bilanciare i pesi (empirico)

Regola di base:

```
peso_dati * MSE_dati  ~>=  peso_prior * residuo_prior
```

Se il prior e' piu' forte del dato, la rete **ignora `y_LR`** e converge a
una soluzione piatta/posterizzata tipica del prior stesso (TV porta a
piecewise-constant con colori uniformi, Shock idem).

Default in [config.py](config.py):

```python
"data_lr":          1.0
"data_points":      1.0
"pde_perona_malik": 2e-2
"pde_anisotropic":  2e-2
"bc_neumann":       1e-3
"reg_tv":           2e-3
"pde_shock":        5e-3
pde_warmup_epochs = 50
pde_ramp_epochs   = 200
```

### Diagnostica tipica

| sintomo | causa | fix |
|---|---|---|
| PSNR fermo alla bicubica | solo `data_points`, niente `data_lr` | aggiungi `data_lr` |
| PSNR fermo a fit LR, non si muove | prior troppo debole / PDE gia' soddisfatta dalla bicubica | alza pesi PDE, shock/TV |
| Immagine sbiadita / posterizzata | prior troppo forte | abbassa `reg_tv`, `pde_shock` |
| Blow-up del loss | PDE senza normalizzazione o warmup zero | verifica `coord_scale`, aumenta warmup |
| Colori drift | TV domina, non c'e' ancoraggio ai dati | alza `data_lr` o abbassa `reg_tv` |

---

## Uso

### Download del dataset

Al primo avvio scarica automaticamente Set5 (o DIV2K) in `./datasets/`. In
caso di URL non disponibile, c'e' un fallback che scarica singole immagini
di test. Puoi anche passare direttamente un path locale:

```bash
python main.py --image /path/to/my_image.png
```

### Esempi di training

**Baseline ROF (semplice e robusto):**
```bash
python main.py --loss data_lr reg_tv --epochs 2000
```

**Consigliato (edge-preserving + sparsificante):**
```bash
python main.py --loss data_lr pde_anisotropic reg_tv --epochs 2000
```

**Aggressivo sulla nitidezza:**
```bash
python main.py --loss data_lr pde_anisotropic pde_shock --epochs 2000
```

**Scale 2 invece di 4** (problema molto piu' facile, margini piu' ampi):
```bash
python main.py --loss data_lr pde_anisotropic pde_shock --epochs 1500 --scale 2
```

### Output

Nella directory `./checkPoints/`:

- `model.pt` — pesi della rete a fine training
- `pred_hr.png` — predizione finale
- `input_lr.png`, `gt_hr.png` — riferimenti

In `./checkPoints/snapshots/`:

- `compare_epXXXXX.png` — triptych `[HR | LR (nearest) | pred]` ogni
  `snapshot_every` epoche, con PSNR corrente in titolo.

---

## Cos'e' e cosa NON e' questa PINN

**E'**: un regolarizzatore adattivo dipendente dai dati. Il coefficiente di
diffusione `D(u)` dipende da `u` stessa (via tensore di struttura). Nelle
zone piatte `D ~ I` (smoothing isotropo), sui bordi `D` ha un autovalore
piccolo in direzione normale al bordo -> diffusione solo lungo l'isofota.
E' vettoriale: un bordo visibile solo in un canale vincola anche gli altri
(evita color fringing).

**NON e'**: un modello che apprende da un dataset. La PINN e' single-image
zero-shot: non usa mai altre immagini. Il suo valore e' l'interpretabilita'
e l'assenza di training offline, ma il tetto di PSNR raggiungibile senza
prior appresi e' piu' basso di EDSR/SwinIR/diffusion (~1-3 dB sopra
bicubica contro i ~10 dB dei modelli SOTA).

**Loss vs metrica**: il PSNR mostrato in training e' calcolato contro la
HR ground-truth (che conosciamo solo perche' abbiamo sintetizzato noi
la LR). Non entra nella loss — sarebbe barare rispetto al problema
inverso. La loss vede solo `y_LR` e i prior.

---

## Aggiungere una nuova loss

Bastano 3 righe. In un file nuovo dentro `losses/`:

```python
from .registry import register

@register("il_mio_termine")
def my_loss(net, collocation, coord_scale=1.0, **_):
    coords = collocation.clone().requires_grad_(True)
    # ... calcola il tuo residuo usando pde.operators.channel_gradients
    return (residual ** 2).mean()
```

Poi importalo in [losses/\_\_init\_\_.py](losses/__init__.py), aggiungi un
peso in `config.LossConfig.weights` e attivalo con
`--loss ... il_mio_termine`.
