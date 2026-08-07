# Investigation — demo

Généré le 2026-08-07T21:55:13+00:00. Vue locale, complète : elle inclut les constatations de tier T4-T5 que la barrière d'externalisation refuse. Ne pas diffuser.

Constatations ouvertes : 5 (sur 5 au total, 0 résolue(s)).
Obligations au catalogue : 12.

## T1 — documenté, plusieurs sources concordantes (2)

### Intégrité du graphe

- **[T1]** `couverture_absente` — Intégrité du graphe : aucune donnée de couverture (coverage.jsonl) pour ce dossier.
  - constat : Sans coverage.jsonl, une absence ne peut pas être distinguée d'un défaut de collecte : toute constatation d'absence est plafonnée à T5.
  - id : `9b699aaa69e4`
- **[T1]** `document_sans_fait` — Intégrité du graphe : des documents extraits n'ont produit aucun fait.
  - constat : 1 document(s) sur 1 : sample_text
  - id : `ba3d079b8a89`

## T5 — hypothèse de travail — périmètre non couvert ou substrat défectueux (3)

### Contrôle des obligations

- **[T5]** `indiv-etendue-des-droits-etablie` — Obligation indiv-etendue-des-droits-etablie (Code civil, art. 815-11 al. 2) : le périmètre documentaire nécessaire au contrôle n'est pas couvert par le dossier.
  - constat : statut=unverifiable · faits_retenus=0 · périmètre=0 document(s), couvert=False
  - id : `731fe8e99609`
- **[T5]** `qu-caution-ou-dispense-expresse` — Obligation qu-caution-ou-dispense-expresse (Code civil, art. 601) : le périmètre documentaire nécessaire au contrôle n'est pas couvert par le dossier.
  - constat : statut=unverifiable · faits_retenus=0 · périmètre=0 document(s), couvert=False
  - id : `f01225e0c6ec`
- **[T5]** `qu-emploi-des-sommes-a-defaut-de-caution` — Obligation qu-emploi-des-sommes-a-defaut-de-caution (Code civil, art. 602) : le périmètre documentaire nécessaire au contrôle n'est pas couvert par le dossier.
  - constat : statut=unverifiable · faits_retenus=0 · périmètre=0 document(s), couvert=False
  - id : `6f10185ca008`
