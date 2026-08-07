# Investigation — vitrine

Généré le 2026-08-07T21:35:54+00:00. Vue locale, complète : elle inclut les constatations de tier T4-T5 que la barrière d'externalisation refuse. Ne pas diffuser.

Constatations ouvertes : 13 (sur 14 au total, 1 résolue(s)).
Obligations au catalogue : 12.

## T1 — documenté, plusieurs sources concordantes (2)

### Intégrité du graphe

- **[T1]** `couverture_absente` — Intégrité du graphe : aucune donnée de couverture (coverage.jsonl) pour ce dossier.
  - constat : Sans coverage.jsonl, une absence ne peut pas être distinguée d'un défaut de collecte : toute constatation d'absence est plafonnée à T5.
  - id : `9b699aaa69e4`
- **[T1]** `role_ambigu` — Intégrité du graphe : des faits portent une attribution de rôle non résolue.
  - constat : 20 fait(s) sur 235 ; exclus des prédicats portant sur un rôle, et plafonnés à T5 lorsqu'ils étayent une constatation.
  - id : `b927f56e0334`

## T2 — documenté, source unique (8)

### Contrôle des obligations

- **[T2]** `assur-reclamation-prealable-a-la-garantie` — Obligation assur-reclamation-prealable-a-la-garantie (Code des assurances, art. L. 124-1) : l'exécution est attestée par le dossier.
  - constat : statut=satisfied · faits_retenus=2 · périmètre=13 document(s), couvert=False · déclencheur=camille_martin_s_01_creance_convention_de_quasi_usufruit_du_ref-f003 (2024-02-05) · faits=camille_martin_s_00_note_de_presentation-f005, camille_martin_s_00_note_de_presentation-f006
  - id : `30f988033f3b`
- **[T2]** `fisc-passif-dettes-justifiees` — Obligation fisc-passif-dettes-justifiees (Code général des impôts, art. 768) : l'exécution est attestée par le dossier.
  - constat : statut=satisfied · faits_retenus=13 · périmètre=26 document(s), couvert=False · déclencheur=camille_martin_s_01_creance_x_reconnaissance_de_x_ref-f001 (2022-03-24) · faits=camille_martin_s_00_note_de_presentation-f001, camille_martin_s_00_note_de_presentation-f003, camille_martin_s_02_successions_02b_comptes_et_actifs_avis_d_impots_remboursement_1761_eur_ref-f002, camille_martin_s_04_precontentieux_ref_camille_a_x_demande_d_information-f003, camille_martin_s_04_precontentieux_ref_camille_reponse_a_sophie_dubois_0609-f003, camille_martin_s_04_precontentieux_ref_camille_reponse_a_sophie_dubois_0609-f005
  - id : `76bb94ed6865`
- **[T2]** `indiv-etendue-des-droits-etablie` — Obligation indiv-etendue-des-droits-etablie (Code civil, art. 815-11 al. 2) : l'exécution est attestée par le dossier.
  - constat : statut=satisfied · faits_retenus=11 · périmètre=31 document(s), couvert=False · faits=camille_martin_s_01_creance_x_etude_relsophie_des_parts_x_ref-f003, camille_martin_s_02_successions_02a_actes_de_x_acte_de_notoriete_henri_martin_aae_annexes-f001, camille_martin_s_02_successions_02a_actes_de_x_attestation_devolutive_acte_n_ref_ref-f004, camille_martin_s_02_successions_02a_actes_de_x_attestation_devolutive_acte_n_ref_ref-f007, camille_martin_s_02_successions_02a_actes_de_x_declaration_x_mme_nomne_martin_ref-f005, camille_martin_s_04_precontentieux_ref_camille_a_x_demande_d_information-f003
  - id : `e3a7751cc392`
- **[T2]** `indiv-prelevement-creancier-avant-partage` — Obligation indiv-prelevement-creancier-avant-partage (Code civil, art. 815-17 al. 1) : l'exécution est attestée par le dossier.
  - constat : statut=satisfied · faits_retenus=2 · périmètre=27 document(s), couvert=False · déclencheur=camille_martin_s_04_precontentieux_ref_sophie_dubois_response_a_camille_28th_premiere_mention_de_l_acte_1999-f002 (1999-08-26) · faits=camille_martin_s_04_precontentieux_ref_camille_saisine_77-f003, camille_martin_s_04_precontentieux_ref_camille_saisine_77-f004
  - id : `3ba9b2e5be95`
- **[T2]** `qu-emploi-des-sommes-a-defaut-de-caution` — Obligation qu-emploi-des-sommes-a-defaut-de-caution (Code civil, art. 602) : l'exécution est attestée par le dossier.
  - constat : statut=satisfied · faits_retenus=2 · périmètre=26 document(s), couvert=False · faits=camille_martin_s_04_precontentieux_ref_camille_saisine_77-f001, camille_martin_s_04_precontentieux_ref_camille_saisine_77-f003
  - id : `3dc305eaab36`
- **[T2]** `qu-restitution-fin-usufruit` — Obligation qu-restitution-fin-usufruit (Code civil, art. 587) : l'exécution est attestée par le dossier.
  - constat : statut=satisfied · faits_retenus=15 · périmètre=24 document(s), couvert=False · déclencheur=camille_martin_s_02_successions_02a_actes_de_x_declaration_x_mme_nomne_martin_ref-f002 (1981-02-05) · faits=camille_martin_s_00_note_de_presentation-f001, camille_martin_s_00_note_de_presentation-f002, camille_martin_s_00_note_de_presentation-f003, camille_martin_s_04_precontentieux_ref_camille_a_x_demande_d_information-f002, camille_martin_s_04_precontentieux_ref_camille_a_x_demande_d_information-f003, camille_martin_s_04_precontentieux_ref_camille_saisine_77-f001
  - id : `ddd6a45a5d6c`
- **[T2]** `resp-reparation-du-dommage-cause` — Obligation resp-reparation-du-dommage-cause (Code civil, art. 1240) : l'exécution est attestée par le dossier.
  - constat : statut=satisfied · faits_retenus=2 · périmètre=40 document(s), couvert=False · déclencheur=camille_martin_s_04_precontentieux_ref_camille_saisine_77-f007 (2026-05-01) · faits=camille_martin_s_01_creance_x_etude_relsophie_des_parts_x_ref-f003, camille_martin_s_04_precontentieux_ref_camille_reponse_a_sophie_dubois_0609-f006
  - id : `5540ae4b360e`
- **[T2]** `succ-deblocage-dans-la-proportion-de-l-acte` — Obligation succ-deblocage-dans-la-proportion-de-l-acte (Code civil, art. 730-4 ; art. 815-3 in fine) : l'exécution est attestée par le dossier.
  - constat : statut=satisfied · faits_retenus=3 · périmètre=29 document(s), couvert=False · déclencheur=camille_martin_s_00_note_de_presentation-f001 (2024-03-08) · faits=camille_martin_s_01_creance_x_etude_relsophie_des_parts_x_ref-f003, camille_martin_s_01_creance_x_etude_relsophie_des_parts_x_ref-f004, camille_martin_s_04_precontentieux_ref_camille_a_x_demande_d_information-f003
  - id : `9b0428410120`

## T5 — hypothèse de travail — périmètre non couvert ou substrat défectueux (3)

### Contrôle des obligations

- **[T5]** `not-conservation-du-depot-et-delivrance` — Obligation not-conservation-du-depot-et-delivrance (Ordonnance n° 45-2590 du 2 novembre 1945, art. 1er) : le périmètre documentaire nécessaire au contrôle n'est pas couvert par le dossier.
  - constat : statut=unverifiable · faits_retenus=0 · périmètre=32 document(s), couvert=False · déclencheur=camille_martin_s_02_successions_02b_comptes_et_actifs_facture_edf_villeneuve_ref-f001 (2026-04-19) · candidats=camille_martin_s_02_successions_02c_immobilier_villeneuve_donation_partage_1999_incomplet_ref-f002, camille_martin_s_00_note_de_presentation-f001, camille_martin_s_00_note_de_presentation-f003, camille_martin_s_01_creance_convention_de_quasi_usufruit_du_ref-f001, camille_martin_s_01_creance_x_acte_facture_n_fref_ref-f001, camille_martin_s_01_creance_x_acte_facture_n_fref_ref-f003
  - id : `34a7eef16b8b`
- **[T5]** `qu-caution-ou-dispense-expresse` — Obligation qu-caution-ou-dispense-expresse (Code civil, art. 601) : le périmètre documentaire nécessaire au contrôle n'est pas couvert par le dossier.
  - constat : statut=unverifiable · faits_retenus=0 · périmètre=25 document(s), couvert=False · candidats=camille_martin_s_00_note_de_presentation-f001, camille_martin_s_00_note_de_presentation-f002, camille_martin_s_00_note_de_presentation-f003, camille_martin_s_01_creance_convention_de_quasi_usufruit_du_ref-f001, camille_martin_s_01_creance_convention_de_quasi_usufruit_du_ref-f002, camille_martin_s_01_creance_convention_de_quasi_usufruit_du_ref-f003
  - id : `f01225e0c6ec`
- **[T5]** `not-garantie-rcp-mobilisable` — Obligation not-garantie-rcp-mobilisable (Ordonnance n° 45-2590 du 2 novembre 1945, art. 6-2) : le périmètre documentaire nécessaire au contrôle n'est pas couvert par le dossier.
  - constat : statut=unverifiable · faits_retenus=0 · périmètre=32 document(s), couvert=False · déclencheur=camille_martin_s_04_precontentieux_ref_conseil_regional_des_notaires_de_la_cour_d_appel_de_beaumont_ii_confirmation_de_reception_de_saisine_77-f001 (2026-06-24) · candidats=camille_martin_s_00_note_de_presentation-f001, camille_martin_s_00_note_de_presentation-f003, camille_martin_s_01_creance_convention_de_quasi_usufruit_du_ref-f001, camille_martin_s_01_creance_x_acte_facture_n_fref_ref-f001, camille_martin_s_01_creance_x_acte_facture_n_fref_ref-f003, camille_martin_s_01_creance_x_etude_relsophie_des_parts_x_ref-f003
  - id : `e92bcf659352`
