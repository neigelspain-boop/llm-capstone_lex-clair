lex-clair/
├── README.md                     # problem, run instructions, screenshots
├── pyproject.toml                # deps, pinned via uv
├── uv.lock                       # committed — reproducibility 2 pts
├── .python-version
├── .env.example                  # OPENAI_API_KEY=, ANTHROPIC_API_KEY=, POSTGRES_*
├── .gitignore                    # .env, data/raw/, data/chroma/, __pycache__/
├── Dockerfile                    # app image
├── docker-compose.yml            # app + postgres + grafana — containerization 2 pts
├── app.py                        # Streamlit entry — interface 2 pts
│
├── ingestion/                    # PLANE I — offline, re-run on corpus change
│   ├── __init__.py               # re-exports load_index
│   ├── fetch.py                  # PISTE → data/raw/
│   ├── parse.py                  # data/raw/ → data/articles.csv
│   ├── chunk.py                  # data/articles.csv → data/chunks.csv
│   ├── index.py                  # data/chunks.csv → BM25 + Chroma
│   ├── build.py                  # CLI: python -m ingestion.build
│   └── load.py                   # load_index() — hot-path interface
│
├── rag/                          # PLANE II — online, hot path
│   ├── __init__.py               # re-exports answer
│   ├── rewrite.py                # plain FR → legal register  [BP: query rewriting +1]
│   ├── retrieve.py               # BM25 + vector fusion       [BP: hybrid search +1]
│   ├── rerank.py                 # bge-reranker-v2-m3         [BP: reranking +1]
│   ├── prompt.py                 # templates + builder
│   ├── generate.py               # Anthropic + OpenAI clients
│   └── flow.py                   # answer(question) — orchestrator
│
├── eval/                         # PLANE III — measurement
│   ├── __init__.py
│   ├── ground_truth.py           # generates data/ground_truth.csv
│   ├── retrieval_eval.py         # Hit Rate / MRR, multi-approach — 2 pts
│   └── llm_eval.py               # LLM-as-judge, multi-approach — 2 pts
│
├── monitoring/                   # PLANE IV — runtime
│   ├── __init__.py
│   ├── db.py                     # Postgres: save_conversation, save_feedback
│   └── grafana/
│       ├── init.py               # scripted provisioning
│       └── provisioning/
│           ├── datasources/postgres.yml
│           └── dashboards/dashboards.yml
│
├── data/                         # artifact landing zone (plane-agnostic)
│   ├── raw/                      # gitignored — PISTE payloads
│   ├── articles.csv              # committed — inspectable by reviewers
│   ├── chunks.csv                # committed — inspectable by reviewers
│   ├── chroma/                   # gitignored — rebuilt on container start
│   ├── ground_truth.csv          # committed
│   ├── retrieval_eval_results.csv
│   └── llm_eval_results.csv
│
└── docs/                         # reference material, screenshots for README
    ├── lex-clair-system-map.pdf
    ├── lex_clair_marche_a_suivre.pdf
    └── screenshots/

Ten-day build order
Fixed sequence. Each day ends with a git commit tagged with the rubric line completed. If a day slips, drop cloud deploy first (–2), then reranking (–1). Never drop from the core 18.

1. ingestion/fetch.py + parse.py running end to end data/articles.csv

2. chunk.py + index.py +build.py + load.py complete 

3. eval/ground_truth.py -> data/ground_trut.csv; retrievall_eval.py. run on baseline(BM25-only)

4. rag/retreive.py (hybrid fusion) + rewrite.py;
retreival_eval compares 4 approaches, picks winner. 

5. rag/rerank.py + prompt.py + generate.py;
retreival_eval compares 4 approaches, pick winner

6. eval/ llm_eval.py compares claude vs GPT on ~200 samples. 

7. app.py streamlit with question input, cited, feedback buttons. 

8. monitoring/db.py + pootgres schema; 
feedback wired end-to-end; Grafana dashboard with 5 charts

9. Dockfile + docker-compose.yml for app + postgres + grafana; 
uv.lock committed; README problem statement + run instruction + screenshot. 

10. HF space deploy + READMEURL + buffer for peer review polish


## CODE USED ##
1. Code civil ;
Successions (réserve, quotité, rapport, réduction), usufruit (578, 587, 600-601), responsabilité délictuelle (1240-1241) — the case's spine

2. Code general des impots (CGI);
641 (délai déclaration), 774 bis (anti-abus quasi-usufruit LF 2024), 1133 (exonération extinction usufruit), 1727 (intérêts de retard)

3. Code de proc/dure civile; 
How to actually sue: prescription, assignation, tribunal judiciaire compétence, référé, expertise

4. Code des assurances;
Action directe contre RC pro notaire (L.124-3), garantie dans le temps (L.124-5)

5. Livre des procedures fiscales;
Contestation d'un redressement, réclamation contentieuse, délais

6. Code penal;
Abus de confiance (314-1), abus de faiblesse (223-15-2), faux (441-1) — the criminal angle you invoked with "swallowed"

7. Code de l'organisation judiciaire; 
Small — just competent court rules. Skip unless bandwidth allows


# LODA TEXT #

    1. Ordonnance n° 45-2590;
        Statut du notariat — the notaire's obligations
    
    2. Décret n° 73-609
        Règles professionnelles notariales

    3. Décret n° 74-737
        Discipline (sanctions disciplinaires du notaire)
    
    4. Décret n° 2023-1297
        Already on your list

