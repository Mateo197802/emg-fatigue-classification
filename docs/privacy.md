# Public code and private research outputs

The course guide requires public code and forbids redistribution of the course dataset. Source recordings, original paths, identity mappings, individual feature/label tables, OOF predictions, private manuscript and signal dashboards remain outside the checkout. Output CLI roots inside the repository are rejected.

Public results are an explicit export of model name, opaque fold key and whitelisted performance/count fields. The private original group key, names, input paths, model parameters containing context and signal samples are excluded. The opaque mapping salt remains private and is reused to make repeated exports stable. These metrics are deidentified research results; opacity alone does not promise mathematical anonymity against every auxiliary dataset. The public release contains no identity lookup or source archive membership.

This resolves the template's request for a committed model×fold results.csv without publishing the dataset or a person-linked table. The teacher's complete results.csv and model are available in the private deliverable. The model remains private as a derivative of restricted data. CI uses synthetic software fixtures only, and never uploads course data to GitHub Actions.

The original report template is supplied locally to the manuscript builder. University logos and templates are preserved only in private documents. Authorship, consent, calibrated hardware provenance and balanced student contributions cannot be inferred from repository authorship; missing attestations remain explicitly unresolved.
