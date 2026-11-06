# Data license

The contents of `data/`, `results/`, `figures/` and `dashboard/data.json` are released under the
Creative Commons Attribution 4.0 International License (CC BY 4.0),
https://creativecommons.org/licenses/by/4.0/. You may share and adapt them for any purpose,
including commercially, as long as you give appropriate credit.

These files are derived from SMART-DS, which is also released under CC BY 4.0:

> B. Palmintier, C. Mateo Domingo, F. Postigo Marcos, T. Gomez San Roman, F. de Cuadra,
> N. Gensollen, T. Elgindy, and P. Duenas, *SMART-DS: Synthetic Electrical Network Data OpenDSS
> Models for SFO, GSO, and AUS*, National Renewable Energy Laboratory, Open Energy Data Initiative
> submission 2981, 2020. https://data.openei.org/submissions/2981

Changes made to the SMART-DS source: the two 120 V legs of each center-tapped residential service
were combined, customer loads were summed per service transformer and converted from kW to
apparent power with a conversion fitted to OpenDSS power flow, a stratified sample of 1,500
transformers was drawn, and the forecasting results in `results/` were computed from that panel.

When you reuse these files, please credit both this repository (see `CITATION.cff`) and SMART-DS.
The code in this repository is licensed separately under the MIT License (`LICENSE`).
