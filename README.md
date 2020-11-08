# Community Identification Using Last.fm Data



***
# Files

* [Last.fm](https://www.last.fm/) is an internet radio and music community based in the UK

  * $n_{node} = 7624$

  * $n_{edge} = 27806$

* [Dataset](https://github.com/benedekrozemberczki/datasets/blob/master/lasftm_asia/lastfm_asia_edges.csv) [<sup>[1]</sup>](#data_source): [lastfm_asia_edges.csv](lastfm_asia_edges.csv)

* Python script: [community_identification.py](community_identification.py)



***
# Model

* Graph features such as power law distribution of degree

* Community identifying using algorithms of Modularity, Label Propagation, Girvan-Newman, and Louvain

* Measuring performances of each algorithm by methods of coverage, modularity, and performance

* Visualization of original graph and community



***
## Reference
<div id="data_source"></div>
[1] Benedek Rozemberczki, and Rik Sarkar 2020. Characteristic Functions on Graphs: Birds of a Feather, from Statistical Descriptors to Parametric Models. In Proceedings of the 29th ACM International Conference on Information and Knowledge Management (CIKM '20) (pp. 1325–1334).