# LUCI: Linguistic Uncertainty Classifier Interface

Pronounced: &#91;lusi&#93;

### Description

A Python implementation of a classifier for linguistic uncertainty classifer. This is loosely based on the work described in:

```
Vincze, V. (2015). Uncertainty detection in natural language texts (Doctoral dissertation, szte).
```

### Corpora

This classifier was trained using the human-annotated [Szeged Uncertainty Corpus](http://rgai.inf.u-szeged.hu/index.php?lang=en&page=uncertainty), which is composed of three databases - BioScape 2.0, FactBank 2.0, and WikiWeasel 2.0.

```
    1) Vincze, V., Szarvas, G., Farkas, R., Móra, G., & Csirik, J. (2008). The BioScope corpus: biomedical texts annotated for uncertainty, negation and their scopes. BMC bioinformatics, 9(11), S9.
    2) Saurí, R., & Pustejovsky, J. (2009). FactBank: a corpus annotated with event factuality. Language resources and evaluation, 43(3), 227.
    3) Farkas, R., Vincze, V., Móra, G., Csirik, J., & Szarvas, G. (2010, July). The CoNLL-2010 shared task: learning to detect hedges and their scope in natural language text. In Proceedings of the Fourteenth Conference on Computational Natural Language Learning---Shared Task (pp. 1-12). Association for Computational Linguistics.
```

Links: 1) [BioScope](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-9-S11-S9), 2) [FactBank](https://link.springer.com/article/10.1007/s10579-009-9089-9), and 3) [WikiWeasel](https://www.researchgate.net/profile/Domonkos_Tikk2/publication/234786271_A_simple_ensemble_method_for_hedge_identification/links/0912f5108e2a8a5ca2000000.pdf#page=13).

### Install

Use the following commands to install dependencies:

```
    git clone https://github.com/meyersbs/uncertainty.git
    cd uncertainty/
    pip -r requirements.txt
```

### Usage

* Coming soon!

### Contact
If you have questions regarding this API, please contact [bsm9339@rit.edu](mailto:bsm9339@rit.edu) (Benjamin Meyers).

For questions regarding the annotated dataset or the theory behind the uncertainty classifier, please contact [szarvas@inf.u-szeged.hu](mailto:szarvas@inf.u-szeged.hu) (György Szarvas), [rfarkas@inf.u-szeged.hu](mailto:rfarkas@inf.u-szeged.hu) (Richárd Farkas), and/or [vinczev@inf.u-szeged.hu](mailto:vinczev@inf.u-szeged.hu) (Veronika Vincze).
