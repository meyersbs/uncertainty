# LUCI: Linguistic Uncertainty Classifier Interface

Pronounced: &#91;lusi&#93;

### Description

A Python implementation of a classifier for linguistic uncertainty classifer. This is loosely based on the work described in:

```
<b>[1]</b> Vincze, V. (2015). Uncertainty detection in natural language texts (Doctoral dissertation, szte).
```

### Corpora

This classifier was trained using the human-annotated [Szeged Uncertainty Corpus](http://rgai.inf.u-szeged.hu/index.php?lang=en&page=uncertainty), which is composed of three sub-corpora - BioScape 2.0<b>[`[2]`](#f2)</b></sup>, FactBank 2.0<b>[`[3]`](#f3)</b></sup>, and WikiWeasel 2.0<b>[`[4]`](#f4)</b></sup>.

### Install

Use the following commands to install dependencies:

```
    git clone https://github.com/meyersbs/uncertainty.git
    cd uncertainty/
    pip -r requirements.txt
```

### Usage

```
    # Train the classifier, withholding <ntesting> documents for testing.
    # NOTE: This repository has a pre-trained classifier included.
    python model.py train <ntesting>

    # Classify the given documents. <filename> should be a text file containing
    # one sentence per line.
    # NOTE: A sample test file is included: /test_data.txt
    python model.py classify <filename>

    # Convert the Szeged Uncertainty Corpus (SUC) to the expected format for
    # training.
    # NOTE: This repository has the converted SUC included in /corpus/new/
    python model.py convert
```

### Contact
If you have questions regarding this API, please contact [bsm9339@rit.edu](mailto:bsm9339@rit.edu) (Benjamin Meyers).

For questions regarding the annotated dataset or the theory behind the uncertainty classifier, please contact [szarvas@inf.u-szeged.hu](mailto:szarvas@inf.u-szeged.hu) (György Szarvas), [rfarkas@inf.u-szeged.hu](mailto:rfarkas@inf.u-szeged.hu) (Richárd Farkas), and/or [vinczev@inf.u-szeged.hu](mailto:vinczev@inf.u-szeged.hu) (Veronika Vincze).

### Footnotes

<a name="f1">`[1]`</a> [Vincze, V. (2015). Uncertainty detection in natural language texts (Doctoral dissertation, szte).](http://doktori.bibl.u-szeged.hu/2291/1/Vincze_Veronika_tezis.pdf)
<a name="f2">`[2]`</a> [Vincze, V., Szarvas, G., Farkas, R., Móra, G., & Csirik, J. (2008). The BioScope corpus: biomedical texts annotated for uncertainty, negation and their scopes. BMC bioinformatics, 9(11), S9.](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-9-S11-S9)
<a name="f3">`[3]`</a> [Saurí, R., & Pustejovsky, J. (2009). FactBank: a corpus annotated with event factuality. Language resources and evaluation, 43(3), 227.](https://link.springer.com/article/10.1007/s10579-0$)
<a name="f4">`[4]`</a> [Farkas, R., Vincze, V., Móra, G., Csirik, J., & Szarvas, G. (2010, July). The CoNLL-2010 shared task: learning to detect hedges and their scope in natural language text. In Proceedings of the Fourteenth Conference on Computational Natural Language Learning---Shared Task (pp. 1-12). Association for Computational Linguistics.](https://www.researchgate.net/profile/Domonkos_Tikk2/publication/2347862$)
