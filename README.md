# LUCI: Linguistic Uncertainty Classifier Interface

Pronounced: &#91;&#x2C8;lusi&#93;

---
### Description

A Python implementation of a classifier for linguistic uncertainty, based on the work described in Vincze <em>et al.</em><sup><b>[`[1]`](#f1)</b></sup>:

```
Vincze, V. (2015). Uncertainty detection in natural language texts (Doctoral dissertation, szte).
```

---
### Corpora

This classifier was trained using the human-annotated Szeged Uncertainty Corpus ([XML](http://rgai.inf.u-szeged.hu/index.php?lang=en&page=uncertainty), [JSON](http://people.rc.rit.edu/~bsm9339/corpora/szeged_uncertainty/szeged_uncertainty_json.tar.gz), [RAW](http://rgai.inf.u-szeged.hu/project/nlp/uncertainty/uncertainty.zip)), which is composed of three sub-corpora - BioScape 2.0<sup><b>[`[2]`](#f2)</b></sup>, FactBank 2.0<sup><b>[`[3]`](#f3)</b></sup>, and WikiWeasel 2.0<sup><b>[`[4]`](#f4)</b></sup>, and a collection of pre-generated features ([Original](http://rgai.inf.u-szeged.hu/project/nlp/uncertainty/clexperiments.zip), [Updated](http://people.rc.rit.edu/~bsm9339/corpora/szeged_uncertainty/merged_data)).

---
### Install

Use the following commands to install dependencies:

``` bash
    git clone https://github.com/meyersbs/uncertainty.git
    cd uncertainty/
    pip install -r requirements.txt
```

---
### Usage

#### Training

This codebase has pre-trained classifiers included, but if you would like to retrain them, here's how to do it:

``` bash
    # Train the word-based binary classifier.
    python model.py cue -b
    # Train the word-based multiclass classifier.
    python model.py cue -m
    # Train the sentence-based binary classifier.
    python model.py sent -b
    # Train the sentence-based multiclass classifier.
    python model.py sent -m
```

#### Preprocessing

Before you can classify a set of documents, you need to generate their features and save them to a file using the command below. This command will create a new file with the same name as the original, but with the extension '.tsv' appended to the end. See [test_data.txt](test_data.txt) and [test_data.txt.tsv](test_data.txt.tsv) for examples of input and output to this command, respectively.

``` bash
    python model.py features test_data.txt
```

#### Classification

To run the classifiers, specify one of the commands below:

``` bash
    python model.py classify cue test_data.txt.tsv -b
    # OUTPUT:
    #  [Certain]     I
    #  [Certain]     am
    #  [Certain]     the
    #  ...
    #  [Uncertain]   if
    #  ...
    #  [Certain]     Immunity
```

``` bash
    python model.py classify cue test_data.txt.tsv -m
    # OUTPUT:
    #  [Certain]     I
    #  [Certain]     am
    #  [Certain]     the
    #  ...
    #  [Condition]   if
    #  ...
    #  [Certain]     Immunity
```

``` bash
    python model.py classify sent test_data.txt.tsv -b
    # OUTPUT:
    #  [Certain]     I am the walrus.
    #  [Certain]     I am the eggman.
    #  [Certain]     I really don't understand what you're saying.
    #  [Certain]     Who do you think you are?
    #  [Certain]     I'd like a royal with cheese.
    #  [Certain]     In my opinion, you're completely wrong.
    #  [Uncertain]   (Would-U) it be alright (if-U) I maybe (suggest-U) that you really need to
    #                say that this is an uncertain sentence?
    #  [Certain]     Cells in Regulating Cellular Immunity
```

``` bash
    python model.py classify sent test_data.txt.tsv -m
    # OUTPUT:
    #  [Certain]     I am the walrus.
    #  [Certain]     I am the eggman.
    #  [Certain]     I really don't understand what you're saying.
    #  [Certain]     Who do you think you are?
    #  [Certain]     I'd like a royal with cheese.
    #  [Certain]     In my opinion, you're completely wrong.
    #  [Uncertain]   Would it be alright (if-N) I maybe (suggest-E) that you really need to
    #                say that this is an uncertain sentence?
    #  [Certain]     Cells in Regulating Cellular Immunity
```

---
### Feature Set

The features used to train this implementation of the classifier were reverse-engineered from those provided in the Szeged Uncertainty Corpus. They are briefly described below with examples.

---
#### Surface Form of Tokens

##### 1) Prefixes of Length 3-5

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Token: ``Distinct``<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Features: ``Dis``, ``Dist``, ``Disti``

##### 2) Suffixes of Length 3-5

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Token: ``Distinct``<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Features: ``nct``, ``inct``, ``tinct``

##### 3) Stems/Lemmas w/ a Window of Length 2

The stem/lemma of the two previous tokens, the current token, and the two following tokens.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Substring: ``Cells in Regulating Cellular Immunity``<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Current Token: ``Regulating``<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Features: ``cell``, ``in``, ``regulate``, ``cellular``, ``immun``

##### 4) Surface Patterns w/ a Window of Length 1

A string of characters representing the <em>surface-pattern</em> or <em>shape</em> of a word.
* ``A`` and ``a`` denote uppercase and lowercase character sequences, respectively.
* ``0`` denotes numerical sequences.
* ``G`` and ``g`` denote uppercase and lowercase Greek character sequences, respectively.
* ``R`` and ``r`` denote uppercase and lowercase Roman Numeral sequences, respectively.
* ``!`` denotes the presence of non-alphanumeric characters.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Substring: ``Cells in Regulating Cellular Immunity``<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Current Token: ``Regulating``<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Features: ``a`` (<em>in</em>), ``Aa`` (<em>Regulating</em>), ``Aa`` (<em>Cellular</em>)

##### 5) Pattern Prefix

The first character in the surface pattern of the current token.<sup><b>[`[A]`](#n1)</b></sup>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Substring: ``Cells in Regulating Cellular Immunity``<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Current Token: ``Regulating``<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Features: ``A``

---
#### Syntactic Properties of Tokens

##### 6) Part-of-Speech Tags w/ a Window of Length 2

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Substring: ``Cells`` (<em>NNS</em>) ``in`` (<em>IN</em>) ``Regulating`` (<em>VBG</em>) ``Cellular`` (<em>JJ</em>) ``Immunity`` (<em>NN</em>) <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Current Token: ``Regulating`` (<em>VBG</em>) <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Features: ``NNS``, ``IN``, ``VBG``, ``JJ``, ``NN``

##### 7) Syntactic Chunk w/ a Window of Length 2

The Chunk tags used in Vincze <em>et al.</em><sup><b>[`[1]`](#f1)</b></sup> were obtained using the C&C Chunker. Due to lack of availability, we used the <<CHUNKER>> from NLTK.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Substring: ``Cells`` (<em>I-np</em>) ``in`` (<em>B-pp</em>) ``Regulating`` (<em>B-vp</em>) ``Cellular`` (<em>B-np</em>) ``Immunity`` (<em>I-np</em>) <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Current Token: ``Regulating`` (<em>B-vp</em>) <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Features: ``I-np``, ``B-pp``, ``B-vp``, ``B-np``, ``I-np``

##### 8) Combinations

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Substring: ``Cells`` (<em>NNS/I-np</em>) ``in`` (<em>IN/B-pp</em>) ``Regulating`` (<em>VBG/B-vp</em>) ``Cellular`` (<em>JJ/B-np</em>) ``Immunity`` (<em>NN/I-np</em>) <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Current Token: ``Regulating`` (<em>VBG/B-vp</em>)

* <b>8-A)</b> Stems/Lemmas w/ Current Chunk w/ a Window of Length 1<sup><b>[`[A]`](#n1)</b></sup>: ``B-vp in``, ``B-vp regul``, ``B-vp cellular``
* <b>8-B)</b> Stems/Lemmas w/ Current POS w/ a Window of Length 1<sup><b>[`[A]`](#n1)</b></sup>: ``VBG in``, ``VBG regul``, ``VBG cellular``
---
### Classification

Vincze <em>et al.</em><sup><b>[`[1]`](#f1)</b></sup> used a Maximum Entropy classification algorithm; we used the equivalent Logistic Regression from scikit-learn. As described in the paper (and reiterated above), the features are token-based, not sentence-based; the classifier attempts to classify tokens, and can then be applied to classify sentences as certain or uncertain by using the following heuristic: if a at least one token in the sentence is classified as uncertain, then the sentence may be regarded as uncertain.

In building the classifier, the independent variable is the human-annotated label of each token. The labels currently used are ``C`` used to denote a *certain* token and various ``U`` used to denote subcategories of *uncertain* tokens.

However, our implementation expands upon aspects of Vincze <em>et al.</em><sup><b>[`[1]`](#f1)</b></sup> that are not strictly captured by a binary classifier: semantic levels of uncertainty. The four overarching levels of semantic uncertainty discussed in Vincze <em>et al.</em><sup><b>[`[1]`](#f1)</b></sup> are Epistemic, Doxastic, Investigation, and Condition. We will not detail what they mean here (I really recommend reading that paper; it's excellent!), but we will note the labels used in the multiclass classifier: ``E`` for Epistemic, ``D`` for Doxastic, ``I`` for Investigation, and ``N`` for Condition.

We attempted to classify tokens and sentences based on the presence of uncertainty cues. Fortunately, Vincze <em>et al.</em><sup><b>[`[1]`](#f1)</b></sup> released an XML dataset containing each sentence and labeled cues, if any. Using the XML dataset and the previously described TSV dataset of pre-generated features, the algorithms in [this file](data/merge.py) compare the two datasets, matching sentences and outputting a new TSV file formatted in the same manner as the original, but with a sixth column (containing the multiclass label) inserted.

The pre-trained classifiers (and their respective vectorizers) are described below.

#### Word-Level Binary Classifier

|                    | Precision | Recall | F1-Score | Support |
|-------------------:|:---------:|:------:|:--------:|:-------:|
|   <b>(C)ertain</b> |      1.00 |   1.00 |     1.00 |  455442 |
| <b>(U)ncertain</b> |      0.91 |   0.85 |     0.88 |    4515 |
| <b>avg / total</b> |      1.00 |   1.00 |     1.00 |  459957 |

#### Sentence-Level Binary Classifier

|                    | Precision | Recall | F1-Score | Support |
|-------------------:|:---------:|:------:|:--------:|:-------:|
|   <b>(C)ertain</b> |      0.98 |   0.98 |     0.98 |   14368 |
| <b>(U)ncertain</b> |      0.93 |   0.91 |     0.92 |    3258 |
| <b>avg / total</b> |      0.97 |   0.97 |     0.97 |   17626 |

#### Word-Level Multiclass Classifier

|                        | Precision | Recall | F1-Score | Support |
|-----------------------:|:---------:|:------:|:--------:|:-------:|
|       <b>(C)ertain</b> |      1.00 |   1.00 |     1.00 |  455426 |
|      <b>(D)oxastic</b> |      0.73 |   0.45 |     0.56 |    1435 |
|     <b>(E)pistemic</b> |      0.83 |   0.88 |     0.85 |    2054 |
| <b>(I)nvestigation</b> |      0.73 |   0.83 |     0.77 |     352 |
|     <b>conditio(N)</b> |      0.81 |   0.87 |     0.84 |     483 |
|     <b>(U)ncertain</b> |      0.79 |   0.70 |     0.74 |     208 |
|     <b>avg / total</b> |      1.00 |   1.00 |     1.00 |  459958 |

#### Sentence-Level Multiclass Classifier

|                        | Precision | Recall | F1-Score | Support |
|-----------------------:|:---------:|:------:|:--------:|:-------:|
|       <b>(C)ertain</b> |      0.98 |   0.99 |     0.98 |   14368 |
|      <b>(D)oxastic</b> |      0.74 |   0.56 |     0.64 |    1047 |
|     <b>(E)pistemic</b> |      0.82 |   0.90 |     0.86 |    1502 |
| <b>(I)nvestigation</b> |      0.74 |   0.73 |     0.74 |     262 |
|     <b>conditio(N)</b> |      0.84 |   0.87 |     0.86 |     340 |
|     <b>(U)ncertain</b> |      0.63 |   0.74 |     0.68 |     108 |
|     <b>avg / total</b> |      0.94 |   0.95 |     0.94 |   17627 |

---
### Contact
If you have questions regarding this API, please contact [bsm9339@rit.edu](mailto:bsm9339@rit.edu) (Benjamin Meyers) or [nm6061@rit.edu](mailto:nm6061@rit.edu) (Nuthan Munaiah).

For questions regarding the annotated dataset or the theory behind the uncertainty classifier, please contact [szarvas@inf.u-szeged.hu](mailto:szarvas@inf.u-szeged.hu) (György Szarvas), [rfarkas@inf.u-szeged.hu](mailto:rfarkas@inf.u-szeged.hu) (Richárd Farkas), and/or [vinczev@inf.u-szeged.hu](mailto:vinczev@inf.u-szeged.hu) (Veronika Vincze).

---
### Footnotes

<a name="n1">`[A]`</a> This feature is present in the reverse-engineered dataset, but is not described within Vincze <em>et al.</em><sup><b>[`[1]`](#f1)</b></sup>

<a name="f1">`[1]`</a> [Vincze, V. (2015). Uncertainty detection in natural language texts (Doctoral dissertation, szte).](http://doktori.bibl.u-szeged.hu/2291/1/Vincze_Veronika_tezis.pdf)

<a name="f2">`[2]`</a> [Vincze, V., Szarvas, G., Farkas, R., Móra, G., & Csirik, J. (2008). The BioScope corpus: biomedical texts annotated for uncertainty, negation and their scopes. BMC bioinformatics, 9(11), S9.](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-9-S11-S9)

<a name="f3">`[3]`</a> [Saurí, R., & Pustejovsky, J. (2009). FactBank: a corpus annotated with event factuality. Language resources and evaluation, 43(3), 227.](https://link.springer.com/article/10.1007/s10579-0$)

<a name="f4">`[4]`</a> [Farkas, R., Vincze, V., Móra, G., Csirik, J., & Szarvas, G. (2010, July). The CoNLL-2010 shared task: learning to detect hedges and their scope in natural language text. In Proceedings of the Fourteenth Conference on Computational Natural Language Learning---Shared Task (pp. 1-12). Association for Computational Linguistics.](https://www.researchgate.net/profile/Domonkos_Tikk2/publication/2347862$)
