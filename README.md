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

This classifier was trained using the human-annotated Szeged Uncertainty Corpus ([XML](http://rgai.inf.u-szeged.hu/index.php?lang=en&page=uncertainty), [JSON](http://people.rc.rit.edu/~bsm9339/corpora/szeged_uncertainty/szeged_uncertainty_json.tar.gz), [RAW](http://rgai.inf.u-szeged.hu/project/nlp/uncertainty/clexperiments.zip)), which is composed of three sub-corpora - BioScape 2.0<sup><b>[`[2]`](#f2)</b></sup>, FactBank 2.0<sup><b>[`[3]`](#f3)</b></sup>, and WikiWeasel 2.0<sup><b>[`[4]`](#f4)</b></sup>.

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

``` bash
    # Train the word-based classifier.
    # NOTE: This repository has a pre-trained classifier included.
    python model.py cue

    # Train the sentence-based classifier.
    # NOTE: This repository has a pre-trained classifier included.
    python model.py sentence

    # Classify the given documents.
    # NOTE: A sample test file is included: /test_data.txt
    python model.py classify [cue|sentence] <filename>
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

##### 4) Pattern Prefixes w/ a Window of Length 1

A string of characters representing the <em>surface-pattern</em> or <em>shape</em> of a word.
* ``A`` and ``a`` denote uppercase and lowercase character sequences, respectively.
* ``0`` denotes numerical sequences.
* ``G`` and ``g`` denote uppercase and lowercase Greek character sequences, respectively.
* ``R`` and ``r`` denote uppercase and lowercase Roman Numeral sequences, respectively.
* ``!`` denotes the presence of non-alphanumeric characters.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Substring: ``Cells in Regulating Cellular Immunity``<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Current Token: ``Regulating``<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Features: ``a`` (<em>in</em>), ``Aa`` (<em>Regulating</em>), ``Aa`` (<em>Cellular</em>)

---
#### Syntactic Properties of Tokens

##### 5) Part-of-Speech Tags w/ a Window of Length 2

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Substring: ``Cells`` (<em>NNS</em>) ``in`` (<em>IN</em>) ``Regulating`` (<em>VBG</em>) ``Cellular`` (<em>JJ</em>) ``Immunity`` (<em>NN</em>) <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Current Token: ``Regulating`` (<em>VBG</em>) <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Features: ``NNS``, ``IN``, ``VBG``, ``JJ``, ``NN``

##### 6) Syntactic Chunk w/ a Window of Length 2

The Chunk tags used in Vincze <em>et al.</em><sup><b>[`[1]`](#f1)</b></sup> were obtained using the C&C Chunker. Due to lack of availability, we used the <<CHUNKER>> from NLTK.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Substring: ``Cells`` (<em>I-np</em>) ``in`` (<em>B-pp</em>) ``Regulating`` (<em>B-vp</em>) ``Cellular`` (<em>B-np</em>) ``Immunity`` (<em>I-np</em>) <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Current Token: ``Regulating`` (<em>B-vp</em>) <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Features: ``I-np``, ``B-pp``, ``B-vp``, ``B-np``, ``I-np``

##### 7) Combinations

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Substring: ``Cells`` (<em>NNS/I-np</em>) ``in`` (<em>IN/B-pp</em>) ``Regulating`` (<em>VBG/B-vp</em>) ``Cellular`` (<em>JJ/B-np</em>) ``Immunity`` (<em>NN/I-np</em>) <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Current Token: ``Regulating`` (<em>VBG/B-vp</em>)
* <b>Stem &amp; Chunk: </b> ``regul B-vp``
* <b>Current Chunk w/ Neighboring Stems/Lemmas: </b> ``B-vp in``, ``B-vp cellular``
* <b>Current POS w/ Neighboring Stems/Lemmas:<sup><b>[`[A]`](#n1)</b></sup> </b> ``VBG in``, ``VBG cellular``
* <b>Current Stem/Lemma w/ Current POS or Current Chunk:<sup><b>[`[A]`](#n1)</b></sup> </b> ``regul B-vp``, ``regul VBG``
* <b>Current Stem/Lemma w/ Neighboring POS:<sup><b>[`[B]`](#n2)</b></sup> </b>

---
### Contact
If you have questions regarding this API, please contact [bsm9339@rit.edu](mailto:bsm9339@rit.edu) (Benjamin Meyers) or [nm6061@rit.edu](mailto:nm6061@rit.edu) (Nuthan Munaiah).

For questions regarding the annotated dataset or the theory behind the uncertainty classifier, please contact [szarvas@inf.u-szeged.hu](mailto:szarvas@inf.u-szeged.hu) (György Szarvas), [rfarkas@inf.u-szeged.hu](mailto:rfarkas@inf.u-szeged.hu) (Richárd Farkas), and/or [vinczev@inf.u-szeged.hu](mailto:vinczev@inf.u-szeged.hu) (Veronika Vincze).

---
### Footnotes

<a name="n1">`[A]`</a> This feature is present in the reverse-engineered dataset, but is not described within Vincze <em>et al.</em><sup><b>[`[1]`](#f1)</b></sup>

<a name="n2">`[B]`</a> This feature is described in Vincze <em>et al.</em><sup><b>[`[1]`](#f1)</b></sup>, but is not present in the dataset.

<a name="f1">`[1]`</a> [Vincze, V. (2015). Uncertainty detection in natural language texts (Doctoral dissertation, szte).](http://doktori.bibl.u-szeged.hu/2291/1/Vincze_Veronika_tezis.pdf)

<a name="f2">`[2]`</a> [Vincze, V., Szarvas, G., Farkas, R., Móra, G., & Csirik, J. (2008). The BioScope corpus: biomedical texts annotated for uncertainty, negation and their scopes. BMC bioinformatics, 9(11), S9.](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-9-S11-S9)

<a name="f3">`[3]`</a> [Saurí, R., & Pustejovsky, J. (2009). FactBank: a corpus annotated with event factuality. Language resources and evaluation, 43(3), 227.](https://link.springer.com/article/10.1007/s10579-0$)

<a name="f4">`[4]`</a> [Farkas, R., Vincze, V., Móra, G., Csirik, J., & Szarvas, G. (2010, July). The CoNLL-2010 shared task: learning to detect hedges and their scope in natural language text. In Proceedings of the Fourteenth Conference on Computational Natural Language Learning---Shared Task (pp. 1-12). Association for Computational Linguistics.](https://www.researchgate.net/profile/Domonkos_Tikk2/publication/2347862$)
