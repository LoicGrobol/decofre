DeCOFre
========================================

**De**tecting **C**oreferences for **O**ral **Fre**nch<a href="#foottext1" id="footnote1">¹</a>.

This was developed for application on spoken French as part of my PhD thesis, it is relatively easy
to apply it to other languages and genres, though.

## Installation

1. Install with pip (Using the `--pre` flag until allennlp release a stable 0.9.1, at which point
   we'll also put a release on pypi)

   ```console
   python -m pip install --pre git+https://github.com/LoicGrobol/decofre
   ```

2. Install the additional dependencies

   ```bash
   python -m spacy download fr_core_news_sm
   ```

## Running

1. Produce the intermediate representation for mention detection, run `python
   decofre/formats/raw_text.py monfichier.txt spans.json`
2. Run the mention detector `python decofre/detmentions.py detector.model spans.json detected.json`

## Citation

```bibtex
@inproceedings{grobol2019NeuralCoreferenceResolution,
  author = {Grobol, Loïc},
  date = {2019-06},
  eventtitle = {Proceedings of the {{Second Workshop}} on {{Computational Models}} of {{Reference}}, {{Anaphora}} and {{Coreference}}},
  pages = {8-14},
  title = {Neural {{Coreference Resolution}} with {{Limited Lexical Context}} and {{Explicit Mention Detection}} for {{Oral French}}},
  url = {https://www.aclweb.org/anthology/papers/W/W19/W19-2802/},
  urldate = {2019-06-24}
}
```

---
<sub>1. Let me know if you think of a better name. <a href="#footnote1" id="foottext1">↑</a></sub>

## Licence

Unless otherwise specified (see <a href="#licence-exceptions">below</a>), the following licence (the
so-called “MIT License”) applies to all the files in this repository.
See also [LICENCE.md](LICENCE.md).

```text
Copyright 2020 Loïc Grobol <loic.grobol@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

### <a id="licence-exceptions">Licence exceptions</a>

The files listed here are distributed under different terms.
When redistributing or building upon this work, you have to comply with their respective
restrictions separately.

#### ANCOR

[![CC-BY-NC-SA-4.0 badge](https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png)](http://creativecommons.org/licenses/by-nc-sa/4.0/)

The following files are derived from [the ANCOR
Corpus](https://www.ortolang.fr/market/corpora/ortolang-000903) and distributed under a [Creative
Commons Attribution-NonCommercial-ShareAlike 4.0 International
License](http://creativecommons.org/licenses/by-nc-sa/4.0/).

- [`tests/fixtures/antecedents.json`](tests/fixtures/antecedents.json)
- [`tests/fixtures/mentions.json`](tests/fixtures/mentions.json)

- **Authors** Judith Muzerelle, Anaïs Lefeuvre, Aurore Pelletier, Emmanuel Schang, Jean-Yves Antoine
- **Origin** <https://www.ortolang.fr/market/corpora/ortolang-000903>
