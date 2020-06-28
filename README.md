DeCOFre
=======

[![Build Status](https://github.com/LoicGrobol/decofre/workflows/CI/badge.svg)](https://github.com/LoicGrobol/decofre/actions?query=workflow%3ACI)
[![PyPI](https://img.shields.io/pypi/v/decofre.svg)](https://pypi.org/project/decofre)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**De**tecting **C**oreferences for **O**ral **Fre**nch<a href="#foottext1" id="footnote1">¹</a>.

This was developed for application on spoken French as part of my PhD thesis, it is relatively easy
to apply it to other languages and genres, though.

## Installation

1. Install with pip

   ```console
   python -m pip install --pre decofre
   ```

2. Install the additional dependencies

   ```console
   python -m spacy download fr_core_news_lg
   ```

## Running a pretrained model

Use `decofre-infer`, e.g.

```console
decofre-infer path/to/detector.model path/to/coref.model path/to/raw_text.txt
```

Its output is still rather crude and mostly meant for demonstration purpose.

## Training a model

### Downloading ANCOR

So far the only corpus we officially support (more in preparation, along with an easier bootstrap
procedure).

-  Clone this repo `git clone https://github.com/LoicGrobol/decofre && cd decofre`
- Ensure you are in an environment where DeCOFre has been installed (to be sure that all the
  dependencies are correct)
- Run the bootstrap script `python -m doit run -f datasets/ancor/ancor.py`

### Actual training

Use `decofre-train`, e.g.

```console
decofre-train --config tests/sanity-check.jsonnet --model-config decofre/models/default.jsonnet --out-dir /path/to/an/output/directory
```

This will put a `detector.model` and a `coref.model` files in the selected output directory, that
you can then load in `decofre-infer`.

The `sanity-check` trainig config is, well, *a sanity check*, meant to see if DeCOFre actually
 runs in your environment and uses a tiny training set to make it fast. The resulting models
will therefore be awful. This is normal, don't be alarmed.

You probably want to substitute the config files for your own, see also ANCOR config files in
[datasets/ancor/](datasets/ancor). The config files are not really documented right now, but you can
take inspiration from the provided examples. See also `decofre-train --help` for other options.

**This is by no mean fast, you have been warned.**

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
