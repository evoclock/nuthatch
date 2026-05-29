# Changelog

All notable changes to Nuthatch are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions
follow [Semantic Versioning](https://semver.org/).

## [0.0.3] - 2026-05-29

### Changed

- **Licence changed from MIT + Commercial Attribution Rider to GNU
  Affero General Public License v3 (AGPLv3) plus a Section 7(b)
  author-attribution clause.** AGPLv3 is OSI-approved and delivers
  a structural source-disclosure obligation for any conveyance or
  network use; the §7(b) additional term preserves author attribution
  explicitly. Both obligations are waived under a commercial licence,
  available for for-profit entities and any use in a paid product
  or service.
- The MIT + Rider in 0.0.2 was a social-pressure mechanism with weak
  enforcement teeth and non-OSI status. AGPLv3 + §7(b) is a structural
  mechanism with OSI approval, source-disclosure teeth, and the same
  attribution preservation. This is the standard dual-licence pattern
  used by Nextcloud, Plausible, Cal.com, and iText.
- `pyproject.toml` licence field reverts from `{ file = "LICENSE" }`
  custom rider back to a recognised OSI classifier:
  `License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)`.
- `LICENSE` now contains: project preamble, the verbatim FSF AGPLv3
  text (canonical from <https://www.gnu.org/licenses/agpl-3.0.txt>),
  the §7(b) additional terms section, and a commercial-licence option
  notice pointing to a forthcoming `COMMERCIAL.md` template.
- SPDX headers across 142 source / test / doc / script files updated
  from `LicenseRef-MIT-Commercial-Attribution` to `AGPL-3.0-or-later`.
- README licence badge swapped (from orange "MIT + Commercial
  Attribution" to blue "AGPLv3 + Attribution") and the licence section
  rewritten with plain-English guidance distinguishing open-source use
  (courtesy attribution requested, AGPL terms apply) from commercial
  use (commercial licence required, AGPL waived).
- `docs/Design_Decisions.md` licence subsection rewritten with the
  AGPLv3 rationale; explicit pointer to the 0.0.2 → 0.0.3 transition.
- Graph-visualisation HTML footer string updated.

### Chandra OCR — supported, not shipped

Nuthatch's position on Chandra OCR is explicit: we support Chandra as
a backend because it is an excellent OCR engine for research-grade
work, but we do not ship the chandra-ocr package as a default
dependency. Chandra-ocr's model weights are distributed under the
OpenRAIL-M licence, which carries use restrictions that Nuthatch
users should not inherit silently by installing the base package.

The end state, targeted for an upcoming release:

- `chandra-ocr` lives in an opt-in extra, installed as
  `pip install nuthatch[chandra]`. The user accepts the OpenRAIL-M
  obligations knowingly when they choose to install it.
- The default OCR backend is **tesseract** (Apache 2.0), wrapped via
  pytesseract. Tesseract handles the standard text-extraction cases
  cleanly and inherits no restrictive licence.
- Internal call sites use deferred dynamic imports so the chandra
  backend is only loaded when explicitly selected. A missing chandra
  install raises a clear `ImportError` pointing users to the opt-in
  command.

For the duration of this release, `chandra-ocr` remains in the
mandatory dependency list while the refactor lands. The licence
compatibility holds either way: the chandra-ocr package itself is
Apache 2.0; only the runtime-downloaded weights carry the OpenRAIL-M
terms, and those obligations attach to whoever downloads the
weights.

## [0.0.2] - 2026-05-28

### Changed

- **License changed from Apache 2.0 to MIT with a Commercial Attribution Rider.**
  Non-commercial, academic, and personal use remain unrestricted beyond
  preservation of the copyright notice. Any use by a for-profit entity, or
  any use in a paid product or service, must include attribution to Julen
  Gamboa as the author of the original work with a link to
  `https://github.com/evoclock/nuthatch` in the README or equivalent primary
  documentation. This is intentionally not an OSI-approved license. Trade-off
  knowingly accepted: GitHub's license detection will mark the repository as
  "Other" and corporate adopters that screen for OSI-only licenses will flag
  it for manual review. Full text in `LICENSE`; rationale in
  `docs/Design_Decisions.md`.
- `pyproject.toml` license field updated to `{ file = "LICENSE" }`; PyPI
  classifier changed from `License :: OSI Approved :: Apache Software License`
  to `License :: Other/Proprietary License`.
- SPDX headers across all 142 source, test, doc, and script files updated
  from `Apache-2.0` to `LicenseRef-MIT-Commercial-Attribution`.
- README licence badge and section rewritten.
- Graph-visualisation HTML footer string updated.

## [0.0.1] - 2026-05-24

### Added

- Initial public release under Apache 2.0 (superseded by 0.0.2 license change).
