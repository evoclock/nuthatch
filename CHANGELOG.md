# Changelog

All notable changes to Nuthatch are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions
follow [Semantic Versioning](https://semver.org/).

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
