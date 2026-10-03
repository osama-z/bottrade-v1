# License scope and third-party dependencies

NeuronTrade's [MIT LICENSE](../LICENSE) applies to the project's own code. It
does not replace licenses for installed dependencies, external data or remotely
loaded assets. The root license text was checked against the
[Open Source Initiative MIT text](https://opensource.org/license/mit); its
copyright holder remains `osama-z` and its year remains 2026.

The tracked-file inventory contains project Python source, tests, documentation,
configuration and deployment templates. No installed dependency trees, model
binaries, downloaded datasets, secret environment files or database files were
found in that inventory. Imported libraries are installed separately by pip.
This inventory does not establish the original authorship of every code line.

## Selected dependency license inventory

These entries were read from the existing Python environment's distribution
metadata and bundled license files during the 2026-10-03 review. They cover
selected direct dependencies, not the complete transitive environment.

| Distribution | Reviewed version | Declared license |
| --- | --- | --- |
| ccxt | 4.5.59 | MIT |
| pandas-ta | 0.4.71b0 | MIT; bundled `licenses/LICENSE` names Kevin Johnson |
| pydantic | 2.13.4 | MIT |
| xgboost | 3.3.0 | Apache-2.0 |
| groq | 1.5.0 | Apache-2.0 |
| python-telegram-bot | 22.8 | LGPL-3.0-only in package metadata; bundled license files include the LGPL and GPL texts |

The inherited [archived HTML illustration](archive/architecture.html) loads
fonts from Google Fonts rather than bundling font files. Its external resources
are not relicensed by this repository's MIT file.

For any distribution that bundles a Python environment, binaries, fonts, model
artifacts or datasets, inspect the exact bundled versions and retain their own
required notices and license files. The metadata inventory here is not a full
redistribution license audit or an assertion that all dependencies use MIT.

The README's educational/paper-only wording describes supported behavior.
It does not amend MIT or impose a noncommercial-use condition.
