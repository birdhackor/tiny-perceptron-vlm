# Native CI failure diagnostics

The pytest command now writes JUnit XML and an if-failure step preserves failed/error cases in the public job summary. Original failing test exit codes remain failures. The synthetic fixture checks XML rendering and escaping only; it does not diagnose or fix the original Windows failures. Book source, model runtime, tests and numerical claims are unchanged by this workflow diagnostic addition.
