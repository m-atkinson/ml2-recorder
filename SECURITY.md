# Security Policy

## Reporting a vulnerability

If you believe you have found a security vulnerability in this project —
for example, an issue that could leak captured sensor data, allow unauthorized
device access, or be exploited to compromise a host processing recorded
sessions — please report it privately.

**Do not open a public GitHub issue** for security problems.

Instead, open a
[private security advisory](https://github.com/m-atkinson/ml2-pipeline/security/advisories/new)
on this repository, or email the maintainer directly if you cannot use GitHub
Security Advisories.

When reporting, please include:

- A description of the vulnerability and its impact
- Steps to reproduce, including relevant versions (ML2 firmware, MLSDK,
  Python, NDK)
- Any proof-of-concept code (attach, don't paste)
- Whether you're willing to be credited in the fix

## Response timeline

We aim to acknowledge reports within 7 days and provide a status update within
30 days. For confirmed vulnerabilities we will coordinate disclosure with the
reporter.

## Scope

In scope:
- The recorder app source (`app/`)
- The host-side tooling (`tools/`)
- Documented data-format assumptions

Out of scope:
- Vulnerabilities in Magic Leap SDK, Android platform, or VRS/third-party
  dependencies — please report those upstream
- Issues that require physical access to an already-unlocked headset
