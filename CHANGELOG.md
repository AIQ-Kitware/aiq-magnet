# Changelog

This changelog follows the specifications detailed in: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html), although we have not yet reached a `1.0.0` release.

## Version 0.0.1 -- Released 2025-10-24

* Initial release; includes minimum working implementations for:
  * Evaluation card specification and evaluation
  * [HELM](https://github.com/stanford-crfm/helm) benchmark output downloading and data interfaces
  * Benchmark `Predictor` class (with random, and perturbation based examples)
  * Utility for "offline" HELM perturbation application
  * Ad-hoc inference and direct model access through HELM
  * Command-line wrapper for `helm-run` supporting runs against "offline" dataset instances
