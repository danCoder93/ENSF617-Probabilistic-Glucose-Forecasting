# Static Dependency Graph Summary

## Production Overview

- Production modules: 64
- Production dependency edges: 175
- Cross-package edges: 27
- Test-to-production edges: 91

## Cycle Status

- Cycles detected: none

## Highest Fan-In

- `config`: 23
- `evaluation`: 8
- `reporting.types`: 8
- `config.data`: 7
- `config.runtime`: 6
- `data.schema`: 6
- `observability.utils`: 6
- `config.observability`: 5

## Highest Fan-Out

- `workflows.training`: 12
- `data.datamodule`: 9
- `config`: 7
- `models.fused_model`: 7
- `observability.callbacks`: 6
- `reporting`: 6
- `workflows.cli`: 6
- `environment`: 5

## Package Dependencies

- `config` -> `utils`
- `data` -> `config`
- `data` -> `utils`
- `defaults` -> `config`
- `environment` -> `config`
- `main` -> `defaults`
- `main` -> `workflows`
- `models` -> `config`
- `models` -> `evaluation`
- `models` -> `observability`
- `observability` -> `config`
- `observability` -> `reporting`
- `reporting` -> `config`
- `reporting` -> `evaluation`
- `reporting` -> `observability`
- `train` -> `config`
- `train` -> `data`
- `train` -> `environment`
- `train` -> `models`
- `train` -> `observability`
- `workflows` -> `config`
- `workflows` -> `data`
- `workflows` -> `defaults`
- `workflows` -> `environment`
- `workflows` -> `evaluation`
- `workflows` -> `reporting`
- `workflows` -> `train`
