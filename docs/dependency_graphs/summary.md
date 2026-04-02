# Static Dependency Graph Summary

## Production Overview

- Production modules: 51
- Production dependency edges: 137
- Cross-package edges: 23
- Test-to-production edges: 81

## Cycle Status

- Cycles detected: none

## Highest Fan-In

- `config`: 20
- `config.data`: 6
- `config.runtime`: 5
- `utils.tft_utils`: 5
- `data.schema`: 5
- `environment.types`: 5
- `observability.logging_utils`: 5
- `observability.utils`: 5

## Highest Fan-Out

- `workflows.training`: 9
- `data.datamodule`: 8
- `config`: 6
- `models.fused_model`: 6
- `observability.callbacks`: 6
- `workflows.cli`: 6
- `environment`: 5
- `evaluation`: 5

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
- `observability` -> `config`
- `observability` -> `evaluation`
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
- `workflows` -> `observability`
- `workflows` -> `train`
