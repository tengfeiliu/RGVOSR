# Prompt degradation distribution report

- Input: `datasets\LSDIR_unipercept_raw_cache\valid.cleaned.jsonl`
- Usable records: 20,540; errors: 0
- Condition8 distinctiveness (distribution-only heuristic): **strong**
- Mean valid dimensions: 4.54/8
- Unique canonical combinations: 3,989 (19.4%); top combination share: 5.3%

## Field completeness

- `iqa.distortion_location`: 100.0%
- `iqa.distortion_severity`: 100.0%
- `iqa.distortion_type`: 100.0%
- `iqa.overall_quality`: 100.0%
- `suggestion`: 100.0%

## Condition dimensions

| Source | Dimension | Valid | Mean | Std | P10 | P50 | P90 | Level entropy | Top level | Top share | Spearman vs physical |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| condition8 | blur | 98.3% | 0.628 | 0.184 | 0.500 | 0.550 | 0.950 | 0.644 | moderate | 0.485 | 0.108 |
| condition8 | noise | 99.1% | 0.634 | 0.191 | 0.350 | 0.700 | 0.900 | 0.730 | moderate | 0.365 | 0.215 |
| condition8 | compression | 78.3% | 0.543 | 0.182 | 0.250 | 0.500 | 0.750 | 0.647 | moderate | 0.586 | NA |
| condition8 | ringing_aliasing | 10.6% | 0.466 | 0.199 | 0.250 | 0.500 | 0.500 | 0.578 | moderate | 0.701 | 0.048 |
| condition8 | texture_loss | 96.0% | 0.658 | 0.193 | 0.500 | 0.700 | 0.950 | 0.688 | severe | 0.384 | 0.223 |
| condition8 | photometric | 48.6% | 0.506 | 0.240 | 0.250 | 0.500 | 0.750 | 0.790 | moderate | 0.392 | NA |
| condition8 | structure_risk | 20.3% | 0.664 | 0.251 | 0.500 | 0.750 | 1.000 | 0.750 | moderate | 0.368 | 0.035 |
| condition8 | hallucination_risk | 2.7% | 0.490 | 0.122 | 0.500 | 0.500 | 0.500 | 0.226 | moderate | 0.921 | 0.076 |
| iqa | blur | 98.3% | 0.628 | 0.197 | 0.500 | 0.500 | 1.000 | 0.645 | moderate | 0.485 | 0.090 |
| iqa | noise | 98.6% | 0.643 | 0.228 | 0.250 | 0.750 | 1.000 | 0.736 | moderate | 0.365 | 0.192 |
| iqa | compression | 77.9% | 0.551 | 0.197 | 0.250 | 0.500 | 0.750 | 0.649 | moderate | 0.589 | NA |
| iqa | ringing_aliasing | 8.9% | 0.526 | 0.150 | 0.500 | 0.500 | 0.750 | 0.384 | moderate | 0.830 | 0.013 |
| iqa | texture_loss | 95.0% | 0.678 | 0.211 | 0.500 | 0.750 | 1.000 | 0.684 | moderate | 0.374 | 0.245 |
| iqa | photometric | 47.2% | 0.519 | 0.243 | 0.250 | 0.500 | 0.750 | 0.777 | moderate | 0.403 | NA |
| iqa | structure_risk | 20.0% | 0.664 | 0.254 | 0.500 | 0.750 | 1.000 | 0.755 | moderate | 0.362 | 0.026 |
| iqa | hallucination_risk | 0.1% | 0.656 | 0.287 | 0.250 | 0.750 | 1.000 | 0.906 | moderate | 0.292 | 0.260 |
| suggestion | blur | 54.6% | 0.384 | 0.077 | 0.300 | 0.450 | 0.450 | 0.520 | moderate | 0.566 | 0.109 |
| suggestion | noise | 94.2% | 0.366 | 0.084 | 0.300 | 0.300 | 0.450 | 0.576 | mild | 0.514 | 0.121 |
| suggestion | compression | 39.2% | 0.317 | 0.076 | 0.300 | 0.300 | 0.450 | 0.536 | mild | 0.738 | NA |
| suggestion | ringing_aliasing | 2.4% | 0.182 | 0.147 | 0.000 | 0.300 | 0.300 | 0.604 | mild | 0.575 | 0.025 |
| suggestion | texture_loss | 54.4% | 0.333 | 0.094 | 0.150 | 0.300 | 0.450 | 0.675 | mild | 0.560 | -0.019 |
| suggestion | photometric | 8.7% | 0.222 | 0.084 | 0.150 | 0.150 | 0.300 | 0.732 | subtle | 0.554 | NA |
| suggestion | structure_risk | 2.0% | 0.502 | 0.093 | 0.500 | 0.500 | 0.500 | 0.453 | moderate | 0.862 | 0.027 |
| suggestion | hallucination_risk | 2.6% | 0.482 | 0.105 | 0.500 | 0.500 | 0.500 | 0.215 | moderate | 0.947 | 0.087 |

## IQA and suggestion contribution

| Dimension | Both | IQA only | Suggestion only | Neither | Suggestion changes combined when both | Mean absolute change |
|---|---:|---:|---:|---:|---:|---:|
| blur | 54.6% | 43.7% | 0.0% | 1.7% | 42.9% | 0.023 |
| noise | 93.6% | 5.0% | 0.5% | 0.9% | 58.0% | 0.037 |
| compression | 38.8% | 39.1% | 0.4% | 21.7% | 45.8% | 0.027 |
| ringing_aliasing | 0.8% | 8.2% | 1.7% | 89.4% | 41.5% | 0.034 |
| texture_loss | 53.4% | 41.7% | 1.0% | 4.0% | 65.7% | 0.044 |
| photometric | 7.2% | 39.9% | 1.4% | 51.4% | 63.0% | 0.042 |
| structure_risk | 1.8% | 18.2% | 0.3% | 79.7% | 8.7% | 0.025 |
| hallucination_risk | 0.0% | 0.1% | 2.6% | 97.3% | 28.6% | 0.107 |

## Text diversity

- **iqa**: unique 100.0%; top-1 0.0%; entropy 1.000; words p50=161; random-pair Jaccard mean=0.220, p90=0.290.
- **suggestion**: unique 86.3%; top-1 0.4%; entropy 0.982; words p50=22; random-pair Jaccard mean=0.316, p90=0.452.
- **condition8_text**: unique 19.4%; top-1 5.3%; entropy 0.792; words p50=18; random-pair Jaccard mean=0.542, p90=0.875.

## Interpretation guardrail

The heuristic rating only detects distribution collapse. Treat the prompt as genuinely discriminative only when dimensions also align with physical degradation and matched-vs-shuffled prompt inference produces a significant quality advantage.
