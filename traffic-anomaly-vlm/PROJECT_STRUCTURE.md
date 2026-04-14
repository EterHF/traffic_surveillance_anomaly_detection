# PROJECT_STRUCTURE

```text
traffic-anomaly-vlm/
├─ README.md
├─ PROJECT_STRUCTURE.md
├─ requirements.txt
├─ pyproject.toml
├─ .env.example
├─ configs/
├─ scripts/
├─ src/
│  ├─ main.py
│  ├─ schemas.py
│  ├─ settings.py
│  ├─ core/
│  ├─ perception/
│  ├─ features/
│  ├─ triggers/
│  ├─ proposals/
│  ├─ evidence/
│  ├─ vlm/
│  ├─ pipeline/
│  └─ eval/
├─ tests/
└─ outputs/
```

## Core Pipeline

`perception -> features -> triggers -> proposals -> evidence -> vlm`

## Key Contracts

- Tracking results are normalized as `TrackObject`.
- Window representation is `WindowFeature`.
- Event candidates are `EventProposal`.
- Evidence is `EvidencePack`.
- Model verdict is `VLMResult`.
- Final output is `FinalResult`.
