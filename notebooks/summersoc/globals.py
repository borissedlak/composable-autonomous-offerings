from pathlib import Path

DIR_METRICS = Path('../../statics/agent_experience')
PATH_METRICS_DEMO_EXPLORE = DIR_METRICS / 'metrics_DEMO_EXPLORE.csv'
PATH_METRICS_DEMO_EXPLOIT = DIR_METRICS / 'metrics_DEMO_OPERATE.csv'

SPLIT_DATA_INTO_X_PARTS = 60
ITERATE_THROUGH_X_PARTS = 60
RUNS_PER_CONFIG = 50

DIR_VAR_DUMPS = Path('../../statics/var_dumps')
PATH_MODEL_LIST = DIR_VAR_DUMPS / '2_analysis_models.joblib'

DIR_CANDIDATES = Path('../../statics/candidates')