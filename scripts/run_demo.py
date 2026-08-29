import json
import os
import sys
from pathlib import Path


project=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(project/"src"))
os.environ["PRIEVO_MODE"]="demo"

from prievo_agent.application.run_facade import RunApplicationFacade
from prievo_agent.infrastructure.local_runtime import LocalRuntimeComposition


root=project/".demo-runtime"
composition=LocalRuntimeComposition(root,project)
facade=RunApplicationFacade(composition.open_store,composition.execute,True,0,
                            composition.dataset_registry)
try:
    created=facade.create_run("xgboost-Covtype",generations=2,population_size=3,
                              candidate_budget=3,random_seed=7)
    facade.shutdown()
    print(json.dumps(facade.get_run(created["run_id"]),ensure_ascii=False,indent=2))
finally:
    facade.shutdown()
