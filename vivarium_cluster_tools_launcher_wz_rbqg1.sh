
    export VIVARIUM_LOGGING_DIRECTORY=/mnt/team/simulation_science/costeffectiveness/results/vivarium_gates_nutrition_optimization/model4.1/ethiopia/2023_08_07_09_07_19/logs/2023_08_07_09_07_19_run/worker_logs
    export PYTHONPATH=/mnt/team/simulation_science/costeffectiveness/results/vivarium_gates_nutrition_optimization/model4.1/ethiopia/2023_08_07_09_07_19:$PYTHONPATH

    /ihme/homes/pnast/miniconda3/envs/vg_nutri/bin/rq worker -c settings         --name ${SLURM_ARRAY_JOB_ID}.${SLURM_ARRAY_TASK_ID}         --burst         -w "vivarium_cluster_tools.psimulate.worker.core._ResilientWorker"         --exception-handler "vivarium_cluster_tools.psimulate.worker.core._retry_handler" vivarium

    